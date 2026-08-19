"""The binder the receptionist opens only when asked.

The profile holds what a receptionist knows by heart — hours, staff, services,
prices, and crucially what the business does NOT offer. That rides in the prompt
on every call because you never want a lookup, or a miss, on "what time do you
open?".

This holds the other kind of knowledge: the paragraph-long, occasionally-asked
detail. Parking. Insurance and reimbursement. How a prescription refill works.
Accessibility. Too bulky to carry on every turn, too rarely needed to justify it.

    python ingest.py knowledge/clinic_policies.md --profile profiles/clinic.json

What this deliberately does NOT hold:

  - Anything the business does not offer. You cannot retrieve an absence: if a
    search for "MRI" comes back empty, nothing distinguishes "we don't do MRI"
    from "I failed to find the MRI section". One of those is a confident answer
    and the other is a guess, so `not_offered` stays in the prompt for good.
  - Staff and service lists. "Which doctors treat skin problems?" needs ALL the
    dermatologists; nearest-neighbour search returns the closest few. A catalogue
    wants a query, not a similarity score.
  - Anything live. Availability and bookings are tool calls, and always will be.

Rows are scoped by `business_id` and every query filters on it. Not by
convention — the search function takes it as a required argument, because the
day there are two clients, a forgotten filter reads one clinic's documents out
on another clinic's call.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from openai import OpenAI

from db import connect

# text-embedding-3-small: 1536 dimensions, roughly ₹2 per million tokens, and
# well past good enough for policy prose. Reuses OPENAI_API_KEY — no new vendor.
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536

# Aim per chunk. Small enough that a hit is mostly the answer rather than the
# page it sits on; large enough to keep a policy paragraph whole. Measured in
# characters because we chunk before we tokenise — ~4 chars a token.
TARGET_CHARS = 1000
MAX_CHARS = 1600

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS knowledge (
    id          BIGSERIAL PRIMARY KEY,
    business_id TEXT NOT NULL,
    business    TEXT,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR({EMBED_DIMENSIONS}),
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS knowledge_business_idx ON knowledge (business_id);
CREATE INDEX IF NOT EXISTS knowledge_source_idx   ON knowledge (business_id, source);
"""


def init_knowledge():
    """Create the table. Safe to run repeatedly; mirrors db.init_db()."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()
    logger.info("knowledge table ready")


def _client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set — cannot embed.")
    return OpenAI(api_key=key)


def chunk(text):
    """Split a document into retrievable pieces, respecting its own structure.

    Splits on blank lines first, because policy documents are written in
    paragraphs and a paragraph is usually exactly the answer to one question.
    Adjacent paragraphs are packed together up to TARGET_CHARS so a two-line
    paragraph does not become a chunk with no context around it.

    A markdown heading always starts a new chunk and is carried into it, so a
    retrieved piece arrives with the section it belongs to — "Parking" attached
    to the parking rules is the difference between a usable hit and a fragment.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks, current, heading = [], [], ""

    def flush():
        if current:
            body = "\n\n".join(current)
            chunks.append(f"{heading}\n\n{body}" if heading and not body.startswith(heading) else body)
            current.clear()

    for block in blocks:
        if re.match(r"^#{1,6}\s", block):
            flush()
            heading = block.lstrip("#").strip()
            continue
        # A single block over the limit is split on sentence ends rather than
        # mid-word, which would strand half a sentence in each neighbour.
        if len(block) > MAX_CHARS:
            flush()
            for piece in re.findall(r".{1,%d}(?:\.\s|$)" % TARGET_CHARS, block, re.S):
                if piece.strip():
                    current.append(piece.strip())
                    flush()
            continue
        if sum(len(c) for c in current) + len(block) > TARGET_CHARS:
            flush()
        current.append(block)
    flush()
    return [c for c in chunks if c.strip()]


def embed(texts):
    """Vectors for a list of strings, in one API call."""
    if not texts:
        return []
    response = _client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _as_vector(values):
    """psycopg has no vector adapter here, and pgvector accepts its text form.

    Avoids a dependency for what is a bracketed comma-separated list.
    """
    return "[" + ",".join(f"{v:.7f}" for v in values) + "]"


def store(business_id, source, chunks, vectors, business_name=None):
    """Replace everything previously ingested from `source` for this business.

    Replace rather than append: re-uploading an edited policy must not leave the
    old wording in the store answering alongside the new. Delete and insert run
    in one transaction so a failed ingestion cannot empty a business's knowledge
    and leave it with nothing.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM knowledge WHERE business_id = %s AND source = %s",
            (business_id, source),
        )
        removed = cur.rowcount
        for index, (content, vector) in enumerate(zip(chunks, vectors)):
            cur.execute(
                """
                INSERT INTO knowledge
                    (business_id, business, source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (business_id, business_name, source, index, content, _as_vector(vector)),
            )
        conn.commit()
    logger.info(f"{source}: stored {len(chunks)} chunks for {business_id}"
                + (f" (replaced {removed})" if removed else ""))
    return len(chunks)


def search(business_id, question, limit=3, max_distance=0.70):
    """The closest chunks this business has to the question.

    `business_id` is positional and required — scoping is not something a caller
    can forget.

    `max_distance` filters out weak matches. Cosine distance runs 0 (identical)
    to 2 (opposite). 0.70 is not a guess: measured against the clinic document,
    every question the store could genuinely answer came back at 0.31-0.68,
    while questions about things the clinic does NOT do landed at 0.72-0.75 —

        "do you do MRI scans?"    -> "Test results from outside labs"  0.719
        "is there a cardiologist?" -> "Repeat and follow-up visits"    0.746

    Both would have been handed to the model as if relevant. The first chunk
    says scans are done at external centres, which a model could easily relay as
    "yes, we arrange scans" when the profile's not_offered list says no
    radiology of any kind. Nearest is not the same as relevant, and the gap
    between a real answer and the store's closest row is where that gets
    decided.

    Returning nothing is a good outcome. The bot already handles not knowing —
    it says so and offers to take a message.
    """
    if not question.strip():
        return []

    # Embedding is an HTTPS round trip to OpenAI; opening the Postgres
    # connection is a TLS handshake to Supabase. Measured from here they cost
    # roughly 750ms and 700ms, and they do not depend on each other — done in
    # sequence that is a second and a half of dead air on a phone call, done
    # together it is whichever is slower.
    with ThreadPoolExecutor(max_workers=2) as pool:
        vector_job = pool.submit(embed, [question])
        conn_job = pool.submit(connect)
        try:
            vector = _as_vector(vector_job.result()[0])
        except Exception:
            # The connection was still asked for; close it rather than leak it.
            try:
                conn_job.result().close()
            except Exception:
                pass
            raise
        conn = conn_job.result()

    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT source, content, embedding <=> %s::vector AS distance
            FROM knowledge
            WHERE business_id = %s
            ORDER BY distance
            LIMIT %s
            """,
            (vector, business_id, limit),
        )
        rows = cur.fetchall()
    return [
        {"source": source, "content": content, "distance": float(distance)}
        for source, content, distance in rows
        if distance <= max_distance
    ]


def sources(business_id):
    """What documents this business has, and how many chunks each became."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT source, COUNT(*), MAX(created_at)
            FROM knowledge WHERE business_id = %s
            GROUP BY source ORDER BY source
            """,
            (business_id,),
        )
        return [
            {"source": s, "chunks": n, "ingested": t.strftime("%Y-%m-%d %H:%M")}
            for s, n, t in cur.fetchall()
        ]


def forget(business_id, source):
    """Remove one document. Returns how many chunks went."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM knowledge WHERE business_id = %s AND source = %s",
            (business_id, source),
        )
        removed = cur.rowcount
        conn.commit()
    logger.info(f"{source}: removed {removed} chunks from {business_id}")
    return removed
