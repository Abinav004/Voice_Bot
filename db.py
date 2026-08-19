"""Where finished calls are kept.

One table, `conversations` — a row per call, holding the raw message list exactly
as the LLM saw it, plus labels for which business and stack served it. The summary
columns start empty; summarize.py fills them in later.

Storing the transcript verbatim rather than a digest is deliberate: every bug in
this project so far was found by reading a real transcript, and a summary written
at save time would have thrown away the evidence.

    python db.py            # create the table
    python db.py --check    # show what's in there
"""

import json
import os
import sys
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv
from loguru import logger
from psycopg.types.json import Jsonb

load_dotenv(override=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id              BIGSERIAL PRIMARY KEY,
    business        TEXT,
    profile_path    TEXT,
    stack           TEXT,          -- indian / english
    llm_model       TEXT,
    started_at      TIMESTAMPTZ,
    ended_at        TIMESTAMPTZ    DEFAULT now(),
    turns           INTEGER,
    messages        JSONB NOT NULL,

    -- filled in later by summarize.py
    summary         TEXT,
    booking_made    BOOLEAN,
    booking_details JSONB,
    summarized_at   TIMESTAMPTZ
);

-- summarize.py scans for rows still needing work, so index exactly that.
CREATE INDEX IF NOT EXISTS conversations_unsummarized
    ON conversations (id) WHERE summarized_at IS NULL;
"""


def connect():
    """Open a Postgres connection to Supabase.

    Accepts either name — DATABASE_URL is the conventional one, but SUPABASE_URL
    is what got used here. Both must hold the *Postgres* connection string, not
    the https://xxx.supabase.co project URL, which is the REST endpoint and has
    no password or port in it.
    """
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_URL")
    if not url:
        raise SystemExit(
            "No database URL. Supabase → Connect → Session pooler, then put it in "
            ".env as DATABASE_URL="
        )
    if not url.startswith("postgres"):
        raise SystemExit(
            f"That looks like the project URL ({url.split('://')[0]}://…), not the "
            "database one. Supabase → Connect → Session pooler → the postgresql:// URI"
        )
    return psycopg.connect(url)


def init_db():
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()
    logger.info("conversations table ready")


def save_conversation(messages, *, business=None, profile_path=None, stack=None,
                      llm_model=None, started_at=None):
    """Store one finished call. Returns its id, or None if it could not be saved.

    Never raises. A call that has already happened must not be lost — or worse,
    turn into an exception — because the database was unreachable.
    """
    # The system prompt is ~2k tokens of the business profile, identical on every
    # row and reconstructable from the profile. Drop it and keep the conversation.
    spoken = [m for m in messages if m.get("role") != "system"]
    if not spoken:
        return None

    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations
                    (business, profile_path, stack, llm_model, started_at, turns, messages)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (business, profile_path, stack, llm_model, started_at,
                 sum(1 for m in spoken if m.get("role") == "user"), Jsonb(spoken)),
            )
            row_id = cur.fetchone()[0]
            conn.commit()
        logger.info(f"Saved conversation #{row_id} ({len(spoken)} messages)")
        return row_id
    except Exception as e:
        logger.warning(f"Could not save conversation (call itself was fine): {e}")
        return None


def unsummarized(limit=50):
    """Calls still waiting for a summary, oldest first."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, business, messages FROM conversations
            WHERE summarized_at IS NULL
            ORDER BY id
            LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()


def save_summary(row_id, summary, booking_made, booking_details=None):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE conversations
               SET summary = %s, booking_made = %s, booking_details = %s,
                   summarized_at = %s
             WHERE id = %s
            """,
            (summary, booking_made, Jsonb(booking_details) if booking_details else None,
             datetime.now(timezone.utc), row_id),
        )
        conn.commit()


def _check():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*), count(summarized_at) FROM conversations")
        total, done = cur.fetchone()
        print(f"{total} conversations, {done} summarised, {total - done} pending\n")
        cur.execute(
            """
            SELECT id, business, turns, booking_made, summary
            FROM conversations ORDER BY id DESC LIMIT 10
            """
        )
        for row_id, business, turns, booked, summary in cur.fetchall():
            flag = "BOOKED" if booked else ("      " if booked is False else "   ?  ")
            print(f"#{row_id:<4} [{flag}] {(business or '?')[:26]:28} {turns or 0:2} turns  "
                  f"{(summary or '(not summarised)')[:60]}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        init_db()
        print("Done. Make a call, then run: python summarize.py")
