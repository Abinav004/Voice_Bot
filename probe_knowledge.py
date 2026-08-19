"""Check that adding the knowledge store did not change what the bot already knew.

probe.py fires questions at the prompt alone. This fires them at the prompt AND
the tools, and reports which tool the model reached for — which is the only way
to see the thing that actually matters here:

    hours, prices, staff        -> answer from the prompt, NO lookup
    things not offered          -> say no from the prompt, NO lookup
    parking, insurance, refills -> look it up

The middle row is the one to watch. "Can I get an X-ray here?" matches the
outside-labs passage at 0.635 — an ordinary similarity score. If the model
reaches for the store on that question, a clean "no" becomes a maybe, and the
grounding this bot is built on starts leaking.

    python probe_knowledge.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

load_dotenv(override=True)

import knowledge  # noqa: E402
from prompt import (  # noqa: E402
    DEFAULT_PROFILE,
    booking_enabled,
    build_system_prompt,
    business_id,
    business_now,
    load_profile,
    staff_noun,
)
from tools import (  # noqa: E402
    BOOK_APPOINTMENT,
    CANCEL_BOOKING,
    CHECK_AVAILABILITY,
    FIND_BOOKING,
    RESCHEDULE_BOOKING,
    SEARCH_KNOWLEDGE,
    spec_for,
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Questions the store can answer and the prompt cannot. Not in probes/*.json
# because they only make sense once a document has been ingested.
KNOWLEDGE_QUESTIONS = [
    "Is there parking at the clinic?",
    "Do you bill my insurance directly?",
    "How do I get a repeat of my regular tablets?",
    "My mother uses a wheelchair — can she get in?",
    "What should I bring to my first visit?",
    "How long might I be kept waiting?",
]


def openai_tools(profile, with_knowledge):
    noun = staff_noun(profile)
    specs = [CHECK_AVAILABILITY, BOOK_APPOINTMENT, FIND_BOOKING,
             CANCEL_BOOKING, RESCHEDULE_BOOKING]
    if with_knowledge:
        specs.append(SEARCH_KNOWLEDGE)
    return [
        {"type": "function",
         "function": {"name": s["name"], "description": s["description"],
                      "parameters": {"type": "object", "properties": s["properties"],
                                     "required": s["required"]}}}
        for s in (spec_for(spec, noun) for spec in specs)
    ]


async def main():
    profile_path = os.getenv("PROFILE", DEFAULT_PROFILE)
    profile = load_profile(profile_path)
    bid = business_id(profile)
    has_docs = bool(knowledge.sources(bid))

    if not has_docs:
        sys.exit(f"  {bid} has no documents ingested — nothing to probe.")

    system = build_system_prompt(
        profile, business_now(profile),
        booking_enabled=booking_enabled(profile), knowledge_available=True,
    )
    tools = openai_tools(profile, with_knowledge=True)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\n{profile['business']['name']}  |  {MODEL}  |  ~{len(system)//4} prompt tokens"
          f"  |  {len(tools)} tools\n")

    async def ask(question):
        """Returns (tool_the_model_reached_for, spoken_reply)."""
        for _ in range(5):
            try:
                reply = await client.chat.completions.create(
                    model=MODEL, tools=tools,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": question}],
                )
                message = reply.choices[0].message
                called = message.tool_calls[0].function.name if message.tool_calls else None
                return called, (message.content or "").strip()
            except RateLimitError:
                await asyncio.sleep(5)
        return None, "(rate limited)"

    probes = json.loads((Path("probes") / Path(profile_path).name).read_text(encoding="utf-8"))

    # bucket -> whether reaching for search_knowledge is the wrong move
    SECTIONS = [
        ("known — answer from the prompt", probes["known"], "no-lookup"),
        ("not offered — say no from the prompt", probes["not_offered"], "no-lookup"),
        ("unknown — decline", probes["unknown"], "either"),
        ("knowledge — SHOULD look up", KNOWLEDGE_QUESTIONS, "lookup"),
    ]

    problems = 0
    for title, questions, expect in SECTIONS:
        print("=" * 76)
        print(f" {title}")
        print("=" * 76)
        results = await asyncio.gather(*(ask(q) for q in questions))
        for question, (called, answer) in zip(questions, results):
            looked = called == "search_knowledge"
            bad = (expect == "no-lookup" and looked) or (expect == "lookup" and not looked)
            problems += bad
            mark = "  !!" if bad else "    "
            tag = f"[{called}]" if called else "[no tool]"
            print(f"{mark}Q: {question}")
            print(f"      {tag} {answer[:150]}" if answer else f"      {tag}")
        print()

    print("=" * 76)
    print(f"  {problems} question(s) reached for the wrong thing"
          if problems else "  Every question went to the right place.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
