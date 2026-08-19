"""Write a short recap for every call that hasn't got one yet.

Run it whenever — after a call, nightly, whenever. It only touches rows where
summarized_at is null, so running it twice costs nothing.

    python summarize.py
    python summarize.py --limit 5
    python summarize.py --dry-run     # print, don't write

Booking outcome is taken from the transcript's tool calls, not from the model's
opinion. This bot has told a caller "your appointment is booked" without booking
anything, so the only trustworthy evidence is whether book_appointment actually
ran and came back ok.
"""

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

from db import save_summary, unsummarized

load_dotenv(override=True)

# Summarising is a background batch job with nobody waiting, so the free Groq
# tier is fine here even though it is too slow for a live call.
MODEL = os.getenv("SUMMARY_MODEL", "llama-3.3-70b-versatile")
CLIENT = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

INSTRUCTIONS = """You are reading the transcript of one phone call to a business.

Reply with JSON only, no prose, in exactly this shape:
{"summary": "...", "booking_made": true/false, "booking_details": {...} or null}

- summary: one or two plain sentences on what the caller wanted and what happened.
  Name the person and time if an appointment was made. No preamble.
- booking_made: true only if a book_appointment tool call appears AND its result
  says booked. A promise to book is not a booking.
- booking_details: when booked, {"staff": ..., "date": ..., "time": ..., "name": ...,
  "phone": ...} using whatever the tool result gives. Otherwise null."""


def transcript_text(messages):
    """Flatten the stored messages into something readable by the model."""
    lines = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            lines.append(f"CALLER: {message.get('content', '')}")
        elif role == "tool":
            lines.append(f"TOOL RESULT: {message.get('content', '')}")
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                lines.append(f"TOOL CALL: {fn.get('name')}({fn.get('arguments')})")
            if message.get("content"):
                lines.append(f"BOT: {message['content']}")
    return "\n".join(lines)


def booking_evidence(messages):
    """Did book_appointment actually run and succeed?

    Checked in code rather than left to the model. The transcript is full of
    sentences like "your appointment is confirmed" that were said before — or
    instead of — the tool call, and a summary that trusts those would quietly
    report bookings that never happened.
    """
    booked_ids = {
        call["id"]
        for message in messages
        for call in (message.get("tool_calls") or [])
        if call.get("function", {}).get("name") == "book_appointment"
    }
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") in booked_ids:
            try:
                if json.loads(message.get("content") or "{}").get("booked"):
                    return True
            except json.JSONDecodeError:
                continue
    return False


async def summarise(messages):
    for attempt in range(5):
        try:
            reply = await CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": INSTRUCTIONS},
                          {"role": "user", "content": transcript_text(messages)}],
                response_format={"type": "json_object"},
            )
            return json.loads(reply.choices[0].message.content)
        except RateLimitError:
            await asyncio.sleep(8)
        except json.JSONDecodeError:
            continue  # re-roll; the model occasionally wraps the JSON in prose
    raise RuntimeError("could not get a summary after 5 attempts")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pending = unsummarized(args.limit)
    if not pending:
        print("Nothing to summarise — every call already has one.")
        return

    print(f"{len(pending)} call(s) to summarise using {MODEL}\n")
    for row_id, business, messages in pending:
        result = await summarise(messages)

        # The model is asked for booking_made, but the transcript is the authority.
        truth = booking_evidence(messages)
        if result.get("booking_made") and not truth:
            print(f"#{row_id}  (model claimed a booking the transcript does not show — overriding)")
            # Correct the prose too. Flipping only the flag leaves a summary that
            # reads "the caller booked an appointment" next to booking_made=false,
            # and it is the sentence a human actually reads.
            result["summary"] = (
                result["summary"].rstrip(".")
                + ". NOTE: no booking was actually made — the bot said it had booked "
                  "but never called the booking tool."
            )
        result["booking_made"] = truth
        if not truth:
            result["booking_details"] = None

        flag = "BOOKED" if truth else "      "
        print(f"#{row_id:<4} [{flag}] {(business or '?')[:24]:26} {result['summary']}")

        if not args.dry_run:
            save_summary(row_id, result["summary"], truth, result.get("booking_details"))

    if args.dry_run:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    asyncio.run(main())
