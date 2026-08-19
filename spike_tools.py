"""Step 0 spike: can llama-3.3-70b on Groq call booking tools correctly?

Throwaway. It answers one question before we build anything on top of it:
does the model call the right tool, with the right arguments, and — just as
important — does it stay OFF the tools when it shouldn't touch them?

A model that calls tools for everything is as broken as one that never does.

    python spike_tools.py
"""

import asyncio
import json
import os
from datetime import timedelta

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

from prompt import build_system_prompt, business_now, load_profile

load_dotenv(override=True)

MODEL = "llama-3.3-70b-versatile"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Look up the open appointment slots for one doctor on one date. "
                "Call this whenever the caller asks what times are free, and always "
                "before offering the caller any appointment time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor": {
                        "type": "string",
                        "description": "Full name of the doctor, e.g. 'Dr. Priya Nair'",
                    },
                    "date": {
                        "type": "string",
                        "description": "The date to check, in YYYY-MM-DD format",
                    },
                },
                "required": ["doctor", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Create a real appointment in the clinic's calendar. Only call this "
                "once you have the patient's name, their phone number, the doctor, "
                "and a specific slot the caller has agreed to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Patient's full name"},
                    "phone": {
                        "type": "string",
                        "description": "Patient's 10-digit Indian mobile number, digits only",
                    },
                    "doctor": {"type": "string", "description": "Full name of the doctor"},
                    "datetime": {
                        "type": "string",
                        "description": "Appointment start, YYYY-MM-DD HH:MM in 24-hour time",
                    },
                },
                "required": ["name", "phone", "doctor", "datetime"],
            },
        },
    },
]

# Each case: (label, conversation, what we expect)
CASES = [
    (
        "clear availability request",
        [{"role": "user", "content": "I'd like to see the skin doctor on Tuesday."}],
        "check_availability with Dr. Priya Nair + next Tuesday's date",
    ),
    (
        "relative date: tomorrow",
        [{"role": "user", "content": "Is Dr. Rao free tomorrow?"}],
        "check_availability with Dr. Ananya Rao + tomorrow's date",
    ),
    (
        "NEGATIVE - plain fact question",
        [{"role": "user", "content": "What are your hours on Sunday?"}],
        "NO tool call — answer from the profile",
    ),
    (
        "NEGATIVE - booking with no details yet",
        [{"role": "user", "content": "I want to book an appointment."}],
        "NO tool call — should ask which doctor / when",
    ),
    (
        "full booking, all details given",
        [
            {"role": "user", "content": "Book me with Dr. Rao on Monday at 3pm."},
            {"role": "assistant", "content": "Can I take your name and phone number?"},
            {"role": "user", "content": "Abinav, and my number is 9876543210."},
        ],
        "book_appointment with all four fields correct",
    ),
    (
        "NEGATIVE - missing phone",
        [
            {"role": "user", "content": "Book me with Dr. Rao on Monday at 3pm. My name is Abinav."},
        ],
        "NO booking — should ask for the phone number",
    ),
    (
        "after a tool result, must not invent slots",
        [
            {"role": "user", "content": "What's free with Dr. Nair on Tuesday?"},
            {
                "role": "assistant",
                "content": "We have eleven o'clock and three thirty available on Tuesday.",
            },
            {"role": "user", "content": "Is two o'clock free?"},
        ],
        "should say no / re-check, NOT invent 2pm",
    ),
]


def date_reference(now, days=10):
    """A lookup table beats date arithmetic — models are bad at the latter."""
    lines = []
    for i in range(days):
        d = now + timedelta(days=i)
        label = "today" if i == 0 else "tomorrow" if i == 1 else ""
        lines.append(f"{d.strftime('%A')} {d.day} {d.strftime('%B')} = {d.strftime('%Y-%m-%d')}"
                     + (f"  ({label})" if label else ""))
    return "DATES\n" + "\n".join(lines)


BOOKING_RULES = """
BOOKING — this replaces any earlier rule saying you cannot book.
You now have two tools and you must use them.
- Never state an appointment time unless check_availability returned it.
- Before booking you need: the doctor, a specific slot, the patient's name, and
  their phone number. If any is missing, ask for it — do not call book_appointment.
- Read the details back to the caller before you book.
"""


async def main():
    profile = load_profile()
    now = business_now(profile)
    system = "\n\n".join([
        build_system_prompt(profile, now),
        date_reference(now),
        BOOKING_RULES,
    ])

    client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    print(f"Model: {MODEL}   Today: {now.strftime('%A %Y-%m-%d')}\n" + "=" * 70)

    for label, convo, expected in CASES:
        for attempt in range(6):
            try:
                reply = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system}, *convo],
                    tools=TOOLS,
                    tool_choice="auto",
                )
                break
            except RateLimitError:
                if attempt == 5:
                    raise
                await asyncio.sleep(6)

        msg = reply.choices[0].message
        print(f"\n[{label}]")
        print(f"  said:     {convo[-1]['content']}")
        print(f"  expected: {expected}")
        if msg.tool_calls:
            for call in msg.tool_calls:
                print(f"  CALLED:   {call.function.name}({call.function.arguments})")
        else:
            print(f"  no tool.  replied: {(msg.content or '').strip()[:150]}")
        await asyncio.sleep(6)


if __name__ == "__main__":
    asyncio.run(main())
