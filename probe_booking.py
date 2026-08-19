"""Play a whole booking conversation in text, printing every tool call.

One level up from probe.py: a scripted caller goes through the real agent loop —
model decides, the real handler runs against the real calendar, the result goes
back, the model speaks. Tuning the booking flow this way takes seconds per
attempt instead of minutes at a microphone.

The conversation is generated from the profile — first staff member, their next
working day — so adding a business means adding a profile and nothing else.

    PROFILE=profiles/salon.json python probe_booking.py
    PROFILE=profiles/vet.json   CALENDAR_BACKEND=mock python probe_booking.py
"""

import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

from calendar_backend import CalComCalendar, MockCalendar
from guards import NUDGE, PROMISE
from prompt import (DEFAULT_PROFILE, booking_enabled, build_system_prompt,
                    business_now, load_profile, staff_noun)
from tools import (BOOK_APPOINTMENT, CANCEL_BOOKING, CHECK_AVAILABILITY, FIND_BOOKING,
                   RESCHEDULE_BOOKING, book_appointment, cancel_booking,
                   check_availability, find_booking, reschedule_booking, spec_for)

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def tools_for(noun):
    """OpenAI-shape tool list, in this business's word for its people."""
    return [
        {"type": "function", "function": {
            "name": spec["name"], "description": spec["description"],
            "parameters": {"type": "object", "properties": spec["properties"],
                           "required": spec["required"]},
        }}
        for spec in (spec_for(CHECK_AVAILABILITY, noun), spec_for(BOOK_APPOINTMENT, noun),
                     spec_for(FIND_BOOKING, noun), spec_for(CANCEL_BOOKING, noun),
                     spec_for(RESCHEDULE_BOOKING, noun))
    ]


async def dispatch(calendar, name, args):
    if name == "check_availability":
        return await check_availability(
            calendar, args.get("staff_member", ""), args.get("date", ""),
            args.get("preferred_time") or None)
    if name == "book_appointment":
        return await book_appointment(
            calendar, args.get("name", ""), args.get("phone", ""),
            args.get("staff_member", ""), args.get("datetime", ""))
    if name == "find_booking":
        return await find_booking(calendar, args.get("name", ""))
    if name == "cancel_booking":
        return await cancel_booking(calendar, args.get("reference", ""))
    if name == "reschedule_booking":
        return await reschedule_booking(calendar, args.get("reference", ""), args.get("datetime", ""))
    return {"ok": False, "problem": f"no such tool {name}"}


def summarise(result):
    if result.get("booked"):
        return f"BOOKED {result['reference']} | {result['staff_member']} | {result['day']} {result['time']}"
    if result.get("cancelled"):
        return f"CANCELLED | {result['with']} | {result['day']} {result['time']}"
    if result.get("rescheduled"):
        return f"MOVED to {result['day']} {result['time']} | new ref {result['reference']}"
    if "bookings" in result:
        found = result["bookings"]
        return f"{len(found)} found: " + (
            "; ".join(f"{b['with']} {b['day']} {b['time']} (ref {b['reference'][:10]})" for b in found)
            or "none")
    if result.get("ok"):
        slots = result.get("available", [])
        return f"{len(slots)} free | suggested {result.get('suggested')}" + (
            f" | {result['reason']}" if result.get("reason") else "")
    return f"REFUSED: {result.get('problem')}"


async def complete(messages, tools):
    for _ in range(5):
        try:
            reply = await client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, tool_choice="auto")
            return reply.choices[0].message
        except RateLimitError:
            await asyncio.sleep(5)
    raise RuntimeError("rate limited")


async def converse(profile, calendar, turns):
    noun = staff_noun(profile)
    tools = tools_for(noun)
    messages = [{"role": "system", "content": build_system_prompt(
        profile, business_now(profile), booking_enabled=booking_enabled(profile))}]

    for turn in turns:
        print(f"\nCALLER : {turn}", flush=True)
        messages.append({"role": "user", "content": turn})

        tool_ran = False
        for _ in range(4):  # a couple of tool round-trips per caller turn
            msg = await complete(messages, tools)
            if not msg.tool_calls:
                said = (msg.content or "").strip()
                print(f"BOT    : {said}", flush=True)
                messages.append({"role": "assistant", "content": said})
                # Mirror the ToolPromiseGuard that runs in the live pipeline, so the
                # harness shows what a caller would actually experience rather than
                # raw model behaviour.
                if not tool_ran and PROMISE.search(said):
                    print("  GUARD promised an action with no tool call — forcing it", flush=True)
                    messages.append({"role": "system", "content": NUDGE})
                    continue
                break
            messages.append({"role": "assistant", "tool_calls": [
                {"id": t.id, "type": "function",
                 "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in msg.tool_calls]})
            tool_ran = True
            for t in msg.tool_calls:
                args = json.loads(t.function.arguments or "{}")
                result = await dispatch(calendar, t.function.name, args)
                print(f"  TOOL {t.function.name}({json.dumps(args)})", flush=True)
                print(f"  ->   {summarise(result)}", flush=True)
                messages.append({"role": "tool", "tool_call_id": t.id,
                                 "content": json.dumps(result)})


def next_working_day(now, days):
    """The soonest upcoming date this person actually works."""
    for ahead in range(1, 15):
        day = now + timedelta(days=ahead)
        if day.strftime("%A") in days:
            return day
    raise SystemExit(f"Nobody works any of {days} in the next fortnight")


async def main():
    profile_path = os.getenv("PROFILE", DEFAULT_PROFILE)
    profile = load_profile(profile_path)
    now = business_now(profile)

    if os.getenv("CALENDAR_BACKEND", "calcom").lower() == "mock":
        store = Path("bookings.probe.json")
        store.unlink(missing_ok=True)
        calendar = MockCalendar(profile, store_path=str(store))
        backend = "MOCK"
    else:
        calendar = CalComCalendar(profile)
        backend = "CAL.COM (real bookings)"

    person = profile["staff"][0]
    day = next_working_day(now, person["days"])
    noun = staff_noun(profile)

    print(f"{profile['business']['name']}  |  {backend}  |  {noun}: {person['name']}  "
          f"|  {day.strftime('%A %d %B')}")

    await converse(profile, calendar, [
        f"Hi, I'd like to book an appointment with {person['name']}.",
        f"{day.strftime('%A')} please, some time in the afternoon.",
        "The first one works.",
        "Ravi Kumar.",
        "My mobile is 9 8 7 6 5 4 3 2 1 0.",
        "Yes, that's right.",
        "Yes please, go ahead and book it.",
    ])


if __name__ == "__main__":
    asyncio.run(main())
