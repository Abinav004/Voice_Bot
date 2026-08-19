"""Check everything the demo depends on, before the demo.

Read-only. Run it, get a list of ticks, then start talking.

    python preflight.py
"""

import asyncio
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from calendar_backend import CalComCalendar
from prompt import (appointment_minutes, booking_enabled, build_system_prompt,
                    business_now, load_profile, staff_noun)

load_dotenv(override=True)

PROFILES = ["profiles/clinic.json", "profiles/salon.json", "profiles/vet.json"]
DEMO_DATE = "2026-08-14"  # the Friday the demo books into
NEEDED_KEYS = ["OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "CAL_API"]

ok, warn = "  ok  ", " WARN "


def check_keys():
    print("API keys")
    for key in NEEDED_KEYS:
        present = bool(os.getenv(key))
        print(f"  [{ok if present else warn}] {key}")
    return all(os.getenv(k) for k in NEEDED_KEYS)


def check_calendar_clear():
    """Only non-cancelled bookings matter — cancelled ones stay in the list."""
    print(f"\nCal.com bookings still live")
    r = requests.get(
        "https://api.cal.com/v2/bookings",
        headers={"Authorization": f"Bearer {os.getenv('CAL_API')}", "cal-api-version": "2024-08-13"},
        params={"take": 100}, timeout=25,
    )
    if r.status_code >= 300:
        print(f"  [{warn}] Cal.com returned {r.status_code}")
        return False
    live = [b for b in r.json().get("data", []) if b.get("status") != "cancelled"]
    if not live:
        print(f"  [{ok}] none — calendar is clear")
        return True
    for b in live:
        who = (b.get("attendees") or [{}])[0].get("name", "?")
        print(f"  [{warn}] {b.get('start', '')[:16]}  {who}")
    print("        clear these before demoing, or Friday will look busy")
    return False


async def check_profiles():
    print("\nProfiles")
    healthy = True
    for path in PROFILES:
        try:
            profile = load_profile(path)
            build_system_prompt(profile, business_now(profile),
                                booking_enabled=booking_enabled(profile))
        except Exception as e:
            print(f"  [{warn}] {path}: {type(e).__name__} {e}")
            healthy = False
            continue

        missing = [p["name"] for p in profile.get("staff", []) if not p.get("cal_event_type_id")]
        label = f"{profile['business']['name']} ({staff_noun(profile)}s, {appointment_minutes(profile)}min)"
        if missing:
            print(f"  [{warn}] {label}: no cal_event_type_id for {missing} — run setup_calcom.py")
            healthy = False
            continue

        calendar = CalComCalendar(profile)
        counts = []
        for person in profile["staff"]:
            slots = await calendar.get_available_slots(person["name"], DEMO_DATE)
            counts.append(f"{person['name'].split()[-1]} {len(slots)}")
        print(f"  [{ok}] {label}")
        print(f"         Fri {DEMO_DATE}: {' | '.join(counts)}")
    return healthy


async def main():
    print(f"Demo pre-flight — booking date {DEMO_DATE}\n")
    results = [check_keys(), check_calendar_clear(), await check_profiles()]
    print()
    print("READY" if all(results) else "NOT READY — see the WARN lines above")


if __name__ == "__main__":
    asyncio.run(main())
