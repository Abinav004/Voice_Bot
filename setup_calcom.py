"""Create the Cal.com schedules and event types this business needs.

Run once per business at onboarding. Everything is derived from profile.json —
the doctors, their days and the opening hours — so nobody types the working days
twice and Cal.com can't silently drift from what the bot tells callers.

Idempotent: re-running matches on name/slug and reuses what already exists.

    python setup_calcom.py            # show what would be created
    python setup_calcom.py --apply    # actually create it
"""

import os
import sys

import requests
from dotenv import load_dotenv

from calendar_backend import _parse_hours
from prompt import appointment_minutes, load_profile

load_dotenv(override=True)

BASE = "https://api.cal.com/v2"
KEY = os.getenv("CAL_API")
# Appointment length comes from the profile. Cal.com event types are fixed-length,
# so per-service durations would need multi-duration event types later.

V_SCHEDULES = "2024-06-11"
V_EVENT_TYPES = "2024-06-14"

# Callers give a name and a mobile, never an email — asking someone to spell an
# address out loud is miserable. A new event type requires email, so every one we
# manage gets switched to phone-first: email optional and hidden, phone required.
PHONE_FIRST_FIELDS = [
    {"type": "name", "slug": "name", "required": True, "label": "Your name"},
    {"type": "email", "slug": "email", "required": False, "label": "Email", "hidden": True},
    {"type": "phone", "slug": "attendeePhoneNumber", "required": True, "label": "Mobile number"},
]


def _headers(version):
    return {
        "Authorization": f"Bearer {KEY}",
        "cal-api-version": version,
        "Content-Type": "application/json",
    }


def _slugify(name):
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-").replace("--", "-")


def availability_blocks(profile, days):
    """Cal.com availability for one person, grouped by opening window.

    Staff work the business's hours on the days they are in — but those hours can
    differ by day. A barber working Wednesday (till 8pm) and Sunday (till 6pm)
    needs two blocks; collapsing to the narrowest window would quietly lose his
    Wednesday evenings.
    """
    by_window = {}
    for day in days:
        text = profile.get("hours", {}).get(day)
        if not text:
            continue  # business is shut that day, so nobody works it
        # A day can contribute more than one window — a clinic working mornings
        # and evenings produces two, and Cal.com takes a day appearing in both.
        for opens, closes in _parse_hours(text):
            window = (opens.strftime("%H:%M"), closes.strftime("%H:%M"))
            by_window.setdefault(window, []).append(day)

    if not by_window:
        raise SystemExit(f"The profile has no opening hours for any of {days}")
    return [
        {"days": grouped, "startTime": start, "endTime": end}
        for (start, end), grouped in by_window.items()
    ]


def existing(path, version, key="data"):
    r = requests.get(f"{BASE}{path}", headers=_headers(version), timeout=25)
    r.raise_for_status()
    return r.json().get(key, [])


def main():
    apply = "--apply" in sys.argv
    profile = load_profile()
    tz = profile.get("business", {}).get("timezone", "Asia/Kolkata")
    biz = profile.get("business", {}).get("name", "Business")
    minutes = appointment_minutes(profile)

    schedules = existing("/schedules", V_SCHEDULES)
    event_types = existing("/event-types", V_EVENT_TYPES)
    by_sched_name = {s["name"]: s["id"] for s in schedules}
    by_slug = {e["slug"]: e["id"] for e in event_types}

    print(f"{biz} | timezone {tz} | {'APPLYING' if apply else 'DRY RUN — pass --apply to create'}\n")
    mapping = {}

    for person in profile.get("staff", []):
        name, days = person["name"], person.get("days", [])
        blocks = availability_blocks(profile, days)
        shown = ', '.join(f"{b['startTime']}-{b['endTime']} {'/'.join(d[:3] for d in b['days'])}" for b in blocks)
        sched_name = f"{name} — {biz}"
        slug = _slugify(name)

        # --- schedule -----------------------------------------------------
        sched_id = by_sched_name.get(sched_name)
        if sched_id:
            print(f"{name}\n   schedule   reuse id={sched_id}")
        elif not apply:
            print(f"{name}\n   schedule   would create {sched_name!r} | {shown}")
        else:
            r = requests.post(f"{BASE}/schedules", headers=_headers(V_SCHEDULES), timeout=25, json={
                "name": sched_name,
                "timeZone": tz,
                "isDefault": False,
                "availability": blocks,
            })
            if r.status_code >= 300:
                print(f"{name}\n   schedule   FAILED {r.status_code} {r.text[:200]}")
                continue
            sched_id = r.json()["data"]["id"]
            print(f"{name}\n   schedule   created id={sched_id} | {shown}")

        # --- event type ---------------------------------------------------
        et_id = by_slug.get(slug)
        if et_id:
            print(f"   event type reuse id={et_id} slug={slug}")
        elif not apply:
            print(f"   event type would create slug={slug} {minutes}min")
        else:
            body = {
                "title": f"{name} — {person.get('specialty', 'Consultation')}",
                "slug": slug,
                "lengthInMinutes": minutes,
                "description": f"Appointment with {name} at {biz}",
            }
            if sched_id:
                body["scheduleId"] = sched_id
            r = requests.post(f"{BASE}/event-types", headers=_headers(V_EVENT_TYPES), timeout=25, json=body)
            if r.status_code >= 300:
                print(f"   event type FAILED {r.status_code} {r.text[:300]}")
                continue
            et_id = r.json()["data"]["id"]
            print(f"   event type created id={et_id} slug={slug}")

        # Always re-apply: a reused event type may still be on the email default.
        if et_id and apply:
            r = requests.patch(f"{BASE}/event-types/{et_id}", headers=_headers(V_EVENT_TYPES),
                               timeout=25, json={"bookingFields": PHONE_FIRST_FIELDS})
            print(f"   phone-first {'ok' if r.status_code < 300 else f'FAILED {r.status_code} {r.text[:150]}'}")

        if et_id:
            mapping[name] = et_id

    if mapping:
        print("\nAdd to each staff member in the profile:")
        for name, et_id in mapping.items():
            print(f'   {name}:  "cal_event_type_id": {et_id}')


if __name__ == "__main__":
    main()
