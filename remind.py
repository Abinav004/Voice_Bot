"""Ring one patient about their appointment.

The other direction. Everything so far has waited for the phone to ring; this
picks it up and dials. The bot on the far end is the same bot — Exotel connects
the answered call to the same Voicebot applet, which opens the same websocket to
exotel_server.py, which builds the same pipeline.

    python remind.py                 # list, pick one, call it
    python remind.py --dry-run       # show the request, dial nothing
    python remind.py --days 14       # look further ahead than a week

Deliberately one call at a time, chosen by a human, behind a confirmation
prompt. Bulk dialling is a different feature with very different consequences,
and it is not this script.

Before real patients: outbound automated voice sits under TRAI's TCCCPR and
needs the DLT / Principal Entity registration finished. Reminders to someone who
gave you their number when booking are service calls rather than promotional
ones, which is the favourable side of that line, but "favourable" is not
"exempt". Calling your own phone to test is fine.
"""

import argparse
import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

from calendar_backend import CalComCalendar, CalendarError, MockCalendar
from prompt import business_now, load_profile, staff_noun

load_dotenv(override=True)

# Exotel's Connect API. Credentials come from the "API Credentials" page in the
# Exotel dashboard — the same page linked in the top bar.
EXOTEL_SID = os.getenv("EXOTEL_SID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN")
# Mumbai accounts are api.exotel.com; Singapore ones api.in.exotel.com. The
# dashboard shows which. Wrong subdomain gives a 404 that reads like a bad path.
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")
# The ExoPhone the patient sees ringing them.
EXOTEL_CALLER_ID = os.getenv("EXOTEL_CALLER_ID")
# The Voicebot applet — the same flow the inbound number already points at.
# Found in the Exotel App Bazaar URL for your flow.
EXOTEL_FLOW_APP_ID = os.getenv("EXOTEL_FLOW_APP_ID")

CALENDAR_BACKEND = os.getenv("CALENDAR_BACKEND", "calcom").lower()

# TCCCPR restricts automated voice to daytime hours. Enforced here rather than
# left to whoever runs the script at 11pm during a demo rehearsal.
CALL_WINDOW = (9, 21)


def exotel_config_problem():
    """Which credentials are missing, or None if we are ready to dial."""
    missing = [
        name for name, value in (
            ("EXOTEL_SID", EXOTEL_SID),
            ("EXOTEL_API_KEY", EXOTEL_API_KEY),
            ("EXOTEL_API_TOKEN", EXOTEL_API_TOKEN),
            ("EXOTEL_CALLER_ID", EXOTEL_CALLER_ID),
            ("EXOTEL_FLOW_APP_ID", EXOTEL_FLOW_APP_ID),
        ) if not value
    ]
    return f"Missing in .env: {', '.join(missing)}" if missing else None


def applet_url():
    """The flow Exotel should connect the patient to once they answer."""
    return f"http://my.exotel.com/{EXOTEL_SID}/exoml/start_voice/{EXOTEL_FLOW_APP_ID}"


def place_call(to_number, booking_reference, dry_run=False):
    """Dial one number and hand the answered call to the bot's applet.

    Returns Exotel's CallSid. `From` really is the person being called and
    `CallerId` really is our own number — the naming reads backwards, because
    the API models it as "connect this customer to this flow", not "we ring
    them".
    """
    url = f"https://{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_SID}/Calls/connect.json"
    payload = {
        "From": f"+91{to_number}",
        "CallerId": EXOTEL_CALLER_ID,
        "Url": applet_url(),
        # Rides along to the websocket as custom_parameters, which is how the
        # server will eventually know which appointment it is calling about.
        # Whether it actually survives the trip is what the first test call
        # tells us — watch for it in the "Raw call_data" log line.
        "CustomField": booking_reference,
        # Service call, not promotional. Exotel routes and reports on this.
        "CallType": "trans",
    }

    if dry_run:
        print("\n  DRY RUN — nothing dialled")
        print(f"  POST {url}")
        for key, value in payload.items():
            print(f"    {key:12} {value}")
        return None

    response = httpx.post(
        url,
        data=payload,
        # Basic auth rather than credentials in the URL, so a stray traceback
        # or proxy log never contains the token.
        auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN),
        timeout=20.0,
    )
    if response.status_code >= 300:
        raise RuntimeError(f"Exotel refused the call ({response.status_code}): {response.text[:300]}")
    return (response.json().get("Call") or {}).get("Sid", "?")


def friendly(when):
    """'2026-08-14 16:00' -> 'Thu 14 Aug, 4:00 PM'."""
    from datetime import datetime
    parsed = datetime.strptime(when, "%Y-%m-%d %H:%M")
    return parsed.strftime("%a %d %b, %I:%M %p").replace(" 0", " ").lstrip("0")


def show(bookings, noun):
    """The diary, numbered for picking."""
    print(f"\n  Upcoming appointments ({len(bookings)})\n")
    print(f"  {'#':<3}{'Patient':<22}{'Phone':<13}{noun.title():<18}When")
    print(f"  {'-' * 74}")
    for index, booking in enumerate(bookings, start=1):
        print(
            f"  {index:<3}{booking['name'][:21]:<22}{booking['phone']:<13}"
            f"{booking['doctor'][:17]:<18}{friendly(booking['datetime'])}"
        )
    print()


def choose(bookings):
    """Which one to ring, or None to walk away."""
    raw = input(f"  Which one? [1-{len(bookings)}, or Enter to quit] ").strip()
    if not raw:
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= len(bookings):
        print("  Not one of the numbers listed.")
        return None
    return bookings[int(raw) - 1]


async def main():
    parser = argparse.ArgumentParser(description="Call one patient about their appointment.")
    parser.add_argument("--days", type=int, default=7, help="how far ahead to look (default 7)")
    parser.add_argument("--dry-run", action="store_true", help="show the request, dial nothing")
    parser.add_argument("--force", action="store_true", help="dial outside the allowed hours")
    args = parser.parse_args()

    profile = load_profile()
    business = profile.get("business", {}).get("name", "this business")
    noun = staff_noun(profile)

    calendar = MockCalendar(profile) if CALENDAR_BACKEND == "mock" else CalComCalendar(profile)
    print(f"\n  {business} — {'MOCK diary' if CALENDAR_BACKEND == 'mock' else 'Cal.com'}")

    try:
        bookings = await calendar.list_upcoming(days=args.days)
    except CalendarError as e:
        sys.exit(f"  Could not read the diary: {e}")

    if not bookings:
        sys.exit(f"  Nothing booked in the next {args.days} days — nobody to remind.")

    show(bookings, noun)
    booking = choose(bookings)
    if not booking:
        sys.exit("  Nothing dialled.")

    if not booking.get("phone"):
        sys.exit(f"  {booking['name']} has no phone number on the booking — cannot call.")

    # Time-of-day check after the pick, so you can still browse the diary at
    # midnight; it only blocks the part that actually rings someone.
    hour = business_now(profile).hour
    if not CALL_WINDOW[0] <= hour < CALL_WINDOW[1] and not (args.force or args.dry_run):
        sys.exit(
            f"  It is {hour}:00 — outside the {CALL_WINDOW[0]}:00-{CALL_WINDOW[1]}:00 calling window.\n"
            f"  Automated voice has time-of-day restrictions. Use --force if you are "
            f"certain (testing your own phone)."
        )

    print(
        f"\n  Call {booking['name']} on {booking['phone']}\n"
        f"  about {booking['doctor']}, {friendly(booking['datetime'])}?"
    )
    if input("  [y/N] ").strip().lower() != "y":
        sys.exit("  Nothing dialled.")

    problem = exotel_config_problem()
    if problem and not args.dry_run:
        sys.exit(f"  {problem}")

    try:
        call_sid = place_call(booking["phone"], booking["id"], dry_run=args.dry_run)
    except (httpx.HTTPError, RuntimeError) as e:
        sys.exit(f"  Call failed: {e}")

    if call_sid:
        print(f"\n  Dialling {booking['phone']} — CallSid {call_sid}")
        print("  Watch the uvicorn log for 'Raw call_data' when they answer.\n")


if __name__ == "__main__":
    asyncio.run(main())
