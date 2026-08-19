"""The appointment book.

Two implementations behind one interface: a local mock for building and testing
against, and (later) Cal.com. The tools layer and the prompt talk only to this
interface, so swapping the real calendar in changes nothing above it.

Doctor days, clinic hours and staff names all come from profile.json — nothing
about this clinic is hardcoded here, so pointing the bot at a different business
still means editing only the JSON.

The mock persists to bookings.json so a booking made by voice is still there
after the call ends, which is how we prove it actually landed.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from loguru import logger

from prompt import appointment_minutes

# How long one appointment blocks out now comes from the profile
# (`booking.appointment_minutes`), so a salon and a clinic can differ without
# a code change. Still one length per business, not per service.
BOOKING_BUFFER_MINUTES = 30  # Don't offer a slot that's about to start.


class CalendarError(Exception):
    """Base for anything the bot needs to tell the caller about."""


class CalendarUnavailable(CalendarError):
    """The calendar backend could not be reached."""


class SlotTaken(CalendarError):
    """Someone else got there first."""


class DoctorNotFound(CalendarError):
    """No such doctor at this business.

    A real safety net, not just tidiness: if the model ever invents a name, the
    booking is refused here rather than silently written against a fake doctor.
    """


class BookingNotFound(CalendarError):
    """No upcoming booking matches what the caller gave us."""


class InvalidPhone(CalendarError):
    """The phone number isn't usable.

    Enforced at the commit point on purpose. Prompt rules hold ~95% of the time,
    which is fine for tone and not fine for the number someone gets called back on.
    """


def _parse_clock(text):
    """'9:00 AM' -> time. Profile hours stay human-readable for the intake form."""
    return datetime.strptime(text.strip().upper(), "%I:%M %p").time()


def _parse_hours(text):
    """Opening hours for one day, as a list of (open, close) blocks.

        '9:00 AM to 7:00 PM'                        -> [(09:00, 19:00)]
        '9:00 AM to 1:00 PM and 4:00 PM to 8:00 PM' -> [(09:00, 13:00), (16:00, 20:00)]

    A list rather than one pair because most clinics here work a morning and an
    evening session with the middle of the day shut, and the old single-pair form
    could not say that at all — a real clinic's hours crashed the parser, and the
    only way round it was to claim they were open straight through and offer
    people appointments during the break.

    A business that genuinely runs straight through returns a list of one, so
    nothing that worked before changes.
    """
    blocks = []
    for piece in re.split(r"\s*(?:,|&|\band\b)\s*", text.strip()):
        if not piece:
            continue
        try:
            opens, closes = piece.split(" to ")
            opens, closes = _parse_clock(opens), _parse_clock(closes)
        except ValueError as e:
            raise CalendarError(
                f"Could not read opening hours {text!r} from the profile. Expected "
                f"something like '9:00 AM to 7:00 PM', or '9:00 AM to 1:00 PM and "
                f"4:00 PM to 8:00 PM' for a split shift."
            ) from e
        if closes <= opens:
            raise CalendarError(
                f"Opening hours {piece!r} close at or before they open. A block that "
                f"runs past midnight is not supported."
            )
        blocks.append((opens, closes))

    if not blocks:
        raise CalendarError(f"Could not read opening hours {text!r} from the profile")

    blocks.sort()
    # Overlapping blocks would hand the same slot out twice. Cheaper to refuse
    # than to silently double-book a morning.
    for (_, earlier_close), (later_open, _) in zip(blocks, blocks[1:]):
        if later_open < earlier_close:
            raise CalendarError(f"Opening hours {text!r} have overlapping blocks")
    return blocks


def _utc_offset(tz_name, when):
    """'+05:30' — the offset Cal.com needs appended to a naive local time.

    Sending a bare local time would be read as UTC and book the appointment five
    and a half hours out.
    """
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        tz = timezone(timedelta(hours=5, minutes=30))  # IST, same fallback as prompt.py
    offset = when.replace(tzinfo=tz).utcoffset() or timedelta()
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def normalise_phone(raw):
    """Indian mobile: 10 digits, optionally with +91 / 0 in front.

    Returns the bare 10 digits, or raises. Deliberately strict — a number we
    can't dial is worse than no booking.
    """
    digits = "".join(c for c in str(raw) if c.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10 or digits[0] not in "6789":
        raise InvalidPhone(
            f"{raw!r} is not a valid 10-digit Indian mobile number"
        )
    return digits


class Calendar(ABC):
    """What the booking tools are allowed to assume exists."""

    @abstractmethod
    async def get_available_slots(self, doctor: str, date: str) -> list[str]:
        """Open start times as 'HH:MM' for one doctor on one YYYY-MM-DD date."""

    @abstractmethod
    async def create_booking(self, name: str, phone: str, doctor: str, when: str,
                             reason: str = "") -> dict:
        """Book 'YYYY-MM-DD HH:MM'. Returns the confirmed booking.

        `reason` is one line on what the appointment is for, written by the bot
        from what the caller already said. It is what turns a diary entry into
        something the staff can prepare from — a name and a time say who is
        coming, not what to have ready.
        """

    @abstractmethod
    async def find_bookings(self, name: str) -> list[dict]:
        """Upcoming bookings for a caller, most imminent first.

        Matched on name alone for now, which is a demo-grade identifier: it cannot
        tell two Rajeshes apart, and knowing a name is enough to cancel that
        person's appointment. Telephony replaces this with the calling number,
        which is evidence rather than a claim.
        """

    @abstractmethod
    async def find_bookings_by_phone(self, phone: str) -> list[dict]:
        """Upcoming bookings for a number, most imminent first.

        The lookup name matching wanted to be. A number from caller ID is
        evidence the network provided; a name is a claim typed through a lossy
        8kHz channel. On a live call the difference was four failed lookups —
        "Abino", "Aveno", "A V V E N O", "A V E E N O" — against a booking that
        was sitting there under "Aveeno" with the caller's exact number on it.
        """

    @abstractmethod
    async def list_upcoming(self, days: int = 7) -> list[dict]:
        """Every upcoming booking in the next `days`, most imminent first.

        Not for the bot — no tool calls this. It exists for the business side:
        reminder calls, a day sheet, anything that starts from the diary rather
        than from a caller. find_bookings answers "what has this person booked";
        this answers "who is coming in".
        """

    @abstractmethod
    async def cancel_booking(self, reference: str) -> dict:
        """Cancel one booking. Irreversible — Cal.com has no un-cancel."""

    @abstractmethod
    async def reschedule_booking(self, reference: str, when: str) -> dict:
        """Move one booking to 'YYYY-MM-DD HH:MM'. Returns the updated booking."""


class MockCalendar(Calendar):
    """A local stand-in that behaves like the real thing, including badly.

    The failure switches exist because 'a taken slot is refused cleanly' and
    'the calendar is down' are success criteria, and they're painful to trigger
    on a real calendar — you'd have to go and book over yourself first.
    """

    def __init__(self, profile, store_path="bookings.json", now_fn=None):
        self._profile = profile
        self._store = Path(store_path)
        self._now_fn = now_fn or datetime.now

        # Flip these in tests to exercise the unhappy paths.
        self.force_offline = False
        self.force_slot_taken = False

    # ---- staff / hours, read from the profile ----------------------------

    def _resolve_doctor(self, spoken):
        """Match what the caller said to a real name on staff.

        The model may say 'Dr. Rao', 'Ananya Rao', or 'the dermatologist', so
        match on full name, surname, or speciality before giving up.
        """
        want = spoken.lower().replace("dr.", "").replace("dr ", "").strip()
        staff = self._profile.get("staff", [])

        for person in staff:
            if want == person["name"].lower().replace("dr.", "").strip():
                return person
        for person in staff:
            surname = person["name"].split()[-1].lower()
            if want == surname or want.endswith(surname):
                return person
        for person in staff:
            spec = person.get("specialty", "").lower()
            # Either direction: caller may say "dermatologist" or "the dermatologist".
            if spec and (spec in want or want in spec):
                return person
        raise DoctorNotFound(f"No doctor matching {spoken!r} works at this clinic")

    def _all_slots_for(self, person, day):
        """Every slot the doctor could theoretically work that day."""
        weekday = day.strftime("%A")
        if weekday not in person.get("days", []):
            return []

        hours = self._profile.get("hours", {}).get(weekday)
        if not hours:
            return []
        minutes = appointment_minutes(self._profile)
        slots = []
        # One pass per opening block, so a lunchtime closure is a real gap in the
        # slots rather than something the caller finds out about on arrival.
        for opens, closes in _parse_hours(hours):
            cursor = datetime.combine(day, opens)
            last_start = datetime.combine(day, closes) - timedelta(minutes=minutes)
            while cursor <= last_start:
                slots.append(cursor)
                cursor += timedelta(minutes=minutes)
        return slots

    # ---- persistence -----------------------------------------------------

    def _load(self):
        if not self._store.exists():
            return []
        try:
            return json.loads(self._store.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"{self._store} was unreadable — starting a fresh book")
            return []

    def _save(self, bookings):
        self._store.write_text(json.dumps(bookings, indent=2), encoding="utf-8")

    # ---- the interface ---------------------------------------------------

    async def get_available_slots(self, doctor, date):
        if self.force_offline:
            raise CalendarUnavailable("The calendar system is not responding")

        person = self._resolve_doctor(doctor)
        day = datetime.strptime(date, "%Y-%m-%d").date()

        taken = {
            b["datetime"]
            for b in self._load()
            # A cancelled booking must release its slot, or cancelling would be
            # pointless — the time would stay blocked for everyone else.
            if b["doctor"] == person["name"]
            and b["datetime"].startswith(date)
            and not b.get("cancelled")
        }
        # Never offer a slot that has already started, or is about to.
        earliest = self._now_fn().replace(tzinfo=None) + timedelta(minutes=BOOKING_BUFFER_MINUTES)

        return [
            slot.strftime("%H:%M")
            for slot in self._all_slots_for(person, day)
            if slot > earliest and slot.strftime("%Y-%m-%d %H:%M") not in taken
        ]

    async def create_booking(self, name, phone, doctor, when, reason=""):
        if self.force_offline:
            raise CalendarUnavailable("The calendar system is not responding")

        person = self._resolve_doctor(doctor)
        phone = normalise_phone(phone)  # raises InvalidPhone before anything is written

        try:
            start = datetime.strptime(when, "%Y-%m-%d %H:%M")
        except ValueError as e:
            raise CalendarError(f"{when!r} is not a valid appointment time") from e

        date = start.strftime("%Y-%m-%d")
        slot = start.strftime("%H:%M")
        if self.force_slot_taken or slot not in await self.get_available_slots(doctor, date):
            raise SlotTaken(f"{slot} on {date} is not available with {person['name']}")

        bookings = self._load()
        booking = {
            "id": f"BK{len(bookings) + 1:04d}",
            "name": name.strip(),
            "phone": phone,
            "doctor": person["name"],
            "specialty": person.get("specialty", ""),
            "datetime": start.strftime("%Y-%m-%d %H:%M"),
            "reason": (reason or "").strip(),
            "created_at": self._now_fn().strftime("%Y-%m-%d %H:%M:%S"),
        }
        bookings.append(booking)
        self._save(bookings)
        logger.info(f"Booked {booking['id']}: {booking['name']} with {booking['doctor']} at {booking['datetime']}")
        return booking

    # ---- changing an existing booking ------------------------------------

    def _live(self, bookings):
        """Bookings that are neither cancelled nor in the past."""
        now = self._now_fn().replace(tzinfo=None)
        return [
            b for b in bookings
            if not b.get("cancelled")
            and datetime.strptime(b["datetime"], "%Y-%m-%d %H:%M") > now
        ]

    async def find_bookings(self, name):
        if self.force_offline:
            raise CalendarUnavailable("The calendar system is not responding")
        wanted = (name or "").strip().lower()
        if not wanted:
            return []
        found = [b for b in self._live(self._load()) if wanted in b["name"].lower()]
        return sorted(found, key=lambda b: b["datetime"])

    async def find_bookings_by_phone(self, phone):
        if self.force_offline:
            raise CalendarUnavailable("The calendar system is not responding")
        try:
            wanted = normalise_phone(phone)
        except InvalidPhone:
            return []
        found = [b for b in self._live(self._load()) if b.get("phone") == wanted]
        return sorted(found, key=lambda b: b["datetime"])

    async def list_upcoming(self, days=7):
        if self.force_offline:
            raise CalendarUnavailable("The calendar system is not responding")
        cutoff = (self._now_fn().replace(tzinfo=None) + timedelta(days=days))
        found = [
            b for b in self._live(self._load())
            if datetime.strptime(b["datetime"], "%Y-%m-%d %H:%M") <= cutoff
        ]
        return sorted(found, key=lambda b: b["datetime"])

    async def cancel_booking(self, reference):
        if self.force_offline:
            raise CalendarUnavailable("The calendar system is not responding")
        bookings = self._load()
        for booking in bookings:
            if booking["id"] == reference and not booking.get("cancelled"):
                # Marked rather than deleted, so the history of a call still
                # reconciles against what the calendar shows.
                booking["cancelled"] = True
                self._save(bookings)
                logger.info(f"Cancelled {reference}")
                return booking
        raise BookingNotFound(f"No live booking with reference {reference}")

    async def reschedule_booking(self, reference, when):
        if self.force_offline:
            raise CalendarUnavailable("The calendar system is not responding")
        try:
            start = datetime.strptime(when, "%Y-%m-%d %H:%M")
        except ValueError as e:
            raise CalendarError(f"{when!r} is not a valid appointment time") from e

        bookings = self._load()
        for booking in bookings:
            if booking["id"] == reference and not booking.get("cancelled"):
                free = await self.get_available_slots(booking["doctor"], start.strftime("%Y-%m-%d"))
                if self.force_slot_taken or start.strftime("%H:%M") not in free:
                    raise SlotTaken(f"{start:%H:%M} on {start:%Y-%m-%d} is not available")
                booking["datetime"] = start.strftime("%Y-%m-%d %H:%M")
                self._save(bookings)
                logger.info(f"Moved {reference} to {booking['datetime']}")
                return booking
        raise BookingNotFound(f"No live booking with reference {reference}")


class CalComCalendar(Calendar):
    """The real thing. Cal.com holds availability; profile.json only describes it.

    One event type per doctor, each bound to a schedule matching their working
    days — see setup_calcom.py, which generates both from profile.json so the
    days are never typed twice and the two cannot drift apart.
    """

    BASE = "https://api.cal.com/v2"
    V_SLOTS = "2024-09-04"
    V_BOOKINGS = "2024-08-13"

    @staticmethod
    def _key_for(profile):
        """This client's own Cal.com key.

        The profile names an environment variable rather than holding the key
        itself — `"api_key_env": "CAL_API_GLOW_STUDIO"`. Profiles are data: they
        get edited through an onboarding form, stored in a database, shown on
        screen and copied between environments. A credential in there is a
        credential in all of those places. The name is safe to keep in the
        profile; the value stays in the environment.

        A named variable that is not set raises rather than falling back to the
        shared key. Falling back would quietly point one client's bookings at
        another client's calendar, which is the failure this exists to prevent.
        Profiles naming nothing get CAL_API, which is every profile today.
        """
        named = profile.get("booking", {}).get("api_key_env")
        if not named:
            return os.getenv("CAL_API")
        key = os.getenv(named)
        if not key:
            raise CalendarError(
                f"{named} is named in this profile but is not set in the environment. "
                f"Refusing to fall back to CAL_API — that would use another "
                f"business's calendar."
            )
        logger.debug(f"Cal.com key from {named}")
        return key

    def __init__(self, profile, api_key=None, timeout=10.0):
        self._profile = profile
        self._key = api_key or self._key_for(profile)
        self._tz = profile.get("business", {}).get("timezone", "Asia/Kolkata")
        self._timeout = timeout
        if not self._key:
            raise CalendarError(
                "No Cal.com key — set CAL_API, or name this client's own key with "
                "booking.api_key_env in the profile"
            )

    # Reuse the mock's name matching so "the skin doctor" resolves identically.
    _resolve_doctor = MockCalendar._resolve_doctor

    def _event_type_id(self, person):
        et = person.get("cal_event_type_id")
        if not et:
            raise CalendarError(
                f"{person['name']} has no cal_event_type_id in profile.json — run setup_calcom.py"
            )
        return et

    async def _request(self, method, path, version, **kwargs):
        headers = {
            "Authorization": f"Bearer {self._key}",
            "cal-api-version": version,
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(
                    method, f"{self.BASE}{path}", headers=headers, **kwargs
                )
        except httpx.HTTPError as e:
            raise CalendarUnavailable(f"Could not reach Cal.com: {e}") from e
        return response

    async def get_available_slots(self, doctor, date):
        person = self._resolve_doctor(doctor)
        response = await self._request(
            "GET", "/slots", self.V_SLOTS,
            params={
                "eventTypeId": self._event_type_id(person),
                "start": date,
                "end": date,
                "timeZone": self._tz,
            },
        )
        if response.status_code >= 500:
            raise CalendarUnavailable(f"Cal.com returned {response.status_code}")
        if response.status_code >= 300:
            raise CalendarError(f"Cal.com rejected the availability check: {response.text[:200]}")

        # {"2026-08-10": [{"start": "2026-08-10T09:00:00.000+05:30"}, ...]}
        day = response.json().get("data", {}).get(date, [])
        slots = []
        for entry in day:
            start = entry.get("start")
            if start:
                slots.append(datetime.fromisoformat(start).strftime("%H:%M"))
        return sorted(slots)

    async def create_booking(self, name, phone, doctor, when, reason=""):
        person = self._resolve_doctor(doctor)
        phone = normalise_phone(phone)  # raises InvalidPhone before anything is sent

        try:
            start = datetime.strptime(when, "%Y-%m-%d %H:%M")
        except ValueError as e:
            raise CalendarError(f"{when!r} is not a valid appointment time") from e

        # Cal.com wants an offset-aware ISO string and an E.164 number.
        local = start.strftime("%Y-%m-%dT%H:%M:%S") + _utc_offset(self._tz, start)
        body = {
            "eventTypeId": self._event_type_id(person),
            "start": local,
            "attendee": {
                "name": name.strip(),
                "phoneNumber": f"+91{phone}",
                "timeZone": self._tz,
            },
        }
        # 'notes' is a standard booking field on every event type, so the reason
        # shows up in the booking the staff actually look at rather than in
        # metadata nobody opens.
        if (reason or "").strip():
            body["bookingFieldsResponses"] = {"notes": reason.strip()}
        response = await self._request("POST", "/bookings", self.V_BOOKINGS, json=body)
        if response.status_code >= 500:
            raise CalendarUnavailable(f"Cal.com returned {response.status_code}")
        if response.status_code >= 300:
            body = response.text
            # Cal.com reports an unavailable slot as a 400; the caller needs to
            # hear "that time has gone", not a generic failure.
            if "no_available_users_found_error" in body or "not available" in body.lower():
                raise SlotTaken(f"{start.strftime('%H:%M')} on {start:%Y-%m-%d} is no longer free")
            raise CalendarError(f"Cal.com rejected the booking: {body[:200]}")

        data = response.json().get("data", {})
        booking = {
            "id": str(data.get("uid") or data.get("id", "")),
            "name": name.strip(),
            "phone": phone,
            "doctor": person["name"],
            "specialty": person.get("specialty", ""),
            "datetime": start.strftime("%Y-%m-%d %H:%M"),
            "reason": (reason or "").strip(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        logger.info(f"Cal.com booked {booking['id']}: {booking['name']} with {booking['doctor']} at {booking['datetime']}"
                    + (f" — {booking['reason']}" if booking["reason"] else ""))
        return booking

    # ---- changing an existing booking ------------------------------------

    def _my_event_types(self):
        """The event type ids belonging to this business's staff.

        One Cal.com account can host several businesses — it does here, where a
        clinic, a salon and a vet share one key. Without this, a phone lookup
        returns every booking that number has anywhere on the account: a caller
        who rang the salon to cancel a haircut was shown their appointment with
        Dr. Ananya Rao. Reading one business's bookings out on another's phone
        line is a data leak, not an inconvenience.

        Production gives each client its own Cal.com account, which separates
        them anyway. This makes the separation hold regardless — the same reason
        the knowledge store filters on business id rather than trusting that
        only one business is ever present.
        """
        return {
            person["cal_event_type_id"]
            for person in self._profile.get("staff", [])
            if person.get("cal_event_type_id")
        }

    def _as_booking(self, raw):
        """One Cal.com booking in the shape the rest of the code expects."""
        attendee = (raw.get("attendees") or [{}])[0]
        local = datetime.fromisoformat(raw["start"].replace("Z", "+00:00")).astimezone(
            self._zone())
        return {
            "id": raw.get("uid"),
            "name": attendee.get("name", ""),
            "phone": (attendee.get("phoneNumber") or "").lstrip("+").removeprefix("91"),
            "doctor": (raw.get("title") or "").split(" — ")[0].split(" between ")[0],
            "datetime": local.strftime("%Y-%m-%d %H:%M"),
            # Read back so a day sheet or a reminder call can say what the
            # appointment is for, not just who is coming.
            "reason": (raw.get("bookingFieldsResponses") or {}).get("notes", "")
                      or raw.get("description", "") or "",
        }

    def _zone(self):
        try:
            return ZoneInfo(self._tz)
        except ZoneInfoNotFoundError:
            return timezone(timedelta(hours=5, minutes=30))

    async def find_bookings(self, name):
        wanted = (name or "").strip()
        if not wanted:
            return []
        response = await self._request(
            "GET", "/bookings", self.V_BOOKINGS,
            # status=upcoming matters: without it Cal.com also returns cancelled
            # bookings, and the bot would offer to cancel one that is already gone.
            params={"attendeeName": wanted, "status": "upcoming"},
        )
        if response.status_code >= 500:
            raise CalendarUnavailable(f"Cal.com returned {response.status_code}")
        if response.status_code >= 300:
            raise CalendarError(f"Cal.com rejected the lookup: {response.text[:200]}")
        found = [self._as_booking(b) for b in response.json().get("data", [])]
        return sorted(found, key=lambda b: b["datetime"])

    async def _all_upcoming(self):
        """Every upcoming booking on the account, in our booking shape.

        Cal.com's /bookings filters server-side on attendeeName but not on
        attendee phone, so anything phone-based has to fetch and filter here.
        """
        response = await self._request(
            "GET", "/bookings", self.V_BOOKINGS,
            # No attendeeName: we want the whole diary, not one person's. take=100
            # because Cal.com pages, and a silently truncated list is worse than
            # an error — a patient just never gets found.
            params={"status": "upcoming", "take": 100},
        )
        if response.status_code >= 500:
            raise CalendarUnavailable(f"Cal.com returned {response.status_code}")
        if response.status_code >= 300:
            raise CalendarError(f"Cal.com rejected the lookup: {response.text[:200]}")
        data = response.json().get("data", [])
        if len(data) >= 100:
            logger.warning("100 upcoming bookings returned — the list may be truncated")

        # Only this business's bookings. Cal.com has no server-side filter for
        # "bookings belonging to these event types", so it is done here.
        mine = self._my_event_types()
        if not mine:
            raise CalendarError(
                "No cal_event_type_id on any staff member — cannot tell this "
                "business's bookings from another's. Run setup_calcom.py."
            )
        kept = [raw for raw in data if raw.get("eventTypeId") in mine]
        if len(kept) != len(data):
            logger.debug(f"Ignored {len(data) - len(kept)} bookings belonging to other businesses")
        return [self._as_booking(raw) for raw in kept]

    async def find_bookings_by_phone(self, phone):
        try:
            wanted = normalise_phone(phone)
        except InvalidPhone:
            return []
        # No day limit: someone ringing about an appointment six weeks out still
        # deserves to be found. list_upcoming bounds by days because a reminder
        # sheet wants this week; a lookup wants everything.
        found = [b for b in await self._all_upcoming() if b["phone"] == wanted]
        return sorted(found, key=lambda b: b["datetime"])

    async def list_upcoming(self, days=7):
        cutoff = datetime.now(self._zone()).replace(tzinfo=None) + timedelta(days=days)
        found = [
            b for b in await self._all_upcoming()
            if datetime.strptime(b["datetime"], "%Y-%m-%d %H:%M") <= cutoff
        ]
        return sorted(found, key=lambda b: b["datetime"])

    async def cancel_booking(self, reference):
        response = await self._request(
            "POST", f"/bookings/{reference}/cancel", self.V_BOOKINGS,
            json={"cancellationReason": "Cancelled by the caller over the phone"},
        )
        if response.status_code == 404:
            raise BookingNotFound(f"No booking with reference {reference}")
        if response.status_code >= 500:
            raise CalendarUnavailable(f"Cal.com returned {response.status_code}")
        if response.status_code >= 300:
            raise CalendarError(f"Cal.com refused to cancel it: {response.text[:200]}")
        logger.info(f"Cal.com cancelled {reference}")
        return self._as_booking(response.json().get("data", {}))

    async def reschedule_booking(self, reference, when):
        try:
            start = datetime.strptime(when, "%Y-%m-%d %H:%M")
        except ValueError as e:
            raise CalendarError(f"{when!r} is not a valid appointment time") from e

        local = start.strftime("%Y-%m-%dT%H:%M:%S") + _utc_offset(self._tz, start)
        response = await self._request(
            "POST", f"/bookings/{reference}/reschedule", self.V_BOOKINGS,
            json={"start": local, "reschedulingReason": "Moved by the caller over the phone"},
        )
        if response.status_code == 404:
            raise BookingNotFound(f"No booking with reference {reference}")
        if response.status_code >= 500:
            raise CalendarUnavailable(f"Cal.com returned {response.status_code}")
        if response.status_code >= 300:
            body = response.text
            if "no_available_users_found_error" in body or "not available" in body.lower():
                raise SlotTaken(f"{start:%H:%M} on {start:%Y-%m-%d} is not free")
            raise CalendarError(f"Cal.com refused to move it: {body[:200]}")
        logger.info(f"Cal.com moved {reference} to {when}")
        return self._as_booking(response.json().get("data", {}))


if __name__ == "__main__":
    import asyncio

    from prompt import business_now, load_profile

    profile = load_profile()
    cal = MockCalendar(profile, store_path="bookings.demo.json")
    Path("bookings.demo.json").unlink(missing_ok=True)
    now = business_now(profile)

    async def demo():
        # Next Tuesday — the dermatologist works Tue/Thu/Sat.
        tue = now + timedelta(days=(1 - now.weekday()) % 7 or 7)
        date = tue.strftime("%Y-%m-%d")
        print(f"Today is {now.strftime('%A %d %B')}; checking Tuesday {date}\n")

        slots = await cal.get_available_slots("the dermatologist", date)
        print(f"Dr. Nair, Tuesday      : {len(slots)} slots, first few {slots[:5]}")

        mon = now + timedelta(days=(0 - now.weekday()) % 7 or 7)
        print(f"Dr. Nair, Monday       : {await cal.get_available_slots('Dr. Nair', mon.strftime('%Y-%m-%d'))} (she is off)")

        booked = await cal.create_booking("Abinav", "9876543210", "Dr. Nair", f"{date} {slots[0]}")
        print(f"\nBooked                 : {booked['id']} {booked['name']} {booked['doctor']} {booked['datetime']}")

        after = await cal.get_available_slots("the dermatologist", date)
        print(f"Same slot still offered: {slots[0] in after}  (should be False)")

        for label, coro in [
            ("double-book", cal.create_booking("Someone", "9876543211", "Dr. Nair", f"{date} {slots[0]}")),
            ("bad phone", cal.create_booking("Abinav", "12345", "Dr. Nair", f"{date} {slots[1]}")),
            ("unknown doctor", cal.create_booking("Abinav", "9876543210", "Dr. Kapoor", f"{date} {slots[1]}")),
        ]:
            try:
                await coro
                print(f"{label:23}: NO ERROR — that's a bug")
            except CalendarError as e:
                print(f"{label:23}: {type(e).__name__} — {e}")

        cal.force_offline = True
        try:
            await cal.get_available_slots("Dr. Nair", date)
        except CalendarError as e:
            print(f"{'offline':23}: {type(e).__name__} — {e}")

    asyncio.run(demo())
