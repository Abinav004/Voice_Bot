"""Turn a business profile (profile.json) into the bot's system prompt.

The template here is business-agnostic — it knows about hours, staff, services and
policies, but nothing about clinics specifically. Everything clinic-shaped lives in
the profile's can_do / cannot_do / not_offered / escalation fields, so pointing the
bot at a restaurant or a salon means swapping the JSON, not editing this file.

Sections are skipped when the profile omits them, so a business with no staff list
or no price list still produces a valid prompt.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger


DEFAULT_PROFILE = "profiles/clinic.json"


def load_profile(path=None):
    """Read a business profile from disk.

    Which business the bot is depends entirely on this file. Point at another one
    with `PROFILE=profiles/salon.json python agent.py` — no code changes.
    """
    return json.loads(Path(path or os.getenv("PROFILE", DEFAULT_PROFILE)).read_text(encoding="utf-8"))


def _last_ten(number):
    """The comparable part of a phone number.

    Exotel may hand the same line over as 09513886363, +919513886363 or
    9513886363 depending on the route the call took, so nothing may be compared
    as written. The last ten digits are the part that identifies an Indian line.
    """
    digits = "".join(c for c in str(number or "") if c.isdigit())
    return digits[-10:] if len(digits) >= 10 else ""


def phone_registry(directory="profiles"):
    """{last ten digits of an ExoPhone: path to the profile that answers it}.

    Built by reading the profiles directory rather than from a separate routing
    table, so onboarding a client is creating one file and not creating a file
    AND remembering to register it somewhere else. Rebuilt per call, which also
    means a new client's profile is live the moment it is saved.

    Two profiles claiming one number is a configuration error that would send
    callers to whichever happened to be read first, so it raises instead.
    """
    registry = {}
    for path in sorted(Path(directory).glob("*.json")):
        try:
            biz = json.loads(path.read_text(encoding="utf-8")).get("business", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping unreadable profile {path}: {e}")
            continue
        key = _last_ten(biz.get("exophone"))
        if not key:
            continue
        if key in registry and registry[key] != str(path):
            raise ValueError(
                f"ExoPhone {biz['exophone']} is claimed by both {registry[key]} "
                f"and {path} — one number can only answer as one business"
            )
        registry[key] = str(path)
    return registry


def profile_for_number(dialled, directory="profiles"):
    """Which business answers this dialled number, or None if we do not serve it.

    None means refuse the call. It must never mean "use the default business" —
    that is the path where a stranger reaches the wrong clinic's bot, hears the
    wrong prices, and gets answers from another business's documents.
    """
    key = _last_ten(dialled)
    return phone_registry(directory).get(key) if key else None


def business_id(profile):
    """The tenant key — which business owns a row in the knowledge store.

    Explicit `business.id` in the profile wins. It is a real identifier the
    business keeps for life: the display name can be edited, two clients can
    share one, and a filename is an accident of where the file happens to sit.
    Anything scoped on those would eventually read one clinic's documents out
    on another clinic's call.

    Falls back to a slug of the name so a profile without an id still works
    rather than failing at ingestion time, but every profile we ship sets one.
    """
    biz = profile.get("business", {})
    if biz.get("id"):
        return str(biz["id"]).strip().lower()
    name = biz.get("name", "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return slug or "default"


def staff_noun(profile, plural=False):
    """What this business calls its people — 'doctor', 'stylist', 'vet'.

    Used in the prompt and in the tool descriptions, both of which the model reads.
    Without it a salon bot asks callers "which doctor would you like?".
    """
    words = profile.get("assistant", {})
    if plural:
        return words.get("staff_noun_plural") or (staff_noun(profile) + "s")
    return words.get("staff_noun", "staff member")


# Locale code -> what to call it in the prompt. The model needs the name, the
# speech services need the code.
LANGUAGE_NAMES = {
    "en-IN": "English", "hi-IN": "Hindi", "ta-IN": "Tamil", "te-IN": "Telugu",
    "kn-IN": "Kannada", "ml-IN": "Malayalam", "mr-IN": "Marathi", "bn-IN": "Bengali",
    "gu-IN": "Gujarati", "pa-IN": "Punjabi", "od-IN": "Odia",
}


def language(profile):
    """(locale code, human name) for the language this bot speaks.

    From the profile, overridable with LANGUAGE=ta-IN for A/B testing the same
    business in two languages without editing the file.

    Note the honest limitation: the profile's facts are written in English, so a
    non-English bot is translating prices, names and policies live on every call.
    That is exactly the kind of transformation that has produced wrong digits and
    wrong times in this project before — verify accuracy before trusting it.
    """
    code = os.getenv("LANGUAGE") or profile.get("language", {}).get("code", "en-IN")
    return code, LANGUAGE_NAMES.get(code, code)


CODE_FOR_LANGUAGE = {name: code for code, name in LANGUAGE_NAMES.items()}


def multilingual_rules(offered):
    """Rules for a bot that lets the caller pick a language at the start.

    Used by the Sarvam stack, which can hear and speak Indic languages. The
    English stack never sees this block.

    Names and digits stay in English whatever language is chosen: translating them
    turned "Dr. Priya Nair" into an invented surname and dropped a digit from the
    clinic's phone number. Code-switching like this is also how people here
    actually speak.
    """
    names = _join(offered)
    return f"""LANGUAGE — the caller chooses
You can speak {names}.
- Begin in English. Right after your greeting, ask which of {names} they would like
  to continue in. Ask once, briefly.
- The moment they choose, call set_language. Do not start speaking the new language
  before that call — the phone line itself has to be switched over first.
- After it is switched, speak only in that language for the rest of the call.
- Whatever language you are speaking, always say these in English, exactly as written
  above: people's names, the name of the business, the address, and every phone
  number. Never translate a name and never turn digits into another script — the
  caller has to be able to write them down or repeat them back.
- Never change a price, a duration, a time or a date when translating."""


def knowledge_rules():
    """When to open the binder.

    Added only when the business actually has documents stored. Two jobs: tell
    the model the drawer exists, and fence it off from everything the profile
    already answers completely. Without the fence it looks up "what are your
    hours?" — slower, and it invites a near-miss passage to argue with a fact
    that was already correct.
    """
    return """LOOKING SOMETHING UP
Plenty about this business is on file but not written above — parking and
directions, insurance and reimbursement, prescription refills, accessibility,
what to bring, how test results are handled, waiting times, and more.
- You have NO WAY of knowing whether something is on file until you look. So
  never tell a caller you do not have information, and never offer to take a
  message, until you have called search_knowledge with their question and it has
  come back with nothing. Searching first is not optional.
- Do not use it for opening hours, prices, services, who works here, or whether
  the business offers something. Everything above about those is complete, and
  what it says is correct.
- Once it comes back empty, then say you do not have that and offer to take a
  message. Never guess, and never treat an empty result as a yes or a no."""


def booking_enabled(profile):
    """Whether this business can take bookings at all.

    A business with no staff has nothing to book against, so it gets the
    read-only receptionist. Set `booking.enabled` to force it either way.
    """
    booking = profile.get("booking", {})
    if "enabled" in booking:
        return bool(booking["enabled"])
    return bool(profile.get("staff"))


def appointment_minutes(profile):
    """How long one appointment blocks out.

    One length per business rather than per service. Cal.com event types are
    fixed-length, so per-service durations would mean an event type per person per
    service. The trade: a service much longer than this can still overlap the next
    booking — worth knowing before a real salon relies on it.
    """
    return profile.get("booking", {}).get("appointment_minutes", 60)


def business_now(profile):
    """Current time in the business's own timezone, defaulting to IST.

    Windows ships no system timezone database, so this needs the `tzdata`
    package. If it is missing we fall back to a fixed IST offset rather than
    crashing the call — a wrong-by-hours clock would be worse than a stale one.
    """
    tz = profile.get("business", {}).get("timezone", "Asia/Kolkata")
    try:
        return datetime.now(ZoneInfo(tz))
    except ZoneInfoNotFoundError:
        logger.warning(f"No timezone data for {tz} — falling back to IST. `uv add tzdata` to fix.")
        return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def _join(items):
    """'a, b and c' — reads better out loud than a comma list."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _bullets(items):
    return "\n".join(f"- {item}" for item in items)


_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def spoken_number(text):
    """'+91 80 2233 8899' -> 'plus nine one, eight zero, two two three three, ...'

    Spelled out here rather than left to the model. Told only to read digits one
    at a time, it dropped the 9 and the 8-0 from a clinic's number and gave a
    caller '+81 2233 8899' — during a pet emergency. Same fix as the time slots:
    when the model keeps mangling a transformation, do the transformation for it.
    """
    groups = []
    for group in str(text).split():
        spoken = " ".join(
            "plus" if ch == "+" else _DIGIT_WORDS.get(ch, ch) for ch in group if ch != "-"
        )
        if spoken:
            groups.append(spoken)
    return ", ".join(groups)


def _format_now(now):
    # %-d / %-I aren't portable to Windows, so build the date by hand.
    return (
        f"{now.strftime('%A')}, {now.day} {now.strftime('%B %Y')}, "
        f"and the time is {now.strftime('%I:%M %p').lstrip('0')}"
    )


def _service_line(service):
    parts = [service["name"]]
    if service.get("duration_min"):
        parts.append(f"{service['duration_min']} minutes")
    if service.get("price_inr") is not None:
        parts.append(f"{service['price_inr']} rupees")
    return " — ".join(parts)


def date_reference(now, days=10):
    """A dated calendar for the next `days` days.

    Models are unreliable at "what date is next Tuesday" and reliable at reading
    a table, so give them the table and let them look it up. Measured in the
    Day-5 spike: with this block, relative dates resolved correctly every time.
    """
    lines = []
    for i in range(days):
        d = now + timedelta(days=i)
        suffix = "   <- today" if i == 0 else "   <- tomorrow" if i == 1 else ""
        lines.append(f"{d.strftime('%A')} {d.day} {d.strftime('%B')} = {d.strftime('%Y-%m-%d')}{suffix}")
    return "THE NEXT FEW DAYS — use these exact dates, never work one out yourself\n" + "\n".join(lines)


# Kept verbatim from the Day-5 spike, where it took booking from 1-in-5 to 6-in-6.
# The decisive line is the one about not confirming twice: without it the model
# loops politely re-confirming and never actually commits.
def booking_rules(noun, caller_number=None):
    """The booking flow, in this business's own word for its people.

    `noun` is "doctor", "stylist", "vet" — whatever the profile says. It appears in
    the prompt AND in the tool descriptions, both of which the model reads, so a
    salon running the clinic's wording would ask callers "which doctor?".

    `caller_number` is the caller's own number when telephony gave it to us. It
    turns step 6 from "ask for ten digits" into "confirm the one we already have",
    which is the difference between three fragile turns and one. Left None — the
    browser, or a withheld number — the flow is unchanged.
    """
    if caller_number:
        # Confirm, never assume: people ring for a parent or from an office line.
        # Last four digits only, because reading all ten back defeats the point.
        step_six = (
            f"  6. You already have their number: {spoken_number(caller_number)}. Do NOT\n"
            f"     ask them to read it out. Just check it is the right one to use, by\n"
            f"     the last four digits only: \"and can I reach you on the number you're\n"
            f"     calling from, ending {spoken_number(caller_number[-4:])}?\"\n"
            f"     If they say yes, you are done — do not read the whole number back.\n"
            f"     Only if they want a different number, take it and read it back one\n"
            f"     digit at a time."
        )
        phone_rules = (
            f"- The caller is ringing from {spoken_number(caller_number)}. This came from the "
            f"phone line itself, not from speech, so it is correct. Never ask them to "
            f"recite a number you already have.\n"
            f"- Leave the phone argument out of book_appointment entirely and their own "
            f"number is used. Only pass one if they asked for a different number."
        )
        # The whole point of caller ID on this flow: their booking is found by the
        # number, so the name step disappears. On a live call the name step cost
        # four failed lookups and the caller never got their appointment.
        find_step = (
            "  1. Call find_booking straight away with no name at all. You already know\n"
            "     who is calling, so asking them to say or spell their name wastes turns\n"
            "     and mishears them. Only if that finds nothing should you ask whose name\n"
            "     it might be under and try again with that name."
        )
    else:
        step_six = (
            "  6. Then ask for their mobile number, and read it back one digit at a time so\n"
            "     they can correct it there and then."
        )
        phone_rules = "- Repeat the phone number back one digit at a time so the caller can correct it."
        find_step = (
            "  1. Ask for their name and call find_booking. You cannot cancel or move anything\n"
            "     without the reference it gives you — never make a reference up."
        )

    return f"""BOOKING — you can really book appointments, using your two tools

Work through it in this order, and do not jump ahead:
  1. Which {noun} or which service they need. If the service points to only one
     {noun}, just pick that {noun} and say who it will be — match the service to
     whoever offers it. Only ask if it is genuinely unclear.
  2. Which day. Turn that into a real date using the dated list above.
  3. Call check_availability.
     If the caller hinted at a time — "in the morning", "around five" — pass it as
     preferred_time so the suggestions come back near it.
  4. Read out the times in the result's 'suggested' list, all of them, and nothing
     else. They have already been chosen for you. Do not read from 'available',
     do not add times of your own, and never describe a range like "any time until
     six thirty". Finish with "or would another time suit you better?".
  5. Once they pick a time, ask for their name.
{step_six}
  7. Say back just the {noun}, the day and the time — nothing else — and wait for
     them to agree.
  8. Book it.

Only ask for a name and number after they have agreed a time. If the {noun} turns
out to be away or fully booked, you will have taken their details for nothing.

- Call check_availability before you offer any time at all. Never guess or assume a slot.
- Only say times the tool returned. If the caller asks for a time that is not in the
  list, tell them it is not free and offer ones that are.
- Each slot comes back as "15:00 = 3:00 PM". Say the spoken half to the caller and
  pass the 24-hour half to book_appointment.
- The list of times the tool gives you is complete and exact. If a time is in that
  list it IS free — never tell the caller otherwise.
- Say times the way a person speaks them: "three in the afternoon", "half past ten".
  Never say "15:00" or "13:00" out loud.
{phone_rules}
- Confirm each thing as you collect it, then say back ONLY the {noun}, the day and
  the time before booking. Do not recite their name and number again — they have
  already heard the number read back, and a long recap is tiring on a phone.
- The moment they agree, call book_appointment straight away. Do not read the details
  back a second time and do not re-check a slot they have already chosen; confirming
  twice just wastes their time.
- You never need to ask the caller to wait. The moment you call a tool the system
  already says "let me check that for you" out loud on your behalf. So do not write
  "let me check", "one moment" or "please hold on" yourself — just call the tool.
- NEVER say you are about to do something without doing it in the same turn. If you
  do say "let me check" or "I'll go ahead and book that", the matching tool call must
  happen in that very same reply. Saying it and then stopping leaves the caller
  listening to silence with nothing on the way. If you are not calling a tool this
  turn, do not say you are — ask your next question instead.
- Never ask the caller to confirm something they have already confirmed, and never
  call the same tool twice in a row with the same arguments. If you already have the
  {noun}, a time from the tool, the name and the number, and the caller has agreed,
  the only correct next action is to call book_appointment.
- Every tool result includes a 'what_to_say' note. Follow it — it tells you how to
  handle a taken slot, a bad number, or the calendar being down.

CHANGING OR CANCELLING AN EXISTING APPOINTMENT
Cancelling cannot be undone, and a caller who is wrongly told their appointment is
gone simply will not turn up. So:
{find_step}
  2. Read back what it found — who it is with, the day and the time — and ask if
     that is the right one. If it found more than one, describe them and ask which.
     If it found none, say so and offer to take a message; do not guess.
  3. Only once they have clearly said yes to that specific appointment, call
     cancel_booking or reschedule_booking.
  4. To move one: check_availability for the new day first, offer times from it,
     and only call reschedule_booking once they pick one.
  5. If a move fails, say plainly that the original appointment still stands. Never
     leave them thinking they have no appointment when they do.
  6. After a successful move you get a NEW reference. Use that one if they change
     their mind again in the same call — the old one no longer works.
  7. Moving or cancelling needs ONLY the reference. The appointment already has
     their name and number on it, so do not ask for either again — asking for a
     phone number to cancel something is confusing and makes the caller think you
     have lost their booking."""


def build_system_prompt(profile, now=None, booking_enabled=False, offer_languages=None,
                        caller_number=None, knowledge_available=False):
    """Assemble the full system prompt: persona, facts, then rules.

    Args:
        profile: The parsed business profile.
        now: Current datetime. Defaults to the business's local time. Pass this
            explicitly so the date is computed per call rather than at import —
            a long-running server would otherwise serve a stale weekday, and the
            staff schedule is day-of-week dependent.
        booking_enabled: Whether the bot has the booking tools wired up. Adds the
            date table and the booking rules. Left off, the prompt is the Day-4
            read-only receptionist, which is still right for a business with no
            calendar behind it.
        offer_languages: Languages the caller may pick from, e.g.
            ["English", "Tamil", "Hindi"]. Only the Sarvam stack can hear and speak
            Indic languages, so only it passes this. Adds the language menu rules
            and expects a set_language tool to be registered.
        caller_number: The caller's own 10-digit number, when the phone line told
            us who is ringing. Removes the two-to-four turns spent capturing digits
            by voice — the weakest part of the booking flow, and weaker still at
            telephony's 8kHz. None for the browser or a withheld number, which
            falls back to asking.
        knowledge_available: Whether this business has documents ingested. Adds
            the LOOKING SOMETHING UP block and expects a search_knowledge tool.
            Off unless the store actually has rows for this business — telling
            the bot about a drawer that is empty only makes it open it.
    """
    if now is None:
        now = business_now(profile)

    biz = profile.get("business", {})
    assistant = profile.get("assistant", {})
    name = assistant.get("name", "the assistant")
    role = assistant.get("role", "receptionist")

    blocks = []

    intro = f"You are {name}, the {role} at {biz.get('name', 'this business')}"
    if biz.get("type"):
        intro += f", a {biz['type']}"
    if biz.get("address"):
        intro += f" at {biz['address']}"
    intro += ". You are speaking to a caller on the phone right now."
    blocks.append(intro)

    # Right after the persona and before the facts: how to speak, then what to
    # say. Warmth is mostly word choice — the Cartesia emotion setting shapes
    # delivery, but a bright voice reading a curt line still lands as curt.
    if assistant.get("tone"):
        blocks.append("HOW YOU SOUND\n" + _bullets(assistant["tone"]))

    blocks.append(f"TODAY\nIt is {_format_now(now)}.")

    if booking_enabled:
        blocks.append(date_reference(now))

    if biz.get("phone"):
        blocks.append(
            "OUR PHONE NUMBER\n"
            f"{biz['phone']}\n"
            f"Say it exactly like this and nothing else: {spoken_number(biz['phone'])}"
        )

    if profile.get("hours"):
        lines = [f"{day}: {hrs}" for day, hrs in profile["hours"].items()]
        lines += profile.get("hours_notes", [])
        blocks.append("OPENING HOURS\n" + "\n".join(lines))

    if profile.get("staff"):
        lines = []
        for person in profile["staff"]:
            line = person["name"]
            if person.get("specialty"):
                line += f" — {person['specialty']}"
            if person.get("days"):
                line += f" — in on {_join(person['days'])}"
            lines.append(line)
        blocks.append("WHO WORKS HERE\n" + "\n".join(lines))

    if profile.get("services"):
        lines = [_service_line(s) for s in profile["services"]]
        blocks.append("SERVICES, HOW LONG THEY TAKE, AND WHAT THEY COST\n" + "\n".join(lines))

    if profile.get("policies"):
        blocks.append("POLICIES\n" + _bullets(profile["policies"]))

    if profile.get("can_do"):
        blocks.append("YOU CAN HELP WITH\n" + _bullets(profile["can_do"]))

    if profile.get("cannot_do"):
        blocks.append("YOU MUST NOT DO ANY OF THIS\n" + _bullets(profile["cannot_do"]))

    if profile.get("not_offered"):
        blocks.append(
            "WE DO NOT OFFER THESE — say so plainly if asked\n"
            + _bullets(profile["not_offered"])
        )

    escalation = profile.get("escalation", {})
    if escalation.get("emergency"):
        blocks.append("EMERGENCIES\n" + escalation["emergency"])

    if offer_languages:
        blocks.append(multilingual_rules(offer_languages))

    if booking_enabled:
        blocks.append(booking_rules(staff_noun(profile), caller_number))

    # After the booking flow and before RULES: it is a lookup habit, not a
    # boundary, and it must not read as though it outranks "answer using ONLY
    # the information above".
    if knowledge_available:
        blocks.append(knowledge_rules())

    unknown_rule = escalation.get(
        "unknown",
        "Say you do not have that information and offer to take a message.",
    )

    blocks.append(
        "RULES — these override everything else\n"
        + _bullets([
            "Answer using ONLY the information above. It is the complete truth about "
            "this business; nothing else you believe about it is reliable.",
            # Without this it treats a garbled transcript as a question it cannot
            # answer, and says "I don't have that information" to noise. On a live
            # vet call that happened four times running before the caller gave up
            # and rephrased.
            "If what the caller said does not make sense as a sentence, or sounds "
            "like speech that came through badly, assume you misheard. Say you did "
            "not catch that and ask them to repeat it. Do not treat it as a question "
            "you cannot answer.",
            f"If you are asked a clear question the information above does not cover, "
            f"do not guess and do not invent an answer. {unknown_rule}",
            f"Never state a price, a duration or the name of a {staff_noun(profile)} "
            "that is not written above. If you are unsure, say you are not sure.",
            # Availability is the one fact that does NOT come from the profile once
            # tools are on — it comes from the calendar, and only from there.
            "Never state an appointment time unless the check_availability tool "
            "returned it in this conversation."
            if booking_enabled
            else "Never state an appointment availability that is not written above.",
            "You have no access to the internet, news, weather, sports, or any live "
            "information. If asked, say you cannot help with that.",
            "You do not know anything about the outside world — no current events, no "
            "general knowledge questions. Politely steer the caller back to how you "
            "can help them with the business.",
            "Do not guess dates or times other than the date given above.",
        ])
    )

    _, language_name = language(profile)
    speech_rules = []
    # When the caller picks the language mid-call, the LANGUAGE block above owns
    # these rules — repeating them here fixed to one language would contradict it.
    if offer_languages:
        language_name = "English"
    if language_name != "English":
        # The facts above are written in English; the caller is not speaking it.
        # Be explicit that names, prices and times must survive the translation.
        # Names and digits stay in English mid-sentence. Translating them produced
        # "Dr. Priya Nair" -> "ப்ரியா டைரியல்", the business name rendered as a
        # literal "sunrise", and a phone number with a digit dropped and another
        # changed. Code-switching like this is also how people actually speak here.
        speech_rules += [
            f"The caller speaks {language_name}. Reply in {language_name}.",
            f"Translate the business information above into natural {language_name} "
            f"as you speak — but never change a price, a duration, a time or a date.",
            f"Keep these in English, exactly as written above, even in the middle of a "
            f"{language_name} sentence: the names of people, the name of the business, "
            f"and the address. Never translate a name and never spell it in "
            f"{language_name} script — the caller has to be able to repeat it back.",
            f"Say every phone number and postcode in English digits, in the exact words "
            f"given above. Do not convert them into {language_name} numerals or words.",
        ]

    blocks.append(
        "HOW TO SPEAK\n"
        + _bullets(speech_rules + [
            "This is a phone call, so reply in ONE short spoken sentence, two at most.",
            "Never use lists, bullet points, numbering, markdown, or symbols like * or #.",
            "Say numbers, prices and times the way a person would say them out loud.",
            "Read postcodes, phone numbers and any other identifier one digit at a "
            "time — say '560034' as 'five six zero zero three four', never as a "
            "single large number. Read a leading '+' as 'plus'. Never add, drop or "
            "change a single digit; if you are not certain of the exact digits, say "
            "you would rather not risk getting the number wrong.",
            "Be warm and natural, not formal or robotic.",
            f"When the call begins, greet the caller briefly, say you are {name} at "
            f"{biz.get('name', 'the business')}, and ask how you can help.",
        ])
    )

    return "\n\n".join(blocks)


if __name__ == "__main__":
    print(build_system_prompt(load_profile()))
