"""The two buttons the model can press, and what happens when it does.

Split in two deliberately:

  * `check_availability` / `book_appointment` are plain async functions taking
    ordinary arguments. They can be called straight from a test script, which is
    how the text harness proves the flow without a microphone.
  * `build_tools()` wraps them for Pipecat.

Handlers never raise. A raise inside a function call would break the pipeline
mid-sentence; instead every failure comes back as a result the model can read and
speak, with an explicit `what_to_say` so it doesn't have to invent a recovery.
"""

from datetime import datetime

from loguru import logger

from calendar_backend import (
    BookingNotFound,
    CalendarError,
    CalendarUnavailable,
    DoctorNotFound,
    InvalidPhone,
    SlotTaken,
)

CHECK_AVAILABILITY = {
    "name": "check_availability",
    "description": (
        "Look up the open appointment slots for one {noun} on one date. Call this "
        "whenever the caller asks what times are free, and always before offering "
        "the caller any appointment time. Never guess a time without calling this."
    ),
    "properties": {
        "staff_member": {
            "type": "string",
            "description": "Full name of the {noun}, exactly as written in who works here",
        },
        "date": {
            "type": "string",
            "description": "The date to check, in YYYY-MM-DD format",
        },
        "preferred_time": {
            "type": "string",
            "description": (
                "Optional. If the caller hinted at a time — 'in the morning', "
                "'around five' — pass it as HH:MM 24-hour, e.g. '17:00'. Omit it "
                "if they only named a day. It only affects which times get "
                "suggested back to you."
            ),
        },
    },
    "required": ["staff_member", "date"],
}

BOOK_APPOINTMENT = {
    "name": "book_appointment",
    "description": (
        "Create a real appointment in the business's calendar. Only call this once "
        "you have all four details and the caller has agreed to a specific slot "
        "that check_availability returned."
    ),
    "properties": {
        "name": {"type": "string", "description": "The caller's full name"},
        "phone": {
            "type": "string",
            "description": "The caller's 10-digit Indian mobile number, digits only",
        },
        "staff_member": {"type": "string", "description": "Full name of the {noun}"},
        "datetime": {
            "type": "string",
            "description": "Appointment start, YYYY-MM-DD HH:MM in 24-hour time",
        },
        "reason": {
            "type": "string",
            "description": (
                "One short line on what the appointment is for, in the caller's own "
                "words — 'hair falling out for a few months', 'annual check-up', "
                "'follow-up on knee pain'. Write it from what they have ALREADY told "
                "you. Do NOT ask them a separate question for this, and do not invent "
                "a reason: if they never said why, put 'not given'."
            ),
        },
        "also_book": {
            "type": "boolean",
            "description": (
                "Almost always leave this out. Set it to true ONLY when the caller "
                "already has an appointment and has clearly said they want a second, "
                "separate one as well as it. Never set it to move an existing "
                "appointment — that is reschedule_booking."
            ),
        },
    },
    # `reason` is required so the model always writes one rather than treating it
    # as optional and skipping it. The description is what stops it costing a
    # turn — it must be written from what the caller already said, never asked.
    "required": ["name", "phone", "staff_member", "datetime", "reason"],
}


SEARCH_KNOWLEDGE = {
    "name": "search_knowledge",
    "description": (
        "Look up detailed information about this business that is NOT written in "
        "your instructions — parking and directions, insurance and reimbursement, "
        "prescription refills, accessibility, what to bring, how results are "
        "handled, waiting times. Call this whenever the caller asks something your "
        "instructions do not answer, before saying you do not know. "
        "Do NOT call it for opening hours, prices, services, {noun} names or days, "
        "or whether the business offers something — all of that is already in your "
        "instructions and what is written there is complete."
    ),
    "properties": {
        "question": {
            "type": "string",
            "description": "What the caller asked, in their own words",
        },
    },
    "required": ["question"],
}

FIND_BOOKING = {
    "name": "find_booking",
    "description": (
        "Look up a caller's upcoming appointments by their name. Call this first "
        "whenever someone wants to change or cancel an appointment — you cannot "
        "cancel or move anything without the reference this returns."
    ),
    "properties": {
        "name": {"type": "string", "description": "The caller's name as they gave it"},
    },
    "required": ["name"],
}

CANCEL_BOOKING = {
    "name": "cancel_booking",
    "description": (
        "Cancel an existing appointment. This cannot be undone. Only call it after "
        "find_booking has given you the reference AND the caller has confirmed that "
        "is the appointment they mean."
    ),
    "properties": {
        "reference": {
            "type": "string",
            "description": "The reference from find_booking. Never invent one.",
        },
    },
    "required": ["reference"],
}

RESCHEDULE_BOOKING = {
    "name": "reschedule_booking",
    "description": (
        "Move an existing appointment to a different time. Only call it after "
        "find_booking has given you the reference, check_availability has shown the "
        "new time is free, and the caller has agreed to it."
    ),
    "properties": {
        "reference": {
            "type": "string",
            "description": "The reference from find_booking. Never invent one.",
        },
        "datetime": {
            "type": "string",
            "description": "The new start time, YYYY-MM-DD HH:MM in 24-hour time",
        },
    },
    "required": ["reference", "datetime"],
}

SET_LANGUAGE = {
    "name": "set_language",
    "description": (
        "Switch the phone line to another language. Call this as soon as the caller "
        "says which language they want. Speech recognition and the voice both change, "
        "so you must call this before you start speaking the new language."
    ),
    "properties": {
        "language": {
            "type": "string",
            "description": "The language the caller chose, e.g. 'Tamil', 'Hindi', 'English'",
        },
    },
    "required": ["language"],
}


def spec_for(spec, noun):
    """Fill the business's word for its people into a tool spec.

    Descriptions and parameter docs are prompt surface — the model reads them — so
    a salon must see "stylist" here, not "doctor".
    """
    return {
        **spec,
        "description": spec["description"].format(noun=noun),
        "properties": {
            key: {**value, "description": value["description"].format(noun=noun)}
            for key, value in spec["properties"].items()
        },
    }


def _pair(slot):
    """'15:00' -> '15:00 = 3:00 PM'."""
    return f"{slot} = {datetime.strptime(slot, '%H:%M').strftime('%I:%M %p').lstrip('0')}"


def _pick_suggested(slots, preferred=None, count=3):
    """Choose the handful of times to actually read out.

    Done here rather than left to the model. Asked to pick a spread from twenty
    times, both Groq and gpt-4o-mini read out the first three consecutive slots
    and then invented a range — "nine, half past nine, ten, and any time until
    six thirty" — which is the one phrasing the prompt forbids. Choosing in code
    makes it deterministic and reduces the prompt to "read these out".
    """
    if len(slots) <= count:
        return slots
    if preferred:
        try:
            want = datetime.strptime(preferred, "%H:%M")
            nearest = sorted(slots, key=lambda s: abs(datetime.strptime(s, "%H:%M") - want))
            return sorted(nearest[:count])
        except ValueError:
            pass  # unparseable hint — fall through to an even spread
    # Middle of each third. Index-based rather than hardcoded morning/afternoon
    # boundaries, so it still spreads sensibly for a business open unusual hours.
    n = len(slots)
    return [slots[int((i + 0.5) * n / count)] for i in range(count)]


def _speakable(when):
    """'2026-08-11 15:00' -> ('Tuesday 11 August', '3:00 PM') for saying out loud."""
    dt = datetime.strptime(when, "%Y-%m-%d %H:%M")
    return (
        f"{dt.strftime('%A')} {dt.day} {dt.strftime('%B')}",
        dt.strftime("%I:%M %p").lstrip("0"),
    )


def _offline():
    return {
        "ok": False,
        "problem": "the calendar system is not responding",
        "what_to_say": (
            "Apologise, say you cannot reach the appointment system right now, and "
            "offer to take their name and number so the business can call them back."
        ),
    }


async def check_availability(calendar, doctor, date, preferred_time=None):
    """Open slots for one doctor on one date."""
    try:
        slots = await calendar.get_available_slots(doctor, date)
    except CalendarUnavailable:
        return _offline()
    except DoctorNotFound as e:
        return {
            "ok": False,
            "problem": str(e),
            "what_to_say": "Tell the caller that person does not work here, and say who does.",
        }
    except CalendarError as e:
        return {"ok": False, "problem": str(e), "what_to_say": "Ask the caller to repeat the day they want."}

    day = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
    # Give both clock forms on every slot. With bare 24-hour times, both Groq and
    # gpt-4o-mini told a caller who asked for "three in the afternoon" that it was
    # unavailable while 15:00 sat in the list — they were not reliably mapping
    # spoken time to 24-hour. Spelling out the pairing removes the translation step.
    result = {
        "ok": True,
        "staff_member": doctor,
        "date": date,
        "day": day,
        # `suggested` is what to read out; `available` is the full set, kept so the
        # bot can still answer "is five in the evening free?" accurately.
        "suggested": [_pair(s) for s in _pick_suggested(slots, preferred_time)],
        "available": [_pair(s) for s in slots],
        "booking_format": "Pass the 24-hour side of the pair to book_appointment, e.g. '15:00'.",
        "what_to_say": (
            "Offer the times in 'suggested' and nothing else — do not list more, and "
            "never describe a range like 'any time until six thirty'. Then ask if "
            "another time would suit them better. Use 'available' only to answer "
            "whether a specific time the caller names is free."
        ),
    }

    if not slots:
        # Say *why* there is nothing, and when there would be — otherwise the model
        # has to guess at an alternative, which is where invention creeps in.
        person = calendar._resolve_doctor(doctor)
        works = person.get("days", [])
        if day not in works:
            result["reason"] = f"{person['name']} does not work on {day}s."
            result["works_on"] = works
            result["what_to_say"] = f"Say they are not in that day, and offer the days they are in."
        else:
            result["reason"] = f"{person['name']} is fully booked on {date}."
            result["what_to_say"] = "Say that day is full and ask if another day works."
    return result


async def book_appointment(calendar, name, phone, doctor, datetime_str, reason="",
                           also_book=False):
    """Commit a booking. Returns a confirmation, or a problem the bot must relay.

    Refuses if this number already has an upcoming appointment, because on a
    live call that meant a reschedule going out as a second booking: the caller
    was told their Thursday appointment had moved to Saturday, and ended up with
    both. The clinic got a no-show and a surprise. Rule 7 of the prompt says not
    to do this in plain words and the model did it anyway — a numbered BOOKING
    procedure outranks a prose rule — so it stops being a prompt problem.

    `also_book` is the escape hatch for someone who genuinely wants a second,
    separate appointment. The model has to ask for it deliberately.
    """
    if phone and not also_book:
        try:
            existing = await calendar.find_bookings_by_phone(phone)
        except CalendarError:
            # A flaky lookup must not block a real booking. Losing the
            # duplicate check is a smaller failure than refusing everyone.
            logger.warning("duplicate check failed — booking anyway")
            existing = []
        if existing:
            clash = _summary_of(existing[0])
            logger.info(f"refusing double booking — {phone} already has {clash['reference']}")
            return {
                "ok": False,
                "problem": "this caller already has an upcoming appointment",
                "existing": clash,
                "what_to_say": (
                    f"Do NOT book. They already have an appointment with "
                    f"{clash['with']} on {clash['day']} at {clash['time']}. If they "
                    f"are moving that one, call reschedule_booking with reference "
                    f"{clash['reference']} and the new time — that moves it properly "
                    f"instead of leaving them with two. Only if they clearly want an "
                    f"ADDITIONAL appointment as well as that one, call "
                    f"book_appointment again with also_book set to true."
                ),
            }
    try:
        booking = await calendar.create_booking(name, phone, doctor, datetime_str, reason)
    except CalendarUnavailable:
        return _offline()
    except InvalidPhone:
        return {
            "ok": False,
            "problem": "that phone number is not a valid 10-digit Indian mobile number",
            "what_to_say": (
                "Do not book. Tell the caller the number did not sound right, ask them "
                "to repeat it slowly, and read it back one digit at a time."
            ),
        }
    except SlotTaken:
        # Hand back real alternatives so 'offer another time' is grounded in the
        # calendar rather than improvised.
        date = datetime_str.split(" ")[0]
        wanted_time = datetime_str.split(" ")[-1]
        try:
            alternatives = await calendar.get_available_slots(doctor, date)
        except CalendarError:
            alternatives = []
        # Never offer back the slot we just refused — it reads as a contradiction.
        alternatives = [s for s in alternatives if s != wanted_time]
        # Nearest to the time they actually asked for — someone who wanted 3pm
        # does not want to hear about nine in the morning first.
        try:
            wanted = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            alternatives.sort(
                key=lambda s: abs(datetime.strptime(f"{date} {s}", "%Y-%m-%d %H:%M") - wanted)
            )
        except ValueError:
            pass
        return {
            "ok": False,
            "problem": "that slot has just been taken",
            "alternatives": [
                f"{s} = {datetime.strptime(s, '%H:%M').strftime('%I:%M %p').lstrip('0')}"
                for s in alternatives[:4]
            ],
            "what_to_say": (
                "Say that time is no longer free, then offer two of the alternatives "
                "listed. Do not offer any time that is not in that list."
            ),
        }
    except DoctorNotFound as e:
        return {
            "ok": False,
            "problem": str(e),
            "what_to_say": "Tell the caller that person does not work here, and say who does.",
        }
    except CalendarError as e:
        return {"ok": False, "problem": str(e), "what_to_say": "Ask the caller to confirm the day and time again."}

    spoken_day, spoken_time = _speakable(booking["datetime"])
    return {
        "ok": True,
        "booked": True,
        "reference": booking["id"],
        "name": booking["name"],
        "phone": booking["phone"],
        "staff_member": booking["doctor"],
        "day": spoken_day,
        "time": spoken_time,
        "what_to_say": (
            "Confirm the booking warmly in one sentence: who it is with, the day and the "
            "time. Do not read out the reference number unless the caller asks."
        ),
    }


def _summary_of(booking):
    """One booking in the shape the model should read out."""
    day, time = _speakable(booking["datetime"])
    return {
        "reference": booking["id"],
        "with": booking.get("doctor", ""),
        "day": day,
        "time": time,
        "name": booking.get("name", ""),
    }


async def search_knowledge(business_id, question):
    """Look in the business's document store for something the prompt lacks.

    Runs the lookup off the event loop. `knowledge.search` embeds the question
    over HTTP and then queries Postgres, both synchronous; calling them inline
    would stall the pipeline, and a stalled pipeline on a phone call is silence
    the caller hears.

    The result carries an explicit reminder that the profile outranks anything
    retrieved. That is not decoration. Measured on the clinic document, "can I
    get an X-ray here?" matches the outside-labs paragraph at 0.635 — a perfectly
    ordinary similarity score, indistinguishable by distance from a real answer.
    That paragraph says scans are done at external centres, which a model could
    relay as "yes, we arrange those" while the profile's not_offered list says no
    radiology of any kind. No threshold separates the two, so the ranking is
    stated at the point of use instead.
    """
    import asyncio

    import knowledge

    try:
        # Two passages, not three: this is a spoken answer, and more text is
        # more room for the model to wander off the question.
        hits = await asyncio.to_thread(knowledge.search, business_id, question, 2)
    except Exception as e:
        # A knowledge store being down must never take a call with it. The bot
        # falls back to exactly what it did before any of this existed.
        logger.warning(f"knowledge lookup failed: {e}")
        return {
            "ok": False,
            "found": False,
            "what_to_say": (
                "Say you do not have that information to hand and offer to take a "
                "message for the staff. Do not guess."
            ),
        }

    if not hits:
        return {
            "ok": True,
            "found": False,
            "what_to_say": (
                "Nothing on file about that. Say you do not have that information "
                "and offer to take a message for the staff. Do not guess, and do "
                "not assume this means the business does or does not do it."
            ),
        }

    return {
        "ok": True,
        "found": True,
        "passages": [h["content"] for h in hits],
        "what_to_say": (
            # Scoped hard to THIS answer. An earlier wording said "answer from
            # these passages only" and "do not add anything of your own", which
            # was meant for one reply but stays in context for the rest of the
            # call — after one lookup the model started refusing to answer from
            # its own instructions at all, saying "I can't share that" about
            # prices printed in the prompt. Reproduced 0 out of 3 after a lookup
            # against 3 out of 3 before one.
            "For THIS question, answer from the passages below: say the part that "
            "answers it in one or two spoken sentences, and do not read them out "
            "word for word. "
            "Your own instructions are unaffected and still complete — keep "
            "answering from them as normal for everything else, including opening "
            "hours, prices, services and who works here. These passages are extra "
            "detail for this one question, not a replacement for what you know. "
            "They are background details, NOT a list of what the business offers. "
            "If anything here seems to suggest a service that your instructions say "
            "is not offered, your instructions are right and this is not — say the "
            "business does not offer it."
        ),
    }


def _despell(name):
    """'A V E E N O' -> 'AVEENO'. Leaves 'Priya Nair' alone.

    When a caller spells their name out, the model passes the letters through
    separated by spaces, and no calendar will match that against the stored
    name — Cal.com filters on attendeeName as a literal string. On a live call
    the caller finally spelled it correctly and the lookup still failed for
    exactly this reason. Only collapses when most tokens are single characters,
    so real multi-word names survive.
    """
    parts = (name or "").split()
    if len(parts) >= 3 and sum(len(p) == 1 for p in parts) >= len(parts) - 1:
        # .title(), not the letters as given: Cal.com matches attendeeName
        # case-sensitively, so "AVEENO" misses a booking stored as "Aveeno".
        # Spelled-out letters carry no real case anyway.
        return "".join(parts).title()
    return name


async def find_booking(calendar, name, caller_number=None):
    """The caller's upcoming appointments.

    Tries the number the phone line gave us before the name it heard. The
    number is evidence; a name spoken over 8kHz is a guess, and on a live call
    four spellings in a row missed a booking that was sitting there under the
    caller's own number. Name lookup stays as the fallback — people ring about
    a spouse's appointment, or from a different phone.
    """
    matched_on = "phone"
    try:
        found = []
        if caller_number:
            found = await calendar.find_bookings_by_phone(caller_number)
        if not found and name:
            matched_on = "name"
            found = await calendar.find_bookings(_despell(name))
    except CalendarUnavailable:
        return _offline()
    except CalendarError as e:
        return {"ok": False, "problem": str(e), "what_to_say": "Ask the caller for their name again."}

    if not found:
        return {
            "ok": True,
            "bookings": [],
            "what_to_say": (
                "Tell the caller you cannot find an appointment"
                + (" under that name" if name else " on the number they are calling from")
                + ", and ask whether it might have been booked under a different name "
                "or number. Do not guess at an appointment."
            ),
        }
    if matched_on == "phone":
        return {
            "ok": True,
            "bookings": [_summary_of(b) for b in found],
            "what_to_say": (
                "Found from the number they are calling from, so this is their "
                "appointment — do NOT ask for their name or number to confirm it. "
                "Greet them by the name on the booking and read the appointment "
                "back: who it is with, the day and the time. Ask if that is the one "
                "they mean. If there is more than one, describe them and ask which. "
                "Wait for a clear yes before cancelling or moving anything. This "
                "reference is all you need to cancel or move it."
            ),
        }
    return {
        "ok": True,
        "bookings": [_summary_of(b) for b in found],
        "what_to_say": (
            "Read the appointment back — who it is with, the day and the time — and "
            "ask if that is the right one. Wait for a clear yes before cancelling or "
            "moving anything. If there is more than one, describe them and ask which. "
            "Never act on the name alone. "
            # The numbered booking flow tells the model to collect a name and mobile,
            # and it kept doing that here too — asking a caller for their number in
            # order to cancel, which reads as though their booking has been lost.
            # Said at the moment of the lookup it sticks; said in the prompt it did not.
            "You now have everything you need: this reference is enough to cancel or "
            "move it. Do NOT ask for their phone number or name again — the "
            "appointment already carries them."
        ),
    }


async def cancel_booking(calendar, reference):
    """Cancel. Irreversible, so the caller must already have confirmed."""
    try:
        booking = await calendar.cancel_booking(reference)
    except CalendarUnavailable:
        return _offline()
    except BookingNotFound:
        return {
            "ok": False,
            "problem": "that appointment is no longer there",
            "what_to_say": (
                "Say the appointment does not seem to be in the system any more, and "
                "offer to look it up again. Do not claim it was cancelled."
            ),
        }
    except CalendarError as e:
        return {"ok": False, "problem": str(e),
                "what_to_say": "Say you could not cancel it and offer to take a message."}

    day, time = _speakable(booking["datetime"])
    return {
        "ok": True, "cancelled": True, "day": day, "time": time,
        "with": booking.get("doctor", ""),
        "what_to_say": "Confirm in one sentence that it is cancelled, and say the day and time it was.",
    }


async def reschedule_booking(calendar, reference, datetime_str):
    """Move an appointment. The new slot must already be free."""
    try:
        booking = await calendar.reschedule_booking(reference, datetime_str)
    except CalendarUnavailable:
        return _offline()
    except BookingNotFound:
        return {
            "ok": False,
            "problem": "that appointment is no longer there",
            "what_to_say": "Say you cannot find it any more and offer to look it up again.",
        }
    except SlotTaken:
        # No alternatives offered here: a reference does not tell us whose diary it
        # is in, and guessing the wrong person's free slots would be worse than
        # asking. The model can call check_availability itself.
        return {
            "ok": False,
            "problem": "that new time is not free",
            "what_to_say": (
                "Say the new time is not available and — importantly — that the "
                "appointment has NOT been moved, it still stands at the original "
                "time. Then call check_availability to offer times that are free."
            ),
        }
    except CalendarError as e:
        return {"ok": False, "problem": str(e),
                "what_to_say": "Say you could not move it, and that the original appointment still stands."}

    day, time = _speakable(booking["datetime"])
    return {
        "ok": True, "rescheduled": True, "day": day, "time": time,
        "with": booking.get("doctor", ""),
        # Cal.com issues a fresh uid on reschedule and retires the old one, so a
        # later cancel in the same call must use this reference, not the first.
        "reference": booking["id"],
        "what_to_say": (
            "Confirm the new day and time in one sentence. If they ask you to change "
            "or cancel it again, use the reference in this result, not the earlier one."
        ),
    }


async def switch_language(params, chosen, offered):
    """Retune speech recognition and the voice to the caller's language.

    The model can only change the words it writes; the two ends of the phone line
    have to be changed underneath it. STT sits upstream of the LLM and TTS
    downstream, so the frames go in opposite directions.
    """
    from pipecat.frames.frames import STTUpdateSettingsFrame, TTSUpdateSettingsFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.transcriptions.language import Language

    from prompt import CODE_FOR_LANGUAGE

    wanted = (chosen or "").strip().title()
    code = CODE_FOR_LANGUAGE.get(wanted)
    if code is None or wanted not in offered:
        return {
            "ok": False,
            "problem": f"{chosen!r} is not one of the languages this line can speak",
            "what_to_say": f"Apologise and ask them to choose between {', '.join(offered)}.",
        }

    language = Language(code)
    await params.llm.push_frame(
        STTUpdateSettingsFrame(delta={"language": language}), FrameDirection.UPSTREAM)
    await params.llm.push_frame(
        TTSUpdateSettingsFrame(delta={"language": language}), FrameDirection.DOWNSTREAM)
    logger.info(f"Language switched to {wanted} ({code})")

    return {
        "ok": True,
        "language": wanted,
        "what_to_say": (
            f"The line is now in {wanted}. Speak only {wanted} from here on. Carry on "
            f"with what the caller was asking — do not greet them again. Keep names, "
            f"the business name, the address and all phone numbers in English."
        ),
    }


def _find_with_caller_id(spec, caller_number):
    """Make FIND_BOOKING's `name` optional when the phone line identified them.

    Same reasoning as _with_caller_id: a required argument is a stronger
    instruction than any prompt line, so while `name` stays required the model
    will keep asking callers to spell it — which is the exact failure this
    change exists to remove.
    """
    if not caller_number:
        return spec
    return {
        **spec,
        "description": (
            "Look up the caller's upcoming appointments. Call this immediately "
            "when someone wants to change or cancel — you already know who is "
            "calling, so do NOT ask for their name first. Only pass a name if "
            "the first call found nothing and they say it was booked under "
            "someone else."
        ),
        "properties": {
            **spec["properties"],
            "name": {
                "type": "string",
                "description": (
                    "Leave this out. The caller is identified by the number they "
                    "are calling from. Only pass a name if a call without one "
                    "found nothing."
                ),
            },
        },
        "required": [key for key in spec["required"] if key != "name"],
    }


def _with_caller_id(spec, caller_number):
    """Drop `phone` from BOOK_APPOINTMENT's required list when we already know it.

    Leaving it required makes the model invent digits rather than omit the field —
    a required argument is a strong instruction, stronger than any prompt line
    telling it to leave the argument out.
    """
    if not caller_number:
        return spec
    return {
        **spec,
        "properties": {
            **spec["properties"],
            "phone": {
                "type": "string",
                "description": (
                    "Leave this out. The caller's own number is used automatically. "
                    "Only pass a number if the caller explicitly asked to be reached "
                    "on a different one."
                ),
            },
        },
        "required": [key for key in spec["required"] if key != "phone"],
    }


def build_tools(calendar, offer_languages=None, caller_number=None, business_id=None):
    """Return (ToolsSchema, {name: pipecat_handler}) for this calendar.

    Imported lazily so the plain functions above stay usable in a test script
    without pulling in Pipecat.

    `caller_number` is the number the phone line says the call came from. When we
    have it, book_appointment stops requiring a phone argument and falls back to
    it — so the booking carries the right number whether or not the model plays
    along. The prompt asks; this guarantees.

    `business_id` adds search_knowledge over that business's documents. Left None
    the tool is absent entirely, which is what a business with nothing ingested
    should get — a tool that can only ever come back empty is worse than no tool,
    because the model will keep reaching for it.
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    from prompt import staff_noun

    noun = staff_noun(calendar._profile)
    schemas = [
        FunctionSchema(
            name=spec["name"],
            description=spec["description"],
            properties=spec["properties"],
            required=spec["required"],
        )
        for spec in (spec_for(CHECK_AVAILABILITY, noun),
                     _with_caller_id(spec_for(BOOK_APPOINTMENT, noun), caller_number),
                     _find_with_caller_id(spec_for(FIND_BOOKING, noun), caller_number),
                     spec_for(CANCEL_BOOKING, noun),
                     spec_for(RESCHEDULE_BOOKING, noun))
    ]
    if offer_languages:
        schemas.append(FunctionSchema(
            name=SET_LANGUAGE["name"],
            description=SET_LANGUAGE["description"],
            properties=SET_LANGUAGE["properties"],
            required=SET_LANGUAGE["required"],
        ))

    if business_id:
        spec = spec_for(SEARCH_KNOWLEDGE, noun)
        schemas.append(FunctionSchema(
            name=spec["name"],
            description=spec["description"],
            properties=spec["properties"],
            required=spec["required"],
        ))

    async def run_check(params):
        args = params.arguments or {}
        logger.info(f"tool check_availability({args})")
        result = await check_availability(
            calendar,
            args.get("staff_member", ""),
            args.get("date", ""),
            args.get("preferred_time") or None,
        )
        await params.result_callback(result)

    async def run_book(params):
        args = params.arguments or {}
        logger.info(f"tool book_appointment({args})")
        # Caller ID is the default, not an override: a caller booking for a parent
        # or ringing from an office line can still give a different number, and
        # whatever the model passes wins. But if it passes nothing — or passes the
        # empty string it sometimes emits for an optional field — the number the
        # phone line gave us is used, so a booking can never land without one.
        phone = (args.get("phone") or "").strip() or caller_number or ""
        if caller_number and phone != args.get("phone"):
            logger.info(f"using caller ID for phone (model sent {args.get('phone')!r})")
        result = await book_appointment(
            calendar,
            args.get("name", ""),
            phone,
            args.get("staff_member", ""),
            args.get("datetime", ""),
            (args.get("reason") or "").strip(),
            also_book=bool(args.get("also_book")),
        )
        await params.result_callback(result)

    async def run_set_language(params):
        args = params.arguments or {}
        logger.info(f"tool set_language({args})")
        await params.result_callback(
            await switch_language(params, args.get("language", ""), offer_languages))

    async def run_find(params):
        args = params.arguments or {}
        logger.info(f"tool find_booking({args})")
        await params.result_callback(
            await find_booking(calendar, args.get("name", ""), caller_number))

    async def run_cancel(params):
        args = params.arguments or {}
        logger.info(f"tool cancel_booking({args})")
        await params.result_callback(await cancel_booking(calendar, args.get("reference", "")))

    async def run_reschedule(params):
        args = params.arguments or {}
        logger.info(f"tool reschedule_booking({args})")
        await params.result_callback(await reschedule_booking(
            calendar, args.get("reference", ""), args.get("datetime", "")))

    async def run_search_knowledge(params):
        args = params.arguments or {}
        logger.info(f"tool search_knowledge({args})")
        await params.result_callback(
            await search_knowledge(business_id, args.get("question", "")))

    handlers = {
        "check_availability": run_check,
        "book_appointment": run_book,
        "find_booking": run_find,
        "cancel_booking": run_cancel,
        "reschedule_booking": run_reschedule,
    }
    if offer_languages:
        handlers["set_language"] = run_set_language
    if business_id:
        handlers["search_knowledge"] = run_search_knowledge
    return ToolsSchema(standard_tools=schemas), handlers




if __name__ == "__main__":
    import asyncio
    import json
    from datetime import timedelta
    from pathlib import Path

    from calendar_backend import MockCalendar
    from prompt import business_now, load_profile

    profile = load_profile()
    Path("bookings.demo.json").unlink(missing_ok=True)
    cal = MockCalendar(profile, store_path="bookings.demo.json")
    now = business_now(profile)
    tue = (now + timedelta(days=(1 - now.weekday()) % 7 or 7)).strftime("%Y-%m-%d")
    mon = (now + timedelta(days=(0 - now.weekday()) % 7 or 7)).strftime("%Y-%m-%d")

    async def show(label, coro):
        print(f"\n--- {label}")
        print(json.dumps(await coro, indent=2)[:520])

    async def demo():
        await show("free slots, Tuesday", check_availability(cal, "Dr. Priya Nair", tue))
        await show("doctor off that day", check_availability(cal, "Dr. Priya Nair", mon))
        await show("invented doctor", check_availability(cal, "Dr. Kapoor", tue))
        await show("good booking", book_appointment(cal, "Abinav", "9876543210", "Dr. Nair", f"{tue} 15:00"))
        await show("same slot again", book_appointment(cal, "Ravi", "9876543211", "Dr. Nair", f"{tue} 15:00"))
        await show("bad phone", book_appointment(cal, "Abinav", "12345", "Dr. Nair", f"{tue} 16:00"))
        cal.force_offline = True
        await show("calendar offline", book_appointment(cal, "Abinav", "9876543210", "Dr. Nair", f"{tue} 16:00"))

    asyncio.run(demo())
    Path("bookings.demo.json").unlink(missing_ok=True)
