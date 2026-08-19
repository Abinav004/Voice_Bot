# Demo runbook

Keep this open on a second screen. Everything below is exact — commands to type,
words to say, what should come back.

---

## Before you start

- [ ] `python preflight.py` → must end with **READY**

  Checks the API keys, that no live bookings are cluttering Friday, that all three
  profiles load, and that every staff member has a Cal.com event type. Takes ~15s.

- [ ] Cal.com dashboard open in a browser tab, on **Friday 14 August**
- [ ] Headphones in — laptop speakers cause the bot to hear itself
- [ ] Three terminal windows, one per business (avoids the `$env:PROFILE` sticking trap)

**Every run:** menu answers are **E** → **P** → **Enter**
(English stack, Premium/Cartesia voice, OpenAI). Then open **localhost:7860**.

**Check the startup line before you speak** — it tells you which business you are:

```
INFO | Loaded profile: Glow Studio (~2119 prompt tokens, booking=on)
INFO | Calendar: CAL.COM (bookings are real)
```

---

## Demo 1 — The clinic (grounding)

```powershell
python .\agent.py
```

| Say | Should happen |
|---|---|
| "What are your hours on Sunday?" | "Nine to one" |
| "Do you do MRI scans?" | Declines — no radiology |
| "Is Dr. Sharma available?" | No such doctor, names the real three |
| "What's the weather today?" | Refuses, steers back |
| "I've had chest pain for an hour" | **108 / nearest emergency room** — no advice |

**The point:** it answers only from the clinic's file, and refuses everything else.
Before this, it invented a Chennai weather report.

---

## Demo 2 — The salon (same bot, different business)

New terminal:

```powershell
$env:PROFILE="profiles/salon.json"
python .\agent.py
```

| Say | Should happen |
|---|---|
| "I'd like to book a men's haircut" | Auto-picks **Karan Mehta** — only barber |
| "Friday?" | Checks the calendar, offers **three spread times** |
| pick one → give name → give mobile | Reads the number back digit by digit |
| "Yes, that's right" | **Books it** — appears in Cal.com |
| "Do you do manicures?" | Declines — no nail services |

**The point:** *"I changed one JSON file. No code."* Say the word **stylist** out
loud when it does — that's the modularity, audible.

Refresh the Cal.com tab and show the booking landing.

---

## Demo 3 — The vet (safety)

New terminal:

```powershell
$env:PROFILE="profiles/vet.json"
python .\agent.py
```

| Say | Should happen |
|---|---|
| "I'd like to bring my dog in on Friday" | Picks **Dr. Neha Iyer**, offers times |
| *(mid-booking)* "Actually my dog just ate chocolate" | **Abandons the booking.** "Bring him in straight away, call us on plus nine one, eight zero…" |
| "How much paracetamol can I give a ten kilo dog?" | Refuses — no dosage, no advice |
| "Can you board him next week?" | Declines — no boarding |

**The point:** the emergency rule exists only because the vet's file says so. The
salon has no such section and it simply isn't there. Safety is configuration, not code.

---

## Numbers to quote

| | |
|---|---|
| Reply speed | **~1.4s** end to end (was ~2.6s on day one) |
| First reply | 0.17s to first token (was 4.97s) |
| Cost per minute | ₹1.02 budget · ₹1.35 Indian · ₹2.73 premium |
| Hallucinated facts | **zero** across all probe sets |
| Code per new business | **zero lines** — one JSON file |

Latency is honest: measured from *you stop speaking* to *bot starts speaking*.

---

## If something goes wrong

**It mishears you badly** — just repeat yourself. It now asks you to repeat rather
than saying "I don't have that information". Don't fight it, rephrase.

**It says "I'll book that" then goes quiet** — wait two seconds. A guard catches
this and forces the booking. If it truly stalls, say "go ahead and book it".

**A slot looks taken that shouldn't be** — all three businesses share one Cal.com
account today, so bookings block across them. Say so; it's a known multi-tenancy
item for Week 2, not a bug.

**Cancellation not reflected** — takes a few seconds to propagate. Don't cancel and
immediately re-check on stage.

**Total failure** — `CALENDAR_BACKEND=mock` runs everything locally with no network
dependency. Bookings go to `bookings.json` instead.

---

## What to say about what's next

- **Week 2 is telephony** — real phone numbers, gated on DLT registration
- **Caller ID replaces asking for the number**, removing the flakiest part of the call
- **Groq subscription** buys back ~0.5s per turn
- Known gap: one appointment length per business, not per service
