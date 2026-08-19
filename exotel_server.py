"""The phone front door.

A second way in for the same bot. agent.py serves a browser over WebRTC; this
serves a real phone call over Exotel's WebSocket. Everything behind the transport
— profile, six tools, Cal.com, the promise guard, Supabase logging — comes from
agent.build_call(), so the phone bot cannot drift from the browser one.

    uvicorn exotel_server:app --host 0.0.0.0 --port 8000
    ngrok http 8000

Two lines, one process:

    wss://<domain>/ws          English — Deepgram + Cartesia
    wss://<domain>/ws/indian   Sarvam — asks English, Tamil or Hindi

Paste whichever you want into the Exotel Voicebot applet. Two routes rather than
two servers because one ExoPhone and one free ngrok tunnel cannot reach two
processes — but they reach two paths fine, and a second ExoPhone pointed at the
other path later needs no code change.

Telephony is 8kHz, a quarter the detail the browser gave us. ExotelFrameSerializer
resamples in both directions; expect speech recognition to be less forgiving than
it has been, especially on digits.
"""

import asyncio
import json
import os
from pathlib import Path

# Must come before any pipecat import: pipecat pulls in nltk, whose import hook
# refuses to load `regex` from the working directory and kills the server at
# startup. agent.py does the same on its first line for the same reason.
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, WebSocket  # noqa: E402
from loguru import logger  # noqa: E402

load_dotenv(override=True)

# Set before importing agent: build_stt_tts() and load_profile() read these at
# call time, and the phone line needs the English stack — Sarvam at 8kHz is
# untested. setdefault so an explicit environment variable still wins.
os.environ.setdefault("PROFILE", "profiles/clinic.json")
os.environ.setdefault("TTS_TIER", "premium")   # Cartesia resamples cleanly to 8k
os.environ.setdefault("LLM_PROVIDER", "openai")
# No TIER here any more: each route names its own stack when it calls build_call,
# because one process now serves both. A stray TIER in the environment would be
# ignored by these lines, which is the point.

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.runner.utils import parse_telephony_websocket  # noqa: E402
from pipecat.serializers.exotel import ExotelFrameSerializer  # noqa: E402
from pipecat.transports.websocket.fastapi import (  # noqa: E402
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

import prompt  # noqa: E402
from agent import SARVAM_LANGUAGES, USER_SPEECH_TIMEOUT, build_call  # noqa: E402
from calendar_backend import InvalidPhone, normalise_phone  # noqa: E402

app = FastAPI()


def caller_id(call_data):
    """The caller's number from the phone line, or None if we can't trust it.

    Exotel sends it in the same 'start' event as the stream id — it arrives as
    something like '07338755434'; normalise_phone strips the leading zero. This
    is the number the network says the call came from, so unlike ten digits
    read out over an 8kHz line it cannot be misheard.

    Returns None rather than raising for a withheld or non-mobile caller
    (landline, international, unknown). Every path downstream treats None as
    "ask them", which is exactly what we did before caller ID existed.
    """
    raw = call_data.get("from") or getattr(call_data, "from_number", None)
    if not raw:
        logger.info("No caller ID on this call — will ask for the number")
        return None
    try:
        number = normalise_phone(raw)
    except InvalidPhone:
        logger.info(f"Caller ID {raw!r} is not an Indian mobile — will ask for the number")
        return None
    logger.info(f"Caller ID: {number}")
    return number


@app.get("/")
def health():
    """Open this in a browser to confirm the server is up before wiring Exotel."""
    try:
        answering_for = {n: Path(p).stem for n, p in prompt.phone_registry().items()}
        problem = None
    except ValueError as e:            # two profiles claiming one ExoPhone
        answering_for, problem = {}, str(e)
    return {
        "status": "error" if problem else "ok",
        "service": "exotel-voicebot",
        "answering_for": answering_for,
        "routing_problem": problem,
        "lines": {
            "/ws": "english, routed by the number dialled",
            "/ws/indian": f"sarvam ({', '.join(SARVAM_LANGUAGES)}), routed by the number dialled",
            "/ws/for/<business>": "english, this client only",
            "/ws/for/<business>/indian": "sarvam, this client only",
        },
        "clients": sorted(answering_for.values()) or None,
    }


@app.websocket("/ws")
async def english_line(websocket: WebSocket):
    """Deepgram + Cartesia. English only, and the line that has real call time."""
    await run_call(websocket, tier="english")


@app.websocket("/ws/for/{business}")
async def english_line_for(websocket: WebSocket, business: str):
    """One client's own English line — /ws/for/glow-studio.

    The 'for' segment keeps client ids out of the same namespace as /ws/indian,
    so a business could be called "indian" without shadowing a route.
    """
    await run_call(websocket, tier="english", business=business)


@app.websocket("/ws/for/{business}/indian")
async def indian_line_for(websocket: WebSocket, business: str):
    """One client's own Sarvam line — /ws/for/glow-studio/indian."""
    await run_call(websocket, tier="indian", business=business)


@app.websocket("/ws/indian")
async def indian_line(websocket: WebSocket):
    """Sarvam. Greets in English, asks which of English/Tamil/Hindi to continue in.

    A second route rather than a second server: one ExoPhone and one ngrok
    tunnel cannot reach two processes, but they can reach two paths. Point a
    second ExoPhone here later and both stacks run at once, unchanged.
    """
    await run_call(websocket, tier="indian")


def profile_for_business_id(business):
    """The profile whose business.id matches, or None.

    Used by the /ws/for/<id> routes, where the URL names the client outright.
    """
    for path in sorted(Path("profiles").glob("*.json")):
        try:
            if prompt.business_id(json.loads(path.read_text(encoding="utf-8"))) == business.lower():
                return str(path)
        except (json.JSONDecodeError, OSError):
            continue
    return None


def resolve_business(call_data, business=None):
    """Which client this call is for.

    Two mechanisms, in order of how much they can be trusted:

      1. `business` from the URL — /ws/for/glow-studio. Cannot be absent, because
         the caller reached that path. Each client's ExoPhone is pointed at its
         own URL when the account is set up.
      2. The dialled number, from Exotel's handshake. Nicer onboarding — every
         client gets the identical URL and the number does the work — but it
         relies on the provider sending a 'to' field, which is theirs to send.

    Explicit beats inferred, so the path wins when both are present.

    Three outcomes, and the difference matters:

      - resolves to a profile      -> serve that business
      - names a client we lack     -> None, and the call is refused
      - nothing to route on at all -> fall back to PROFILE, loudly

    The middle case must never fall back to a default. That is the path where a
    stranger reaches the wrong clinic's bot, hears the wrong prices, and gets
    answers out of another business's documents — the knowledge store is scoped
    by business id, and the business id comes from whichever profile answered.

    The last case exists because refusing a call is worse than serving the
    configured business when the fault is ours: if a provider stops sending the
    field, a hard refusal takes every client's phone line down at once.
    """
    if business:
        profile_path = profile_for_business_id(business)
        if not profile_path:
            logger.error(f"Refusing call — the URL names '{business}', which has no profile")
            return None
        logger.info(f"Routed by URL: {business} -> {profile_path}")
        return profile_path

    dialled = call_data.get("to") or getattr(call_data, "to_number", None)
    if not dialled:
        fallback = os.getenv("PROFILE", "profiles/clinic.json")
        logger.warning(
            f"No dialled number in the handshake — cannot route. Falling back to "
            f"{fallback}. Check the 'Raw call_data' line below for a 'to' field."
        )
        return fallback

    profile_path = prompt.profile_for_number(dialled)
    if not profile_path:
        known = ", ".join(sorted(prompt.phone_registry())) or "none"
        logger.error(
            f"Refusing call to {dialled} — no profile claims that number. "
            f"Numbers we answer for: {known}. Add an 'exophone' to the business "
            f"block of the profile that should take it."
        )
        return None

    logger.info(f"Dialled {dialled} -> {profile_path}")
    return profile_path


async def run_call(websocket: WebSocket, tier: str, business: str = None):
    """Everything the lines do identically — stack and client are the variables."""
    await websocket.accept()

    # Exotel sends a 'connected' then a 'start' event before any audio. This
    # reads them and tells us which call we are on.
    transport_type, call_data = await parse_telephony_websocket(websocket)
    stream_sid = call_data["stream_id"]
    call_sid = call_data.get("call_id")
    logger.info(
        f"Incoming {transport_type} call on the {tier.upper()} line "
        f"| stream={stream_sid} call={call_sid}"
    )

    # Everything Exotel told us about this call, verbatim. Cheap on an inbound
    # call and essential on an outbound one: when remind.py dials a patient we
    # do not yet know whether their number arrives as 'from' or as 'to', or
    # whether CustomField survives into custom_parameters. Read this line off a
    # real call rather than guessing — the same way caller ID was settled.
    try:
        logger.info(f"Raw call_data: {call_data.model_dump(by_alias=True)}")
    except Exception:  # a shape we did not expect is not worth dropping a call for
        logger.info(f"Raw call_data (unparsed): {call_data!r}")

    # The one thing a phone call knows that a browser call never can.
    caller_number = caller_id(call_data)

    # Off the loop as well: resolving a business reads every profile off disk,
    # which is nothing for three files and grows with every client onboarded.
    profile_path = await asyncio.to_thread(resolve_business, call_data, business)
    if profile_path is None:
        await websocket.close(code=1008)
        return

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Telephony carries raw PCM; a WAV header would be read as noise.
            add_wav_header=False,
            # Same 0.4s as the browser bot. Turn-taking itself is handled by
            # SpeechTimeoutUserTurnStopStrategy inside build_call().
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
            serializer=ExotelFrameSerializer(stream_sid=stream_sid, call_sid=call_sid),
        ),
    )

    task = await build_call(transport, caller_number=caller_number, tier=tier,
                            profile_path=profile_path)
    logger.info(f"Bot ready (turn timeout {USER_SPEECH_TIMEOUT}s) — greeting the caller")

    # handle_sigint=False: uvicorn owns the signal handlers, and a bot grabbing
    # them stops Ctrl-C shutting the server down.
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
    logger.info(f"Call finished | stream={stream_sid}")
