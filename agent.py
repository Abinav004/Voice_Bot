import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.services.groq.llm import GroqLLMService
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.services.openai.tts import OpenAITTSService
import pipecat.services.openai.tts as _openai_tts_mod
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import TTSSpeakFrame

import asyncio
import importlib
from datetime import datetime, timezone

import knowledge
import prompt as prompt_module
from calendar_backend import CalComCalendar, MockCalendar
from db import save_conversation
from guards import ToolPromiseGuard
from tools import build_tools

load_dotenv(override=True)

GROQ_MODEL = "llama-3.3-70b-versatile"  # check console.groq.com/docs/models for the current best

# Whether booking is on comes from the profile — a business with no staff gets the
# read-only receptionist. See prompt.booking_enabled().

# "calcom" writes to the real calendar; "mock" is the local stand-in, useful for
# rehearsing a demo without filling a real diary with test bookings.
CALENDAR_BACKEND = os.getenv("CALENDAR_BACKEND", "calcom").lower()

# Said aloud the moment a tool call starts, so the wait isn't silence. Per tool:
# "let me check that" is right for a lookup and wrong at the moment of committing
# a booking, which is what the caller heard on the first live run.
TOOL_FILLERS = {
    "check_availability": "Let me check that for you.",
    "book_appointment": "One moment, I'm booking that now.",
    "search_knowledge": "Let me look that up for you.",
}
DEFAULT_FILLER = "One moment."

# A hung calendar should become an apology, not a dead line.
TOOL_TIMEOUT_SECS = 8.0

# Sarvam voice. v3 pronounces far better than v2; v2's voices (arya, anushka…) do
# not exist in v3, so the model and speaker have to change together. Avoid the
# speakers named priya and neha — they collide with real staff names in the
# clinic and vet profiles and get confusing on a call.
SARVAM_TTS_MODEL = "bulbul:v3"
SARVAM_TTS_VOICE = "kavya"

# Offered on the Sarvam stack only — Deepgram and Cartesia are English-only here.
# The caller picks one at the start of the call and set_language retunes the line.
SARVAM_LANGUAGES = ["English", "Tamil", "Hindi"]

# How long to wait after you go quiet before deciding your turn is over.
# At 0.6 a single sentence was being split into two or three separate turns
# ("So" / "are all the doctors available" / "what do they do?"), each firing its
# own LLM call. Lower = snappier but more false starts; higher = calmer but laggier.
USER_SPEECH_TIMEOUT = 0.8

# ---------------------------------------------------------------------------
# Together hosts Kokoro on an OpenAI-compatible /v1/audio/speech endpoint, so
# we reuse Pipecat's OpenAITTSService. But it checks the voice against OpenAI's
# own list (the module-level VALID_VOICES global) and rejects Kokoro names like
# af_heart. run_tts reads that global at call time, so we widen it here.
# ---------------------------------------------------------------------------
_openai_tts_mod.VALID_VOICES = {
    **_openai_tts_mod.VALID_VOICES,
    "af_heart": "af_heart",
    "af_bella": "af_bella",
    "af_sarah": "af_sarah",
    "af_nicole": "af_nicole",
    "am_adam": "am_adam",
    "am_michael": "am_michael",
    "bf_emma": "bf_emma",
    "bm_george": "bm_george",
}


# Cartesia's documented ranges. Out-of-range values are rejected by the API
# mid-call, which would take the bot's voice away rather than fail loudly here.
CARTESIA_LIMITS = {"speed": (0.6, 1.5), "volume": (0.5, 2.0)}


def cartesia_voice(profile):
    """How the Cartesia voice should sound, from the profile's assistant.voice.

    Delivery only — emotion, pace, loudness. What the bot actually *says* is
    shaped by assistant.tone in the system prompt, and that carries most of the
    perceived warmth: a bright voice reading a curt sentence still sounds curt.

    Env vars win over the profile so a voice can be A/B tested on a live call
    without editing JSON and restarting a thought:

        $env:CARTESIA_EMOTION="excited"; $env:CARTESIA_SPEED="1.1"

    Returns None when nothing is configured, which leaves Cartesia at its
    defaults — the behaviour before any of this existed.
    """
    voice = (profile or {}).get("assistant", {}).get("voice", {})
    emotion = os.getenv("CARTESIA_EMOTION") or voice.get("emotion")

    numbers = {}
    for field in ("speed", "volume"):
        raw = os.getenv(f"CARTESIA_{field.upper()}") or voice.get(field)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring non-numeric Cartesia {field}: {raw!r}")
            continue
        low, high = CARTESIA_LIMITS[field]
        if not low <= value <= high:
            logger.warning(f"Cartesia {field} {value} outside {low}-{high} — clamping")
            value = max(low, min(high, value))
        numbers[field] = value

    if not emotion and not numbers:
        return None
    logger.info(f"Voice: emotion={emotion or 'default'} {numbers or ''}".strip())
    return GenerationConfig(emotion=emotion, **numbers)


def resolve_tier(tier=None):
    """Which speech stack this call uses.

    An explicit argument wins over the environment, so one server process can
    run both stacks at once — exotel_server.py serves English on /ws and Sarvam
    on /ws/indian, in the same uvicorn. Reading TIER from the environment was
    fine while a process meant one stack; it cannot express two.
    """
    return (tier or os.getenv("TIER", "indian")).lower()


def build_stt_tts(profile=None, tier=None):
    """Pick STT + TTS based on the chosen tier. The LLM is chosen separately.

    The Indian stack takes its language from the profile (or LANGUAGE=ta-IN), so
    the same bot can answer in Tamil or Hindi without a code change. The English
    stack is Deepgram + Cartesia and stays English.
    """
    TIER = resolve_tier(tier)
    lang_code, lang_name = prompt_module.language(profile or {})
    if TIER == "english":
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        # Sub-choice: cheap (Kokoro) vs premium (Cartesia)
        tts_tier = os.getenv("TTS_TIER", "cheap").lower()
        if tts_tier == "premium":
            logger.info("Stack: ENGLISH / PREMIUM — Deepgram STT + Cartesia TTS")
            tts = CartesiaTTSService(
                api_key=os.getenv("CARTESIA_API_KEY"),
                # settings=, not voice_id=/params=: both of those are deprecated
                # in 1.7.0 and warn on every start.
                settings=CartesiaTTSService.Settings(
                    voice=os.getenv("CARTESIA_VOICE_ID",
                                    "71a7ad14-091c-4e8e-a314-022ece01c121"),
                    # sonic-3.5 is the service default and is what reads
                    # generation_config; older Sonic models ignore it silently.
                    model="sonic-3.5",
                    generation_config=cartesia_voice(profile),
                ),
            )
        else:
            logger.info("Stack: ENGLISH / CHEAP — Deepgram STT + Kokoro TTS (Together hosted)")
            tts = OpenAITTSService(
                api_key=os.getenv("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
                model="hexgrad/Kokoro-82M",
                voice="af_heart",
                sample_rate=24000,
            )
    else:
        logger.info(f"Stack: INDIAN — Sarvam STT + Sarvam TTS ({lang_name}, {lang_code})")
        stt = SarvamSTTService(
            api_key=os.getenv("SARVAM_API_KEY"),
            # Pinned to one language rather than auto-detected: left to itself
            # saaras:v3 returned Odia and Hindi for clear English speech, which makes
            # a wrong answer impossible to attribute between mishearing and
            # hallucination. Which language comes from the profile.
            settings=SarvamSTTService.Settings(model="saaras:v3", language=Language(lang_code)),
        )
        tts = SarvamTTSService(
            api_key=os.getenv("SARVAM_API_KEY"),
            settings=SarvamTTSService.Settings(
                model=SARVAM_TTS_MODEL, voice=SARVAM_TTS_VOICE, language=Language(lang_code)),
        )
    return stt, tts


def build_llm():
    """Pick the LLM based on the chosen provider. Returns (llm, provider, model)."""
    # OpenAI is the default while the Groq key is on the free tier — 12k tokens/min
    # can't carry booking turns (~3k each). Groq is faster (0.17s vs 0.72s TTFB) and
    # should come back as the default once the key is paid.
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "gemini":
        from pipecat.services.google.llm import GoogleLLMService
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        logger.info(f"LLM: GEMINI ({model})")
        return (
            GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model=model),
            provider,
            model,
        )
    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info(f"LLM: OPENAI ({model})")
        return (
            OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model=model),
            provider,
            model,
        )
    # default: Groq
    model = GROQ_MODEL
    logger.info(f"LLM: GROQ ({model})")
    return (
        GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model=model),
        provider,
        model,
    )


async def build_call(transport, caller_number=None, tier=None, profile_path=None):
    """Everything between the transport and the runner — the bot itself.

    Takes a transport and returns a ready PipelineTask. Shared by the browser
    entry point below and by exotel_server.py, so a phone call gets exactly the
    same profile, tools, guard and logging as a browser call. Two copies of this
    would drift the first time a prompt rule changed in one and not the other.

    `caller_number` is the caller's own 10-digit number when the phone line told
    us who is ringing. It saves the two-to-four turns spent capturing digits by
    voice, which is the most error-prone stretch of the booking flow. The browser
    has no such thing and passes None, which keeps the ask-for-digits flow.

    `tier` picks the speech stack for this one call — "english" (Deepgram +
    Cartesia) or "indian" (Sarvam, which can also do Tamil and Hindi). None
    falls back to the TIER environment variable, which is what the browser
    entry point and the console prompt still use.

    `profile_path` is which business this call is for. The phone front door
    resolves it from the number that was dialled, so one server answers for
    every client. None falls back to the PROFILE environment variable — the
    browser bot and the console have no dialled number to work from.
    """
    tier = resolve_tier(tier)
    # Profile first: the speech services need its language, and reloading the
    # module picks up prompt.py edits without restarting the server.
    importlib.reload(prompt_module)
    # Explicit path wins; PROFILE is the fallback for callers with no dialled
    # number to route on (the browser bot, the console).
    profile_path = profile_path or os.getenv("PROFILE", prompt_module.DEFAULT_PROFILE)
    profile = prompt_module.load_profile(profile_path)

    stt, tts = build_stt_tts(profile, tier)
    llm, provider, model = build_llm()

    if provider in ("groq", "openai"):
        try:
            await llm._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            logger.info(f"{provider.upper()} connection warmed up")
        except Exception as e:
            logger.warning(f"{provider} warmup failed (non-fatal): {e}")
    else:
        logger.info("Skipping warmup (Gemini warms on its first turn)")

    # The date is resolved per call — computing it at import would serve a stale
    # weekday on a long-running server, and staff schedules are day-of-week based.
    booking_on = prompt_module.booking_enabled(profile)
    # Only the Sarvam stack can hear and speak Indic languages, so only it offers
    # the caller a choice. Deepgram + Cartesia stay English.
    offer_languages = SARVAM_LANGUAGES if tier != "english" else None
    # Does this business have any documents? Asked once per call rather than
    # per turn, and never allowed to fail a call — a knowledge store that is
    # down simply means the bot behaves as it did before there was one.
    business = prompt_module.business_id(profile)
    try:
        # to_thread, not a direct call: this opens a Postgres connection, which
        # measured ~700ms. On the event loop that is 700ms of frozen audio for
        # every call already in progress, not just this one. Invisible with a
        # single caller and the first thing to bite under concurrency.
        knowledge_on = bool(await asyncio.to_thread(knowledge.sources, business))
    except Exception as e:
        logger.warning(f"knowledge store unreachable, continuing without it: {e}")
        knowledge_on = False

    system_prompt = prompt_module.build_system_prompt(
        profile, prompt_module.business_now(profile),
        booking_enabled=booking_on, offer_languages=offer_languages,
        caller_number=caller_number, knowledge_available=knowledge_on,
    )
    logger.info(
        f"Loaded profile: {profile['business']['name']} "
        f"(~{len(system_prompt) // 4} prompt tokens, booking={'on' if booking_on else 'off'}"
        + (f", languages={'/'.join(offer_languages)}" if offer_languages else "")
        + (f", caller ID {caller_number}" if caller_number else ", no caller ID")
        + (", knowledge ON" if knowledge_on else "") + ")"
    )

    tools_schema = None
    if booking_on:
        if CALENDAR_BACKEND == "mock":
            calendar = MockCalendar(profile)
            logger.info("Calendar: MOCK (bookings go to bookings.json)")
        else:
            calendar = CalComCalendar(profile)
            logger.info("Calendar: CAL.COM (bookings are real)")
        tools_schema, handlers = build_tools(
            calendar, offer_languages=offer_languages, caller_number=caller_number,
            business_id=business if knowledge_on else None)
        for tool_name, handler in handlers.items():
            # timeout_secs so a hung calendar surfaces as an apology rather than
            # the caller listening to silence.
            llm.register_function(tool_name, handler, timeout_secs=TOOL_TIMEOUT_SECS)

        @llm.event_handler("on_function_calls_started")
        async def on_function_calls_started(service, function_calls):
            # The call takes a beat; say something so the line isn't dead.
            name = function_calls[0].function_name if function_calls else ""
            await service.push_frame(TTSSpeakFrame(TOOL_FILLERS.get(name, DEFAULT_FILLER)))

    messages = [{"role": "system", "content": system_prompt}]
    context = LLMContext(messages, tools=tools_schema) if tools_schema else LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=USER_SPEECH_TIMEOUT
                    )
                ],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            # Sits right after the LLM so it sees the response and any tool calls
            # together, and can force the call the model only promised.
            *([ToolPromiseGuard(knowledge_available=knowledge_on)] if booking_on else []),
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    started_at = datetime.now(timezone.utc)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # No extra system message here — the greeting instruction lives in the
        # system prompt. Appending one used to leave it in context permanently,
        # competing with the persona on every later turn.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        # Keep the transcript. Off the event loop's critical path and wrapped in
        # its own error handling inside save_conversation — a call that already
        # happened must never fail, or hold up teardown, because a database was
        # slow or down.
        await asyncio.to_thread(
            save_conversation,
            context.get_messages(),
            business=profile.get("business", {}).get("name"),
            # The profile that actually answered, not the environment's idea of
            # it — with routing those differ, and the log has to be evidence.
            profile_path=profile_path,
            # The resolved tier, not the environment: with two routes in one
            # process the environment no longer says which stack took the call.
            stack=tier,
            llm_model=model,
            started_at=started_at,
        )
        await task.cancel()

    return task


async def bot(runner_args: RunnerArguments):
    """Browser entry point — WebRTC in, same bot behind it."""
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
            ),
        },
    )
    task = await build_call(transport)
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    # 1) Which stack?
    stack = input("Select stack — [I]ndian (Sarvam) or [E]nglish (Deepgram): ").strip().lower()
    os.environ["TIER"] = "english" if stack.startswith("e") else "indian"

    # 1b) English only: cheap (Kokoro) or premium (Cartesia)?
    if os.environ["TIER"] == "english":
        tier_choice = input("English TTS — [C]heap (Kokoro) or [P]remium (Cartesia): ").strip().lower()
        os.environ["TTS_TIER"] = "premium" if tier_choice.startswith("p") else "cheap"

    # 2) Which LLM?  Enter = OpenAI, the default while the Groq key is free-tier.
    # Booking needs tool calls of ~3k tokens and Groq's 12k/min ceiling can't carry
    # them; pressing Enter used to land on Groq and stall mid-booking.
    llm_choice = input("Select LLM — [O]penAI (default)  [Q]Groq  [G]emini: ").strip().lower()
    os.environ["LLM_PROVIDER"] = (
        "gemini" if llm_choice.startswith("g")
        else "groq" if llm_choice.startswith("q")
        else "openai"
    )

    tier_note = f" / {os.environ['TTS_TIER'].upper()}" if os.environ["TIER"] == "english" else ""
    print(f"→ Starting {os.environ['TIER'].upper()}{tier_note} stack with {os.environ['LLM_PROVIDER'].upper()} LLM\n")

    from pipecat.runner.run import main
    main()