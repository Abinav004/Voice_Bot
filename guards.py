"""Catch the bot promising something it never did.

Three separate prompt rules failed to stop this, including one that spelled out
the exact failure. On a live call it produced:

    "Perfect! Your appointment is booked with Dr. Priya Nair on Tuesday at one
     in the afternoon. Thank you, Jonathan, and have a great day!"

with no tool call behind it. Nothing was written to the calendar. A caller hangs
up there believing they have an appointment, which is worse than any error we
could have shown them.

So this stops being a prompt problem. The guard watches each LLM response: if it
promises an action and no tool ran, it appends a correction to the context and
re-runs the model, which then makes the call for real.

The promise has usually already been spoken by the time we notice — text streams
straight to TTS, and buffering it would add latency to every turn. So the caller
still hears "booked" a moment early, and the booking lands about a second later.
Retroactively true beats silently false.
"""

import re

from loguru import logger
from pipecat.frames.frames import (
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    TextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

# Things the bot only gets to say if a tool call is going out in the same breath.
PROMISE = re.compile(
    r"\b("
    # "it is done" in any of its forms. Written as a general shape rather than a
    # list of exact phrases because the list kept missing near-misses on live
    # calls: "is booked" did not catch "is NOW booked", and "has been cancelled"
    # did not catch "has been SUCCESSFULLY CANCELED" — different adverb, American
    # spelling. Both slipped through and left the caller hanging.
    r"(?:is|are|has been|have been|was|were)\s+"
    r"(?:(?:now|already|successfully|all|just)\s+){0,2}"
    r"(?:booked|confirmed|cancelled|canceled|moved|rescheduled)"
    # "I am about to do it" — i(?:'m| am), not i (?:am|'m); the latter needs a
    # space before the apostrophe, so it never matched the contraction at all.
    r"|i'?ve (?:booked|cancelled|canceled|moved|rescheduled)"
    r"|i'?ll (?:go ahead|now)\b|i will now (?:book|cancel|move|reschedule)"
    r"|i(?:'m| am) (?:booking|cancelling|canceling|moving|rescheduling)"
    r"|(?:booked|cancelled|canceled|moved) (?:it|that|you|your)"
    r"|booking (?:it|that) now"
    # "wait a second" with nothing on the way is the same failure, minus the lie.
    r"|let me check|let me look|one moment|hold on|please hold|bear with me"
    r"|i'?ll check"
    r")\b",
    re.IGNORECASE,
)

# "I don't know" — said about the bot's own knowledge, with nothing looked up.
#
# Deliberately first person. "We do not offer dental services" is a fact from
# the profile and must never trigger a search; "I don't have that information"
# is an admission that only becomes true after looking. On the probe run the
# model said "I don't have information about prescription refills" and offered
# to take a message, when the refill policy was sitting in the store — the one
# failure RAG is supposed to remove.
UNCHECKED_DECLINE = re.compile(
    r"\b("
    r"i (?:do not|don'?t) have"
    r"|i (?:do not|don'?t) know"
    r"|i(?:'m| am) not sure"
    r"|i (?:can'?t|cannot) (?:find|provide) (?:that|this|any|information|details)"
    r")\b",
    re.IGNORECASE,
)

SEARCH_NUDGE = (
    "SYSTEM CORRECTION: you told the caller you do not have that information, but "
    "you never searched for it. You cannot know what is on file until you look. "
    "Call search_knowledge now with the caller's question. Do not apologise and do "
    "not repeat yourself."
)

NUDGE = (
    "SYSTEM CORRECTION: you just told the caller you were doing something, but you "
    "did not call any tool, so nothing actually happened. Do not apologise and do "
    "not repeat yourself to the caller. Call the correct tool right now with the "
    "details you already have."
)

MAX_NUDGES_PER_TURN = 2  # a backstop against nudging in circles


class ToolPromiseGuard(FrameProcessor):
    """Re-runs the LLM when it says it will act but calls no tool."""

    def __init__(self, knowledge_available=False, **kwargs):
        super().__init__(**kwargs)
        # Only chase an unchecked "I don't know" when there is somewhere to
        # look. Without a knowledge store that reply is simply the truth.
        self._knowledge_available = knowledge_available
        self._reset_turn()

    def _reset_turn(self):
        self._tool_ran_this_turn = False
        self._nudges = 0
        self._reset_response()

    def _reset_response(self):
        self._text = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._reset_turn()
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._reset_response()
        elif isinstance(frame, TextFrame):
            self._text += frame.text
        elif isinstance(frame, (FunctionCallsStartedFrame, FunctionCallResultFrame)):
            # A tool ran somewhere in this turn, so a later "it's booked" is honest.
            self._tool_ran_this_turn = True

        await self.push_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._check_for_empty_promise()

    async def _check_for_empty_promise(self):
        said = self._text.strip()
        self._reset_response()

        if self._tool_ran_this_turn or not said:
            return

        match = PROMISE.search(said)
        nudge, why = (NUDGE, "Empty promise") if match else (None, None)

        # Second case: it declined for lack of knowledge without ever looking.
        # Checked only when PROMISE did not fire, so one reply cannot earn two
        # corrections. Deliberately first person — "we do not offer dental
        # services" is a profile fact and must never send the bot searching.
        if not match and self._knowledge_available:
            match = UNCHECKED_DECLINE.search(said)
            if match:
                nudge, why = SEARCH_NUDGE, "Declined without searching"

        if not match:
            return
        if self._nudges >= MAX_NUDGES_PER_TURN:
            logger.warning(f"{why} again ({said[:60]!r}) — already nudged, leaving it")
            return

        self._nudges += 1
        logger.warning(f"{why}: said {match.group(0)!r} with no tool call — forcing the call")
        # Downstream: the assistant aggregator handles this and re-runs the LLM.
        await self.push_frame(
            LLMMessagesAppendFrame(messages=[{"role": "system", "content": nudge}], run_llm=True)
        )
