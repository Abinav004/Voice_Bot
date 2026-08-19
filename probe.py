"""Fire a business's probe questions at its grounded prompt and print the replies.

Not a test suite — no pass/fail. It exists so you can see, in seconds, whether the
bot is answering from the profile and declining everything else. Reading twenty
answers side by side catches things a live call hides.

Questions live in probes/<business>.json so adding a business means adding data,
never code — the same rule the bot itself follows.

    PROFILE=profiles/salon.json python probe.py
    PROFILE=profiles/vet.json   python probe.py known unknown
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

from prompt import DEFAULT_PROFILE, build_system_prompt, business_now, booking_enabled, load_profile

load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def probe_path(profile_path):
    """probes/salon.json sits next to profiles/salon.json."""
    return Path("probes") / Path(profile_path).name


async def main():
    profile_path = os.getenv("PROFILE", DEFAULT_PROFILE)
    profile = load_profile(profile_path)
    probes = json.loads(probe_path(profile_path).read_text(encoding="utf-8"))

    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    system = build_system_prompt(profile, business_now(profile), booking_enabled=booking_enabled(profile))
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"{profile['business']['name']}  |  {MODEL}  |  ~{len(system) // 4} prompt tokens")

    async def ask(question):
        for attempt in range(5):
            try:
                reply = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": question}],
                )
                return reply.choices[0].message.content.strip()
            except RateLimitError:
                await asyncio.sleep(5)
        return "(gave up — rate limited)"

    for bucket, questions in probes.items():
        if wanted and bucket not in wanted:
            continue
        print(f"\n{'=' * 74}\n{bucket.upper()}\n{'=' * 74}", flush=True)
        # Independent single-turn questions, so these can run together.
        answers = await asyncio.gather(*(ask(q) for q in questions))
        for question, answer in zip(questions, answers):
            print(f"Q: {question}\nA: {answer}\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
