from __future__ import annotations

import logging
import os
import math

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobExecutorType,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, silero, smallestai

from api import AssistantFnc
from prompts import INSTRUCTIONS, WELCOME_MESSAGE

load_dotenv(override=True)

logger = logging.getLogger("ria-agent")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info("Connecting to room: %s", ctx.room.name)
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    logger.info("Participant joined: %s", participant.identity)

    assistant_fnc = AssistantFnc()

    # Smallest AI STT (Pulse Multi-Language), Smallest AI TTS (Lightning Pro), Silero VAD, OpenAI LLM
    stt_lang = os.getenv("SMALLEST_STT_LANGUAGE", "multi")
    tts_voice = os.getenv("SMALLEST_TTS_VOICE_ID", "meher")
    tts_model = os.getenv("SMALLEST_TTS_MODEL", "lightning_v3.1_pro")
    llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    session = AgentSession(
        stt=smallestai.STT(language=stt_lang),
        llm=openai.LLM(
            model=llm_model,
            temperature=0.6,
        ),
        tts=smallestai.TTS(
            voice_id=tts_voice,
            model=tts_model,
        ),
        vad=silero.VAD.load(
            min_speech_duration=0.35,      # Ignores clicks, breaths and noise < 350ms
            min_silence_duration=0.55,     # Smooth turn closing
            prefix_padding_duration=0.3,   # Clean voice onset
            activation_threshold=0.75,     # High threshold blocks room noise & static
        ),
    )

    agent = Agent(
        instructions=INSTRUCTIONS,
        tools=[
            assistant_fnc.get_plan_and_trial_info,
            assistant_fnc.get_product_features,
            assistant_fnc.record_demo_interest,
        ],
    )

    await session.start(
        room=ctx.room,
        agent=agent,
    )

    # Immediately speak the full self-introduction without LLM skipping
    greeting_text = (
        "హాయ్, నేను RIA! ఈరోజు మీకు RIA Collection Agent గురించి quick demo ఇవ్వబోతున్నాను. "
        "ఒక నిమిషం... ముందు మీ business గురించి కొంచెం తెలుసుకుందాం. "
        "సార్, మీ business network లో approximately ఎంత మంది customers ఉన్నారు?"
    )
    await session.say(greeting_text)

    logger.info("RIA voice session started for %s", participant.identity)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            job_executor_type=JobExecutorType.THREAD,
            num_idle_processes=0,
            load_threshold=math.inf,
            load_fnc=lambda *args: 0.0,
        )
    )