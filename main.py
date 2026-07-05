import asyncio
import logging
import os
from collections.abc import AsyncIterator

import sounddevice as sd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from voxkit.core.pipeline import VoxkitPipeline
from voxkit.stt import SarvamOptions, SarvamSTTProvider

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

options = SarvamOptions(
    api_key=os.getenv("SARVAM_API_KEY"),
    model="saaras:v3",
    mode="transcribe",
    language_code="en-IN",
    high_vad_sensitivity=True,
    vad_signals=True,
    input_audio_codec="pcm_s16le",
    sample_rate=SAMPLE_RATE,
)


async def microphone_stream() -> AsyncIterator[bytes]:
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata, _frames, _time, status):
        if status:
            logger.warning("Microphone: %s", status)
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=callback,
    )

    with stream:
        while True:
            yield await queue.get()


async def log_llm_output(queue: asyncio.Queue) -> None:
    while True:
        sentence = await queue.get()
        if sentence is None:
            logger.info("Turn complete")
        else:
            logger.info("LLM: %s", sentence)


async def main() -> None:
    stt = SarvamSTTProvider(options, asyncio.Queue())
    agent = create_agent(
        model=ChatGroq(model="llama-3.3-70b-versatile"),
        tools=[],
    )
    pipeline = VoxkitPipeline(stt, agent)

    logger.info("Speak into your microphone (Ctrl+C to stop)...")

    pipeline_task = asyncio.create_task(pipeline.run(microphone_stream()))
    output_task = asyncio.create_task(log_llm_output(pipeline.llm_output_queue))

    try:
        await pipeline_task
    finally:
        output_task.cancel()
        await asyncio.gather(output_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
