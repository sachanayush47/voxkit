import asyncio
import base64
import logging
import os
from collections.abc import AsyncIterator

import sounddevice as sd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from voxkit.core.pipeline import PipelineConfig, VoxkitPipeline
from voxkit.stt import SarvamSTTOptions, SarvamSTTProvider
from voxkit.tts import SarvamTTSOptions, SarvamTTSProvider, TTSEvent, TTSEventType
from voxkit.turn import PipecatSmartTurnDetector

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logging.getLogger("voxkit").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)

options = SarvamSTTOptions(
    api_key=os.getenv("SARVAM_API_KEY"),
    model="saaras:v3",
    mode="transcribe",
    language_code="en-IN",
    high_vad_sensitivity=True,
    vad_signals=True,
    input_audio_codec="pcm_s16le",
    sample_rate=SAMPLE_RATE,
)

tts_options = SarvamTTSOptions(
    api_key=os.getenv("SARVAM_API_KEY"),
    model="bulbul:v3",
    target_language_code="en-IN",
    speaker="priya",
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


async def main() -> None:
    stt = SarvamSTTProvider(options)
    tts = SarvamTTSProvider(tts_options)
    agent = create_agent(
        system_prompt=(
            "You are in a realtime voice conversation with a human. Your replies are spoken aloud, "
            "so answer in a few short, plain sentences. Never use markdown, lists, tables or emojis."
        ),
        model=ChatGroq(model="openai/gpt-oss-120b"),
        tools=[],
    )

    playback = sd.RawOutputStream(
        samplerate=tts_options.speech_sample_rate,
        channels=CHANNELS,
        dtype="int16",
    )
    playback.start()

    async def handle_tts_event(event: TTSEvent) -> None:
        if event.type == TTSEventType.AUDIO and event.audio:
            await asyncio.to_thread(playback.write, base64.b64decode(event.audio))
        elif event.type == TTSEventType.INTERRUPT:
            await asyncio.to_thread(playback.abort)
            await asyncio.to_thread(playback.start)

    pipeline = VoxkitPipeline(
        stt,
        tts,
        agent,
        handle_tts_event,
        config=PipelineConfig(interrupt=True, resume_window=2.0),
        end_of_turn=PipecatSmartTurnDetector(sample_rate=SAMPLE_RATE),
    )

    logger.info("Speak into your microphone (Ctrl+C to stop)...")

    try:
        await pipeline.run(microphone_stream())
    finally:
        playback.stop()
        playback.close()


if __name__ == "__main__":
    asyncio.run(main())
