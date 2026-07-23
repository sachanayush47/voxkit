# voxkit

A thin, event-driven Python library for building real-time voice agents on top of **LangChain / LangGraph**.

voxkit wires together a streaming speech-to-text (STT) provider, any LangGraph agent, and a streaming text-to-speech (TTS) provider into one turn-taking pipeline — handling audio-in/audio-out plumbing and barge-in (interrupt) so you don't have to. If you already know LangChain/LangGraph, you already know how to build the "brain" of a voxkit voice agent.

**Design goals:**

- **Thin.** No frame-based transport bus, no call/room management, no bundled VAD — STT/TTS providers report voice activity themselves.
- **LangChain/LangGraph-first.** Bring any compiled `StateGraph` (e.g. from `langchain.agents.create_agent`) as the agent. voxkit streams its tokens and feeds it transcripts; it doesn't wrap or replace it.
- **Event-driven, not callback-soup.** STT, the agent, and TTS all communicate over typed events (`STTEvent`, `LLMEvent`, `TTSEvent`) on `asyncio.Queue`s, so the control flow (turns, interrupts, stream-closed) is explicit and inspectable.

!!! warning "Status"
    Early / pre-alpha (`0.0.1`). The API surface is small and will change. Currently ships one provider pair (Sarvam AI for STT and TTS); the provider interfaces are designed so more can be added without touching the pipeline.

## Installation

```bash
pip install voxkit
```

Requires Python 3.13+.

## Quick start

```python
import asyncio
import base64
import os

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from voxkit import VoxkitPipeline
from voxkit.stt import SarvamSTTOptions, SarvamSTTProvider
from voxkit.tts import SarvamTTSOptions, SarvamTTSProvider, TTSEvent, TTSEventType

stt = SarvamSTTProvider(SarvamSTTOptions(
    api_key=os.environ["SARVAM_API_KEY"],
    model="saaras:v3",
    mode="transcribe",
    language_code="en-IN",
    sample_rate=16000,
))

tts = SarvamTTSProvider(SarvamTTSOptions(
    api_key=os.environ["SARVAM_API_KEY"],
    model="bulbul:v3",
    target_language_code="en-IN",
    speaker="priya",
))

agent = create_agent(model=ChatGroq(model="llama-3.3-70b-versatile"), tools=[])


async def handle_tts_event(event: TTSEvent) -> None:
    if event.type == TTSEventType.AUDIO and event.audio:
        play_audio(base64.b64decode(event.audio))       # your playback code
    elif event.type == TTSEventType.INTERRUPT:
        stop_and_clear_playback()                         # your barge-in handling


async def main() -> None:
    pipeline = VoxkitPipeline(stt, tts, agent, handle_tts_event)
    await pipeline.run(microphone_audio_stream())          # your mic capture code


asyncio.run(main())
```

See [`main.py`](https://github.com/sachanayush47/voxkit/blob/master/main.py) in the repo for a complete, runnable example using `sounddevice` for microphone capture and playback.

Continue to [Architecture](architecture.md) for how the pieces fit together, or jump straight to the [API Reference](reference/pipeline.md).
