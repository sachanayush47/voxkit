# voxkit

A thin, event-driven Python library for building real-time voice agents on top of **LangChain / LangGraph**.

voxkit wires together a streaming speech-to-text (STT) provider, any LangGraph agent, and a streaming text-to-speech (TTS) provider into one turn-taking pipeline — handling audio-in/audio-out plumbing and barge-in (interrupt) so you don't have to. If you already know LangChain/LangGraph, you already know how to build the "brain" of a voxkit voice agent.

**Design goals:**
- **Thin.** No frame-based transport bus, no call/room management, no bundled VAD — STT/TTS providers report voice activity themselves.
- **LangChain/LangGraph-first.** Bring any compiled `StateGraph` (e.g. from `langchain.agents.create_agent`) as the agent. voxkit streams its tokens and feeds it transcripts; it doesn't wrap or replace it.
- **Event-driven, not callback-soup.** STT, the agent, and TTS all communicate over typed events (`STTEvent`, `LLMEvent`, `TTSEvent`) on `asyncio.Queue`s, so the control flow (turns, interrupts, stream-closed) is explicit and inspectable.

> **Status:** early / pre-alpha (`0.0.1`). The API surface is small and will change. Currently ships one provider pair (Sarvam AI for STT and TTS); the provider interfaces are designed so more can be added without touching the pipeline.

## Installation

```bash
pip install voxkit
```

Requires Python 3.13+. Provider SDKs (currently `sarvamai`) and `langchain`/`langgraph` are installed as direct dependencies for now — see [`pyproject.toml`](pyproject.toml).

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

See [`main.py`](main.py) for a complete, runnable example that captures microphone audio with `sounddevice` and plays synthesized speech back through your speakers.

## How it works

```
audio in ──▶ STTProvider ──▶ VoxkitPipeline ──▶ LangGraph agent
                                    │                  │
                                    │  sentence-by-     │ streamed
                                    │  sentence         ▼ tokens
                                    └──────────▶ TTSProvider ──▶ audio out (your callback)
```

1. **You feed raw audio** into `pipeline.run(audio_stream)`. It's forwarded to the STT provider.
2. **STT emits `STTEvent`s** — `SPEECH_START`/`SPEECH_END` (voice activity), `PARTIAL_TRANSCRIPT`, `FINAL_TRANSCRIPT`, `STREAM_CLOSED`.
3. **On `FINAL_TRANSCRIPT`**, the pipeline starts a new agent turn: it streams tokens from your LangGraph agent (`agent.astream(..., stream_mode="messages")`), buffers them, and forwards each complete sentence to TTS as soon as a sentence/clause boundary is detected — so speech synthesis starts well before the agent has finished generating the full reply.
4. **TTS emits `TTSEvent`s** — `AUDIO` (a synthesized chunk), `END_OF_TURN`, `INTERRUPT`, `STREAM_CLOSED` — which the pipeline forwards verbatim to your `callback`. You decide what to do with each: play `AUDIO`, stop playback on `INTERRUPT`, mark the turn done on `END_OF_TURN`.
5. **Barge-in:** if the STT provider reports `SPEECH_START` while the agent is still generating or TTS is still speaking, the pipeline cancels the in-flight turn, tells TTS to interrupt, and notifies your callback — all before the next turn starts. Pass `interrupt=False` to `VoxkitPipeline` to disable this and let turns run to completion regardless of new speech.

### Event types

| Module | Type | Values |
|---|---|---|
| `voxkit.stt` | `STTEventType` | `SPEECH_START`, `SPEECH_END`, `PARTIAL_TRANSCRIPT`, `FINAL_TRANSCRIPT`, `STREAM_CLOSED` |
| `voxkit.llm` | `LLMEventType` | `SENTENCE`, `END_OF_TURN`, `INTERRUPT` |
| `voxkit.tts` | `TTSEventType` | `AUDIO`, `END_OF_TURN`, `INTERRUPT`, `STREAM_CLOSED` |

## Public API

```python
from voxkit import VoxkitPipeline

from voxkit.stt import STTProvider, STTOptions, STTEvent, STTEventType
from voxkit.stt import SarvamSTTProvider, SarvamSTTOptions

from voxkit.tts import TTSProvider, TTSOptions, TTSEvent, TTSEventType
from voxkit.tts import SarvamTTSProvider, SarvamTTSOptions

from voxkit.llm import LLMEvent, LLMEventType
```

- **`VoxkitPipeline(stt, tts, agent, callback, thread_id="default", interrupt=True)`** — the orchestrator. `agent` is any compiled LangGraph graph; `callback` is an `async def(event: TTSEvent) -> None` that receives every TTS event. `thread_id` is passed to the agent's config on every turn so LangGraph-checkpointed memory persists across turns.
- **`STTProvider` / `TTSProvider`** — abstract base classes a new provider implements to plug into the pipeline. See their docstrings (or the [API reference](#documentation) below) for the exact contract.
- **`SarvamSTTProvider` / `SarvamTTSProvider`** — the bundled provider implementations, backed by [Sarvam AI](https://www.sarvam.ai/)'s streaming STT/TTS websockets.

## Adding a new provider

Implement `STTProvider` or `TTSProvider` (`voxkit/stt/base.py`, `voxkit/tts/base.py`) — both are small interfaces (`connect`, `send`/`receive` for STT, `connect`/`synthesize` for TTS, plus `close`) that push/pull typed events through `asyncio.Queue`s. Nothing else in the pipeline needs to change; `VoxkitPipeline` only depends on these interfaces, not on Sarvam specifically.

## Documentation

Full API reference (generated from the docstrings in this repo) is published at **[sachanayush47.github.io/voxkit](https://sachanayush47.github.io/voxkit/)**.

To build the docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## License

TBD.
