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

## Tuning the Sarvam providers

The quick start above passes only the required options. Both option classes mirror Sarvam's full set of streaming websocket parameters — voice and audio format, prosody, and the VAD thresholds that decide when a turn (and a barge-in) starts. **Every default is the `sarvamai` SDK's own default**, so the quick start behaves exactly as the SDK does out of the box, and you only pass what you actually want to change:

```python
stt = SarvamSTTProvider(SarvamSTTOptions(
    api_key=os.environ["SARVAM_API_KEY"],
    model="saaras:v3",
    mode="codemix",                        # English words in English, Indic in native script
    language_code="hi-IN",
    sample_rate=8000,                      # telephony audio
    input_audio_codec="pcm_s16le",
    high_vad_sensitivity=True,
    interrupt_min_speech_frames=6,         # make barge-in less twitchy
    start_speech_volume_threshold=-45.0,   # ignore background noise
    pre_speech_pad_frames=3,               # don't clip the start of the utterance
))

tts = SarvamTTSProvider(SarvamTTSOptions(
    api_key=os.environ["SARVAM_API_KEY"],
    model="bulbul:v3",
    target_language_code="hi-IN",
    speaker="priya",
    speech_sample_rate=24000,
    output_audio_codec="linear16",
    pace=1.1,
    temperature=0.4,                       # bulbul:v3 only
    min_buffer_size=30,                    # flush sooner -> lower first-audio latency
))
```

The STT VAD knobs are the one group with no SDK-side default — those numbers live server-side, so they default to `None` and aren't sent at all unless you set them, leaving Sarvam's default or the `high_vad_sensitivity` preset in charge.

See [`SarvamSTTOptions`][voxkit.stt.sarvam.SarvamSTTOptions] and [`SarvamTTSOptions`][voxkit.tts.sarvam.SarvamTTSOptions] for every field, its accepted values, and which ones are model-specific (`pitch`/`loudness` are `bulbul:v2`-only; `temperature`/`dict_id` are `bulbul:v3`-only — Sarvam ignores the ones that don't apply).

Continue to [Architecture](architecture.md) for how the pieces fit together, or jump straight to the [API Reference](reference/pipeline.md).
