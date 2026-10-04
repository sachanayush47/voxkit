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

Optional features are installed as extras:

| Extra | Installs | Enables |
|---|---|---|
| `smart-turn` | `onnxruntime`, `transformers`, `huggingface-hub`, `numpy` | [End-of-turn check](#end-of-turn-check) (`PipecatSmartTurnDetector`) |
| `all` | every extra above | All optional features |

```bash
pip install "voxkit[all]"
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

### Sarvam provider options

Both option classes are pydantic models that mirror Sarvam's streaming websocket parameters. Only `api_key`, `model`, `mode` (STT) / `target_language_code`, `speaker` (TTS) are required; **every default below is the `sarvamai` SDK's own default**, so passing nothing extra behaves exactly like the SDK does out of the box. The two exceptions are TTS `output_audio_codec` and `speech_sample_rate`, which default to raw 24 kHz PCM (what voxkit's playback path wants) instead of the SDK's 22.05 kHz MP3.

The SDK has no client-side values for the STT VAD knobs — those numbers live server-side — so they default to `None` and are simply not sent, leaving Sarvam's default or the `high_vad_sensitivity` preset in charge. Set one to override it.

`SarvamSTTOptions` (`voxkit/stt/sarvam.py`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `api_key` | `str` | — | Sarvam API subscription key. |
| `model` | `"saaras:v3"` \| `"saarika:v2.5"` | — | `saarika:v2.5` is legacy. |
| `mode` | `"transcribe"` \| `"translate"` \| `"verbatim"` \| `"translit"` \| `"codemix"` | — | Transcript style. `saaras:v3` only. |
| `language_code` | BCP-47 code, `"unknown"`, or `None` | `"unknown"` | `"unknown"`/`None` auto-detects and reports the language back. Twelve extra codes are `saaras:v3`-only. |
| `encoding` | `str` | `"audio/wav"` | MIME-style encoding of the chunks you push in. |
| `input_audio_codec` | `"wav"` \| `"pcm_s16le"` \| `"pcm_l16"` \| `"pcm_raw"` | `"pcm_s16le"` | Raw sample codec. |
| `sample_rate` | `int` | `16000` | Only `8000` and `16000` are valid connection-level values; `8000` is available *only* this way. |
| `vad_signals` | `bool` | `True` | Emit `START_SPEECH`/`END_SPEECH` → `SPEECH_START`/`SPEECH_END`. Defaults on because the pipeline needs it for barge-in. |
| `flush_signal` | `bool \| None` | `None` | Allow forcing finalization via `SarvamSTTProvider.flush()`. |
| `high_vad_sensitivity` | `bool` | `True` | High-sensitivity VAD preset; the thresholds below override it. |
| `positive_speech_threshold` | `float \| None` (0-1) | `None` | Probability above which a frame is speech. |
| `negative_speech_threshold` | `float \| None` (0-1) | `None` | Probability below which a frame is silence. |
| `min_speech_frames` | `int \| None` | `None` | Frames needed to open a speech segment. |
| `first_turn_min_speech_frames` | `int \| None` | `None` | Same, for the first user turn only. |
| `negative_frames_count` | `int \| None` | `None` | Silence frames needed to close a segment. |
| `negative_frames_window` | `int \| None` | `None` | Window the count above is measured over. |
| `start_speech_volume_threshold` | `float \| None` | `None` | dB gate below which audio isn't speech. Unset = no gate. |
| `interrupt_min_speech_frames` | `int \| None` | `None` | Frames needed to register a barge-in — tune this when interrupts fire too eagerly or too reluctantly. |
| `pre_speech_pad_frames` | `int \| None` | `None` | Frames prepended before speech onset so the utterance isn't clipped. |
| `num_initial_ignored_frames` | `int \| None` | `None` | Leading frames discarded at connection start. |

`SarvamTTSOptions` (`voxkit/tts/sarvam.py`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `api_key` | `str` | — | Sarvam API subscription key. |
| `model` | `"bulbul:v2"` \| `"bulbul:v3"` | — | See the model-specific notes below. |
| `target_language_code` | BCP-47 code | — | One of eleven languages Sarvam synthesizes. |
| `speaker` | voice name | — | Must belong to the chosen model (`priya`, `aditya`, … for v3; `anushka`, `abhilash`, … for v2). |
| `send_completion_event` | `bool` | `True` | Sarvam sends a `final` event → `TTSEventType.END_OF_TURN`. |
| `output_audio_codec` | `"linear16"` \| `"mulaw"` \| `"alaw"` \| `"opus"` \| `"flac"` \| `"aac"` \| `"wav"` \| `"mp3"` | `"linear16"` | `linear16` is raw PCM, which most playback paths want. |
| `output_audio_bitrate` | `"32k"`…`"192k"` | `"128k"` | Only meaningful for compressed codecs. |
| `speech_sample_rate` | `8000` \| `16000` \| `22050` \| `24000` | `24000` | Sarvam defaults to 22050 on v2, 24000 on v3. |
| `pace` | `float` | `1.0` | Speech speed. 0.3-3.0 on v2, 0.5-2.0 on v3. |
| `pitch` | `float` | `0.0` | -0.75 to 0.75. **v2 only.** |
| `loudness` | `float` | `1.0` | 0.3 to 3.0. **v2 only.** |
| `temperature` | `float` | `0.6` | 0.01 to 1.0; lower is more deterministic. **v3 only.** |
| `enable_preprocessing` | `bool` | `False` | Normalize English words and numeric entities. Always on for v3 regardless. |
| `dict_id` | `str \| None` | `None` | Pronunciation dictionary to apply. **v3 only.** |
| `min_buffer_size` | `int` | `50` | Buffered characters that trigger a flush to the model — lower cuts first-audio latency. |
| `max_chunk_length` | `int` | `150` | Maximum sentence-split length. |

Because the defaults are always sent, model-specific knobs go out even when the chosen model has no use for them (`pitch`/`loudness` to `bulbul:v3`, `temperature` to `bulbul:v2`). Sarvam ignores those rather than rejecting the connection — the same thing the SDK's own `configure()` does.

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
3. **On `FINAL_TRANSCRIPT`** (once the optional [end-of-turn check](#end-of-turn-check) agrees the user is done), the pipeline starts a new agent turn: it streams your LangGraph agent's reply (`agent.astream(..., stream_mode="messages")`) and forwards each complete sentence to TTS as soon as a sentence/clause boundary is detected — so speech synthesis starts well before the agent has finished generating the full reply. Only the assistant's own text is spoken (tool results are never read aloud), and markdown/HTML (`**bold**`, tables, `<br>`) is stripped first; fragments with nothing speakable are skipped. Turns run one at a time, in order.
4. **TTS emits `TTSEvent`s** — `AUDIO` (a synthesized chunk), `END_OF_TURN`, `INTERRUPT`, `STREAM_CLOSED` — which the pipeline forwards verbatim to your `callback`. You decide what to do with each: play `AUDIO`, stop playback on `INTERRUPT`, mark the turn done on `END_OF_TURN`.
5. **Barge-in:** if the STT provider reports `SPEECH_START` while the agent is still generating or TTS is still speaking, the pipeline cancels the in-flight turn, tells TTS to interrupt, and notifies your callback — all before the next turn starts. If the reply has already been fully synthesized, only your callback is notified (to stop any audio still playing client-side). If the user barges in within `resume_window` seconds (default `2.0`) of first hearing the reply, they were most likely still mid-sentence and the end-of-turn call was premature — so their previous turn isn't dropped: it's merged with what they say next and answered as one turn ("I want to book a ticket to" + "Bangkok from Delhi"). A barge-in after the window is a real interruption ("stop", "ok thanks") and the previous turn is dropped. `resume_window=0` merges only replies the user never heard; `None` never merges. Pass `config=PipelineConfig(interrupt=False)` to `VoxkitPipeline` to disable this and let turns run to completion regardless of new speech. Speech during a reply is then answered after it.

### Connection drops and logging

The Sarvam providers reconnect on their own when a websocket drops (5 attempts, 1 s apart); the pipeline keeps running and you lose at most the audio in flight. `STREAM_CLOSED` is emitted only when reconnecting fails — on STT that stops the pipeline.

voxkit logs routine activity at `DEBUG` — the user's transcript and the agent's full reply on `voxkit.core.pipeline`, voice activity on `voxkit.stt.sarvam` — and only real failures at `WARNING`/`ERROR`. To watch a conversation:

```python
logging.getLogger("voxkit").setLevel(logging.DEBUG)
```

### End-of-turn check

STT VAD ends a turn on silence alone, so a mid-sentence pause ("book it for... um...") gets answered too early. Opt into [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) — a small audio model that judges from intonation and trailing-off whether the user has actually finished:

```bash
pip install "voxkit[smart-turn]"
```

```python
from voxkit import PipelineConfig, PipecatSmartTurnDetector, VoxkitPipeline

pipeline = VoxkitPipeline(
    stt, tts, agent, handle_tts_event,
    config=PipelineConfig(end_of_turn_timeout=2.0),
    end_of_turn=PipecatSmartTurnDetector(sample_rate=16000),  # one detector per pipeline: it buffers that call's audio
)
```

The pipeline feeds the input audio to the detector and scores each `FINAL_TRANSCRIPT`'s turn audio locally on CPU (~20 ms). If it sounds unfinished, the transcript is held for up to `end_of_turn_timeout` seconds of silence; further speech is appended and re-scored, and the text is sent anyway when the timeout expires. Needs `pcm_s16le` input; 16 kHz is recommended (8 kHz works but is less reliable). The ~8 MB model downloads from Hugging Face on first use. Implement `EndOfTurnDetector` to plug in your own logic. Off by default.

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
from voxkit.stt import SarvamSTTModel, SarvamSTTMode, SarvamSTTLanguageCode, SarvamSTTInputAudioCodec

from voxkit.tts import TTSProvider, TTSOptions, TTSEvent, TTSEventType
from voxkit.tts import SarvamTTSProvider, SarvamTTSOptions
from voxkit.tts import (
    SarvamTTSModel, SarvamTTSLanguageCode, SarvamTTSSpeaker,
    SarvamTTSAudioCodec, SarvamTTSAudioBitrate, SarvamTTSSampleRate,
)

from voxkit.llm import LLMEvent, LLMEventType

from voxkit.turn import EndOfTurnDetector, PipecatSmartTurnDetector

from voxkit.core import SentenceSegmenter, stream_agent_text, to_speakable
```

- **`VoxkitPipeline(stt, tts, agent, callback, config=None, end_of_turn=None)`** — the orchestrator. `agent` is any compiled LangGraph graph; `callback` is an `async def(event: TTSEvent) -> None` that receives every TTS event (exceptions it raises are logged, not fatal). `config` is a `PipelineConfig` (defaults to `PipelineConfig()`); `end_of_turn` is an optional `EndOfTurnDetector`. A pipeline runs once — create one per conversation.
- **`PipelineConfig(thread_id="default", interrupt=True, end_of_turn_timeout=2.0, resume_window=2.0)`** — the pipeline's behavioural knobs, as a frozen, stateless dataclass (safe to share). `thread_id` is passed to the agent's config on every turn so LangGraph-checkpointed memory persists across turns; `interrupt` toggles barge-in; `resume_window` decides when a barge-in means "I wasn't finished" (see How it works, step 5); `end_of_turn_timeout` bounds the optional [end-of-turn check](#end-of-turn-check).
- **`EndOfTurnDetector` / `PipecatSmartTurnDetector`** — the end-of-turn check interface (`push_audio`, `start_turn`, async `is_end_of_turn`) and the bundled Smart Turn v3 implementation (`smart-turn` extra).
- **`STTProvider` / `TTSProvider`** — abstract base classes a new provider implements to plug into the pipeline. `TTSProvider` also provides `speak(text)`, `end_turn()` and `interrupt(cancel_synthesis=True)`, which is how the pipeline drives it. See their docstrings (or the [API reference](#documentation) below) for the exact contract.
- **`SentenceSegmenter` / `to_speakable` / `stream_agent_text`** (`voxkit.core`) — the building blocks the pipeline uses to turn agent output into speech: sentence splitting, markdown stripping, and assistant-text-only streaming from a LangGraph agent.
- **`SarvamSTTProvider` / `SarvamTTSProvider`** — the bundled provider implementations, backed by [Sarvam AI](https://www.sarvam.ai/)'s streaming STT/TTS websockets. Configured via `SarvamSTTOptions` / `SarvamTTSOptions` ([full option tables](#sarvam-provider-options)). `SarvamSTTProvider` additionally exposes `await stt.flush()`, which forces Sarvam to finalize buffered audio without waiting for VAD — useful when you know the utterance is over (requires `flush_signal=True`).
- **Sarvam value types** — `SarvamSTTModel`, `SarvamSTTMode`, `SarvamSTTLanguageCode`, `SarvamSTTInputAudioCodec`, `SarvamTTSModel`, `SarvamTTSLanguageCode`, `SarvamTTSSpeaker`, `SarvamTTSAudioCodec`, `SarvamTTSAudioBitrate`, `SarvamTTSSampleRate` are `Literal` aliases enumerating every value Sarvam accepts, so bad models/voices/codecs fail at option construction instead of at connect time.

## Adding a new provider

Implement `STTProvider` or `TTSProvider` (`voxkit/stt/base.py`, `voxkit/tts/base.py`) — both are small interfaces (`connect`, `send`/`receive` for STT, `connect`/`synthesize` for TTS, plus `close`) that push/pull typed events through `asyncio.Queue`s. Providers should recover from dropped connections themselves and emit `STREAM_CLOSED` only when they can't. Nothing else in the pipeline needs to change; `VoxkitPipeline` only depends on these interfaces, not on Sarvam specifically.

## Documentation

Full API reference (generated from the docstrings in this repo) is published at **[sachanayush47.github.io/voxkit](https://sachanayush47.github.io/voxkit/)**.

To build the docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Development

Style and linting are handled entirely by [ruff](https://docs.astral.sh/ruff/), configured in [`pyproject.toml`](pyproject.toml). Docstrings are linted too (`pydocstyle`, google convention), since they're what the published API reference is generated from.

```bash
uv sync                     # install deps, including dev tooling
uv run pre-commit install   # once per clone: enable the git hook
uv run ruff check --fix .   # lint
uv run ruff format .        # format
```

The pre-commit hook runs `ruff check --fix` then `ruff format` on staged files. If it applies a fix, the commit aborts so you can re-stage — run it over everything at once with `uv run pre-commit run --all-files`.

## License

TBD.
