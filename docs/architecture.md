# Architecture

```
audio in ──▶ STTProvider ──▶ VoxkitPipeline ──▶ LangGraph agent
                                    │                  │
                                    │  sentence-by-     │ streamed
                                    │  sentence         ▼ tokens
                                    └──────────▶ TTSProvider ──▶ audio out (your callback)
```

1. **You feed raw audio** into `pipeline.run(audio_stream)`. It's forwarded to the STT provider.
2. **STT emits [`STTEvent`][voxkit.stt.base.STTEvent]s** — `SPEECH_START`/`SPEECH_END` (voice activity), `PARTIAL_TRANSCRIPT`, `FINAL_TRANSCRIPT`, `STREAM_CLOSED`.
3. **On `FINAL_TRANSCRIPT`**, the pipeline starts a new agent turn: it streams tokens from your LangGraph agent (`agent.astream(..., stream_mode="messages")`), buffers them, and forwards each complete sentence to TTS as soon as a sentence/clause boundary is detected — so speech synthesis starts well before the agent has finished generating the full reply.
4. **TTS emits [`TTSEvent`][voxkit.tts.base.TTSEvent]s** — `AUDIO` (a synthesized chunk), `END_OF_TURN`, `INTERRUPT`, `STREAM_CLOSED` — which the pipeline forwards verbatim to your `callback`. You decide what to do with each: play `AUDIO`, stop playback on `INTERRUPT`, mark the turn done on `END_OF_TURN`.
5. **Barge-in:** if the STT provider reports `SPEECH_START` while the agent is still generating or TTS is still speaking, the pipeline cancels the in-flight turn, tells TTS to interrupt, and notifies your callback — all before the next turn starts. Pass `config=PipelineConfig(interrupt=False)` to [`VoxkitPipeline`][voxkit.core.pipeline.VoxkitPipeline] to disable this and let turns run to completion regardless of new speech.

    Because voxkit bundles no VAD of its own, *when* `SPEECH_START` fires is entirely the STT provider's decision — so how twitchy barge-in feels is tuned on the provider's options, not on the pipeline. With [`SarvamSTTOptions`][voxkit.stt.sarvam.SarvamSTTOptions] the relevant knobs are `interrupt_min_speech_frames` (how much speech must accumulate before a barge-in counts), `high_vad_sensitivity` plus `positive_speech_threshold`/`negative_speech_threshold` (what counts as speech at all), and `start_speech_volume_threshold` (a dB gate that keeps background noise from triggering turns).

## Event types

| Module | Type | Values |
|---|---|---|
| `voxkit.stt` | `STTEventType` | `SPEECH_START`, `SPEECH_END`, `PARTIAL_TRANSCRIPT`, `FINAL_TRANSCRIPT`, `STREAM_CLOSED` |
| `voxkit.llm` | `LLMEventType` | `SENTENCE`, `END_OF_TURN`, `INTERRUPT` |
| `voxkit.tts` | `TTSEventType` | `AUDIO`, `END_OF_TURN`, `INTERRUPT`, `STREAM_CLOSED` |

## Adding a new provider

Implement [`STTProvider`][voxkit.stt.base.STTProvider] or [`TTSProvider`][voxkit.tts.base.TTSProvider] — both are small interfaces (`connect`, `send`/`receive` for STT, `connect`/`synthesize` for TTS, plus `close`) that push/pull typed events through `asyncio.Queue`s. Nothing else in the pipeline needs to change; `VoxkitPipeline` only depends on these interfaces, not on any specific vendor.
