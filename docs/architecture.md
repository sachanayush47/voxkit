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
3. **On `FINAL_TRANSCRIPT`** (once any [end-of-turn check](#end-of-turn-check) agrees the user is done), the pipeline starts a new agent turn: it streams your LangGraph agent's reply (`agent.astream(..., stream_mode="messages")`) and forwards each complete sentence to TTS as soon as a sentence/clause boundary is detected — so speech synthesis starts well before the agent has finished generating the full reply. Only the assistant's own text is spoken (tool results are never read aloud), and markdown/HTML (`**bold**`, tables, `<br>`) is stripped first; fragments with nothing speakable are skipped. Turns run one at a time, in order.
4. **TTS emits [`TTSEvent`][voxkit.tts.base.TTSEvent]s** — `AUDIO` (a synthesized chunk), `END_OF_TURN`, `INTERRUPT`, `STREAM_CLOSED` — which the pipeline forwards verbatim to your `callback`. You decide what to do with each: play `AUDIO`, stop playback on `INTERRUPT`, mark the turn done on `END_OF_TURN`.
5. **Barge-in:** if the STT provider reports `SPEECH_START` while the agent is still generating or TTS is still speaking, the pipeline cancels the in-flight turn, tells TTS to interrupt, and notifies your callback — all before the next turn starts. If the reply has already been fully synthesized, only your callback is notified (to stop any audio still playing client-side). If the user barges in within `resume_window` seconds (default `2.0`) of first hearing the reply, they were most likely still mid-sentence and the end-of-turn call was premature — so their previous turn isn't dropped: it's merged with what they say next and answered as one turn ("I want to book a ticket to" + "Bangkok from Delhi"). A barge-in after the window is a real interruption ("stop", "ok thanks") and the previous turn is dropped. `resume_window=0` merges only replies the user never heard; `None` never merges. Pass `config=PipelineConfig(interrupt=False)` to [`VoxkitPipeline`][voxkit.core.pipeline.VoxkitPipeline] to disable this and let turns run to completion regardless of new speech. Speech during a reply is then answered after it.

    Because voxkit bundles no VAD of its own, *when* `SPEECH_START` fires is entirely the STT provider's decision — so how twitchy barge-in feels is tuned on the provider's options, not on the pipeline. With [`SarvamSTTOptions`][voxkit.stt.sarvam.SarvamSTTOptions] the relevant knobs are `interrupt_min_speech_frames` (how much speech must accumulate before a barge-in counts), `high_vad_sensitivity` plus `positive_speech_threshold`/`negative_speech_threshold` (what counts as speech at all), and `start_speech_volume_threshold` (a dB gate that keeps background noise from triggering turns).

## Connection drops and logging

The Sarvam providers reconnect on their own when a websocket drops (5 attempts, 1 s apart); the pipeline keeps running and you lose at most the audio in flight. `STREAM_CLOSED` is emitted only when reconnecting fails — on STT that stops the pipeline.

voxkit logs routine activity at `DEBUG` — the user's transcript and the agent's full reply on `voxkit.core.pipeline`, voice activity on `voxkit.stt.sarvam` — and only real failures at `WARNING`/`ERROR`. To watch a conversation:

```python
logging.getLogger("voxkit").setLevel(logging.DEBUG)
```

## End-of-turn check

STT providers end a speech segment on silence alone, so a user who pauses mid-thought ("I'd like to book an appointment for... um...") gets their half-sentence finalized and answered. Pass an [`EndOfTurnDetector`][voxkit.turn.base.EndOfTurnDetector] as `VoxkitPipeline(..., end_of_turn=...)` to add a second check. voxkit ships [`PipecatSmartTurnDetector`][voxkit.turn.smart_turn.PipecatSmartTurnDetector], which runs pipecat's [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) audio model locally on CPU (~20 ms per check, no API key):

- The pipeline feeds every chunk of the input audio stream to the detector (the same bytes it forwards to STT). On `SPEECH_START` for a fresh turn, the detector drops older audio, keeping a 1 s pre-roll.
- On each `FINAL_TRANSCRIPT`, the transcript is appended to a held transcript and the detector scores the last 8 s of the turn's audio — intonation, trailing fillers, cut-off words.
- **Complete** (probability ≥ `threshold`, default `0.5`) → the held text starts an agent turn immediately.
- **Incomplete** → the pipeline waits `end_of_turn_timeout` seconds (default `2.0`). If the user speaks again (`SPEECH_START`), the wait pauses; their next `FINAL_TRANSCRIPT` is appended and the turn re-scored on the longer audio. If the timeout expires, the held text is sent anyway, so the user is never left unanswered. A `SPEECH_END` with no new transcript (e.g. noise) resumes the wait.
- If the detector raises, the turn is treated as complete.
- A wrong "complete" verdict is recoverable: if the user keeps talking within `resume_window` of the reply starting, barge-in cancels it and the text is merged into the continuation (see step 5 above).
- `PipecatSmartTurnDetector` logs each score at `DEBUG` (`voxkit.turn.smart_turn`), which helps tune `threshold`.

Smart Turn needs mono 16-bit PCM (`pcm_s16le`) input. It's trained on 16 kHz audio; 8 kHz (telephony) input is upsampled but judged less reliably. Install with `pip install "voxkit[smart-turn]"` (or `"voxkit[all]"`); the ~8 MB model downloads from Hugging Face on first use. Off by default — consider also lengthening the STT provider's own silence window (`negative_frames_count`/`negative_frames_window` on [`SarvamSTTOptions`][voxkit.stt.sarvam.SarvamSTTOptions]).

## Event types

| Module | Type | Values |
|---|---|---|
| `voxkit.stt` | `STTEventType` | `SPEECH_START`, `SPEECH_END`, `PARTIAL_TRANSCRIPT`, `FINAL_TRANSCRIPT`, `STREAM_CLOSED` |
| `voxkit.llm` | `LLMEventType` | `SENTENCE`, `END_OF_TURN`, `INTERRUPT` |
| `voxkit.tts` | `TTSEventType` | `AUDIO`, `END_OF_TURN`, `INTERRUPT`, `STREAM_CLOSED` |

## Adding a new provider

Implement [`STTProvider`][voxkit.stt.base.STTProvider] or [`TTSProvider`][voxkit.tts.base.TTSProvider] — both are small interfaces (`connect`, `send`/`receive` for STT, `connect`/`synthesize` for TTS, plus `close`) that push/pull typed events through `asyncio.Queue`s. Providers should recover from dropped connections themselves and emit `STREAM_CLOSED` only when they can't. Nothing else in the pipeline needs to change; `VoxkitPipeline` only depends on these interfaces, not on any specific vendor.
