"""The event-driven STT -> LangGraph agent -> TTS orchestrator.

:class:`VoxkitPipeline` is voxkit's core: it wires an
:class:`~voxkit.stt.base.STTProvider`, a LangGraph agent, and a
:class:`~voxkit.tts.base.TTSProvider` together, streaming audio in and audio
events out while handling turn-taking and barge-in (interrupt) internally.
"""

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from voxkit.core.agent import stream_agent_text
from voxkit.core.text import SentenceSegmenter
from voxkit.core.turn_gate import EndOfTurnGate
from voxkit.stt import STTEventType, STTProvider
from voxkit.tts import TTSEvent, TTSEventType, TTSProvider
from voxkit.turn import EndOfTurnDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Behavioural settings for a :class:`VoxkitPipeline`.

    Holds the pipeline's tuning knobs, keeping them separate from the
    collaborators (providers, agent, callback, detector) passed to
    :class:`VoxkitPipeline`. Every field has a default, so
    ``PipelineConfig()`` is a valid "just use the defaults" config. Frozen and
    stateless, so one config can safely be shared by many pipelines.

    Attributes:
        thread_id: Passed to the agent as ``configurable.thread_id`` on every
            turn, so LangGraph-checkpointed conversation memory persists across
            turns within one pipeline instance. Use a distinct value per
            concurrent conversation.
        interrupt: If ``True`` (default), a detected
            :attr:`~voxkit.stt.base.STTEventType.SPEECH_START` cancels the
            in-flight agent turn and interrupts TTS playback (barge-in). If
            ``False``, turns always run to completion; speech during a reply
            is answered after it.
        end_of_turn_timeout: Seconds of silence to wait after the end-of-turn
            detector judges a transcript incomplete before responding anyway.
            The wait pauses while the user is speaking again. Ignored unless
            :class:`VoxkitPipeline` is given an ``end_of_turn`` detector.
        resume_window: Seconds after the user first hears a reply during which
            barging in counts as "I wasn't finished": the reply is cancelled
            and the user's previous turn is merged with what they say next,
            so the agent answers the whole sentence. Barging in later counts
            as a real interruption ("stop", "ok thanks") and the previous turn
            is dropped. ``0`` merges only replies the user never heard any of;
            ``None`` never merges. Only applies when :attr:`interrupt` is on.

    Raises:
        ValueError: If ``end_of_turn_timeout`` or ``resume_window`` is negative.

    Example:
        >>> config = PipelineConfig(thread_id="caller-42", interrupt=False)
        >>> pipeline = VoxkitPipeline(stt, tts, agent, handle_tts_event, config=config)
    """

    thread_id: str = "default"
    interrupt: bool = True
    end_of_turn_timeout: float = 2.0
    resume_window: float | None = 2.0

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.end_of_turn_timeout < 0:
            raise ValueError(f"end_of_turn_timeout must be >= 0, got {self.end_of_turn_timeout}")
        if self.resume_window is not None and self.resume_window < 0:
            raise ValueError(f"resume_window must be >= 0 or None, got {self.resume_window}")


@dataclass(eq=False)
class _Turn:
    """A committed user turn and the reply being produced for it."""

    text: str
    task: asyncio.Task | None = None
    first_audio_at: float | None = None


class VoxkitPipeline:
    """Runs a full voice-agent turn loop: audio in, agent reasoning, audio out.

    The pipeline consumes an audio stream, feeds it to ``stt``, hands each
    completed user turn to ``agent``, streams the agent's reply to ``tts``
    sentence-by-sentence as it's generated, and forwards every
    :class:`~voxkit.tts.base.TTSEvent` (synthesized audio, turn boundaries,
    interrupts) to ``callback``.

    Turns run one at a time, in order. Barge-in: if
    :attr:`PipelineConfig.interrupt` is enabled and the STT provider reports
    :attr:`~voxkit.stt.base.STTEventType.SPEECH_START` while a reply is being
    generated or spoken, that reply is cancelled and the client is told to
    stop playback.

    The user's transcript and the agent's full reply are logged at ``DEBUG``
    on the ``voxkit.core.pipeline`` logger.

    A pipeline runs once: create a new one per conversation.

    Example:
        >>> async def handle_tts_event(event: TTSEvent) -> None:
        ...     if event.type == TTSEventType.AUDIO:
        ...         play(event.audio)
        >>> pipeline = VoxkitPipeline(stt, tts, agent, handle_tts_event)
        >>> await pipeline.run(microphone_stream())
    """

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        agent: CompiledStateGraph,
        callback: Callable[[TTSEvent], Awaitable[None]],
        config: PipelineConfig | None = None,
        end_of_turn: EndOfTurnDetector | None = None,
    ) -> None:
        """Wire up the pipeline. Call :meth:`run` to start it.

        Args:
            stt: The speech-to-text provider that turns the incoming audio
                stream into transcripts.
            tts: The text-to-speech provider that turns agent sentences into
                audio.
            agent: A compiled LangGraph graph (e.g. from
                ``langchain.agents.create_agent``). Invoked via
                ``agent.astream(..., stream_mode="messages")`` once per user
                turn; only its assistant text is spoken, never tool output.
            callback: Called with every :class:`~voxkit.tts.base.TTSEvent`
                (audio chunks, turn/interrupt markers) as it's produced -- the
                pipeline's only output channel to the caller. Exceptions it
                raises are logged and don't stop the pipeline.
            config: Behavioural settings. Defaults to ``PipelineConfig()``.
            end_of_turn: Optional end-of-turn check (e.g.
                :class:`~voxkit.turn.smart_turn.PipecatSmartTurnDetector`). If set,
                it is fed the input audio and consulted on each transcript;
                one judged incomplete is held until the user finishes or
                :attr:`PipelineConfig.end_of_turn_timeout` expires. Detectors
                hold per-conversation state, so give each pipeline its own.
        """
        self.stt = stt
        self.tts = tts
        self.agent = agent
        self.callback = callback
        self.config = config or PipelineConfig()

        self._gate = EndOfTurnGate(end_of_turn, self.config.end_of_turn_timeout, self._start_agent_turn)
        self._background_tasks: list[asyncio.Task] = []
        self._turns: list[_Turn] = []
        self._speaking: deque[_Turn] = deque()
        self._last_heard: _Turn | None = None
        self._client_has_audio = False
        self._started = False

    @property
    def _bot_active(self) -> bool:
        """Whether a reply is still being generated or synthesized."""
        return bool(self._turns)

    async def run(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Connect the providers and run the pipeline until the STT stream closes.

        Blocks until :attr:`~voxkit.stt.base.STTEventType.STREAM_CLOSED` is
        received from ``stt``, then calls :meth:`shutdown` automatically
        (whether it exits normally or via an exception/cancellation).

        Args:
            audio_stream: An async iterator yielding raw audio byte chunks to
                feed to the STT provider, in the encoding/sample rate that
                provider expects.

        Raises:
            RuntimeError: If the pipeline has already been run.
        """
        if self._started:
            raise RuntimeError("VoxkitPipeline.run() can only be called once; create a new pipeline")
        self._started = True

        await self.stt.connect()
        await self.tts.connect()
        self.tts.synthesize()

        self._background_tasks += [
            asyncio.create_task(self.stt.send(self._tap_audio(audio_stream))),
            asyncio.create_task(self.stt.receive()),
            asyncio.create_task(self._forward_tts_events()),
        ]
        try:
            await self._consume_stt_events()
        finally:
            await self.shutdown()

    async def _tap_audio(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Pass ``audio_stream`` through unchanged, feeding each chunk to the end-of-turn gate.

        Args:
            audio_stream: The caller's audio stream.

        Yields:
            Each chunk of ``audio_stream``, in order.
        """
        async for chunk in audio_stream:
            self._gate.push_audio(chunk)
            yield chunk

    async def _consume_stt_events(self) -> None:
        """Route each :class:`~voxkit.stt.base.STTEvent` until the STT stream closes."""
        queue = self.stt.get_output_queue()
        while True:
            event = await queue.get()
            match event.type:
                case STTEventType.SPEECH_START:
                    await self._on_speech_start()
                case STTEventType.SPEECH_END:
                    self._gate.on_speech_end()
                case STTEventType.FINAL_TRANSCRIPT:
                    if event.text and event.text.strip():
                        self._gate.on_transcript(event.text.strip())
                case STTEventType.STREAM_CLOSED:
                    logger.error("VoxkitPipeline: STT stream closed, stopping pipeline")
                    return

    async def _on_speech_start(self) -> None:
        """Barge in on the current reply (if enabled), then let the gate know the user is talking."""
        if self.config.interrupt and (self._bot_active or self._client_has_audio):
            await self._interrupt()
        self._gate.on_speech_start()

    async def _interrupt(self) -> None:
        """Cancel every queued or running reply and stop TTS and client playback.

        Turns whose reply the user hadn't heard for longer than
        :attr:`PipelineConfig.resume_window` are handed back to the
        end-of-turn gate, to be merged with what the user says next.
        """
        now = asyncio.get_running_loop().time()
        cancel_synthesis = self._bot_active
        candidates = [self._last_heard, *self._turns] if self._last_heard not in self._turns else self._turns
        resumed = [turn.text for turn in candidates if turn is not None and self._in_resume_window(turn, now)]
        logger.debug("VoxkitPipeline: barge-in (cancelling synthesis: %s, resuming: %r)", cancel_synthesis, resumed)

        for turn in self._turns:
            turn.task.cancel()
        self._turns.clear()
        self._speaking.clear()
        self._last_heard = None
        self._client_has_audio = False
        await self.tts.interrupt(cancel_synthesis=cancel_synthesis)
        if resumed:
            self._gate.restore(" ".join(resumed))

    def _in_resume_window(self, turn: _Turn, now: float) -> bool:
        """Whether barging in on ``turn``'s reply now means the user wasn't finished.

        Args:
            turn: The interrupted turn.
            now: The current event-loop time.

        Returns:
            ``True`` if ``turn``'s text should be merged into the next turn.
        """
        window = self.config.resume_window
        if window is None:
            return False
        return turn.first_audio_at is None or now - turn.first_audio_at <= window

    def _start_agent_turn(self, text: str) -> None:
        """Queue an agent reply to ``text`` behind any reply still running.

        Args:
            text: The completed user turn.
        """
        logger.debug("VoxkitPipeline: user turn %r", text)
        turn = _Turn(text)
        previous = self._turns[-1].task if self._turns else None
        turn.task = asyncio.create_task(self._run_agent_turn(turn, previous))
        turn.task.add_done_callback(lambda _task: self._on_turn_done(turn))
        self._turns.append(turn)

    def _on_turn_done(self, turn: _Turn) -> None:
        """Forget a turn whose task ended without handing anything to TTS.

        Turns that did speak stay tracked until TTS reports their ``END_OF_TURN``.

        Args:
            turn: The turn whose task just finished.
        """
        if turn not in self._speaking and turn in self._turns:
            self._turns.remove(turn)

    async def _run_agent_turn(self, turn: _Turn, previous: asyncio.Task | None) -> None:
        """Stream the agent's reply to ``turn`` into TTS, after ``previous`` finishes.

        Cancellation (barge-in) skips the closing ``end_turn``, so a cancelled
        reply never produces a late ``END_OF_TURN``.

        Args:
            turn: The user turn to reply to.
            previous: The task of the turn queued before this one, if any.
        """
        segmenter = SentenceSegmenter()
        spoken: list[str] = []
        try:
            if previous is not None:
                await asyncio.wait({previous})
            async for chunk in stream_agent_text(self.agent, turn.text, self.config.thread_id):
                for sentence in segmenter.push(chunk):
                    await self._speak(turn, sentence, spoken)
            for sentence in segmenter.flush():
                await self._speak(turn, sentence, spoken)
        except asyncio.CancelledError:
            logger.debug("VoxkitPipeline: reply cancelled after %r", " ".join(spoken))
            raise
        except Exception:
            logger.exception("VoxkitPipeline: agent turn failed")

        logger.debug("VoxkitPipeline: agent replied %r", " ".join(spoken))
        if spoken:
            await self.tts.end_turn()

    async def _speak(self, turn: _Turn, sentence: str, spoken: list[str]) -> None:
        """Send one sentence of ``turn``'s reply to TTS.

        Args:
            turn: The turn the sentence belongs to.
            sentence: The sentence to speak.
            spoken: Sentences of this reply sent so far; appended to.
        """
        if not spoken:
            self._speaking.append(turn)
        await self.tts.speak(sentence)
        spoken.append(sentence)

    async def _forward_tts_events(self) -> None:
        """Forward every :class:`~voxkit.tts.base.TTSEvent` from TTS to ``callback``, in order."""
        queue = self.tts.get_output_queue()
        while True:
            event = await queue.get()
            match event.type:
                case TTSEventType.AUDIO:
                    self._client_has_audio = True
                    if self._speaking and self._speaking[0].first_audio_at is None:
                        self._speaking[0].first_audio_at = asyncio.get_running_loop().time()
                        self._last_heard = self._speaking[0]
                case TTSEventType.END_OF_TURN:
                    if self._speaking:
                        finished = self._speaking.popleft()
                        if finished in self._turns:
                            self._turns.remove(finished)
                case TTSEventType.STREAM_CLOSED:
                    logger.error("VoxkitPipeline: TTS stream closed and could not reconnect")
            try:
                await self.callback(event)
            except Exception:
                logger.exception("VoxkitPipeline: callback failed on %s", event.type.name)

    async def shutdown(self) -> None:
        """Cancel all background work and close both providers.

        Called automatically by :meth:`run` on exit; safe to call directly
        (e.g. to stop the pipeline early from outside).
        """
        self._gate.close()
        tasks = [*self._background_tasks, *(turn.task for turn in self._turns)]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.stt.close()
        await self.tts.close()
