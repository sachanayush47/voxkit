"""The event-driven STT -> LangGraph agent -> TTS orchestrator.

:class:`VoxkitPipeline` is voxkit's core: it wires an
:class:`~voxkit.stt.base.STTProvider`, a LangGraph agent, and a
:class:`~voxkit.tts.base.TTSProvider` together, streaming audio in and audio
events out while handling turn-taking and barge-in (interrupt) internally.
"""

import asyncio
import logging
import re

from typing import Awaitable, Callable, AsyncIterator, Optional

from langgraph.graph.state import CompiledStateGraph

from voxkit.stt import STTEvent, STTEventType, STTProvider
from voxkit.tts import TTSEvent, TTSEventType, TTSProvider
from voxkit.llm import LLMEvent, LLMEventType

logger = logging.getLogger(__name__)

SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\s]|[,;][\s]")
"""Matches a sentence/clause boundary in streamed LLM output (terminal punctuation followed by whitespace)."""


class VoxkitPipeline:
    """Runs a full voice-agent turn loop: audio in, agent reasoning, audio out.

    The pipeline consumes an audio stream, feeds it to ``stt``, hands each
    finalized transcript to ``agent`` as a new turn, streams the agent's
    reply to ``tts`` sentence-by-sentence as it's generated, and forwards
    every :class:`~voxkit.tts.base.TTSEvent` (synthesized audio, turn
    boundaries, interrupts) to ``callback`` for the caller to act on (e.g.
    play audio, clear a playback buffer).

    Barge-in is handled internally: if ``interrupt`` is enabled and the STT
    provider reports :attr:`~voxkit.stt.base.STTEventType.SPEECH_START` while
    the agent is still generating or the TTS provider is still speaking, the
    in-flight turn is cancelled and the TTS provider is told to interrupt.

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
        thread_id: str = "default",
        interrupt: bool = True,
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
                turn; any graph exposing that streaming shape works.
            callback: Called with every :class:`~voxkit.tts.base.TTSEvent`
                (audio chunks, turn/interrupt markers) as it's produced. This
                is the pipeline's only output channel to the caller -- e.g.
                write audio to a speaker, or forward it over a websocket.
            thread_id: Passed to the agent as ``configurable.thread_id`` on
                every turn, so LangGraph-checkpointed conversation memory
                persists across turns within this pipeline instance.
            interrupt: If ``True`` (default), a detected
                :attr:`~voxkit.stt.base.STTEventType.SPEECH_START` cancels
                the in-flight agent turn and interrupts TTS playback
                (barge-in). If ``False``, ``SPEECH_START`` never interrupts
                the current turn.
        """
        self.stt: STTProvider = stt
        self.tts: TTSProvider = tts
        self.agent: CompiledStateGraph = agent
        self.callback = callback
        self.thread_id = thread_id
        self.interrupt = interrupt

        self.stt_output_queue: asyncio.Queue[STTEvent] = self.stt.get_output_queue()
        self.llm_output_queue: asyncio.Queue[LLMEvent] = self.tts.get_input_queue()
        self.tts_output_queue: asyncio.Queue[TTSEvent] = self.tts.get_output_queue()

        self._background_tasks: list[asyncio.Task] = []
        self._turn_task: Optional[asyncio.Task] = None
        self._cancel_event: asyncio.Event = asyncio.Event()

        # Tracks whether the bot is actually speaking right now
        self._is_bot_speaking: bool = False

    async def run(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Connect the providers and run the pipeline until the STT stream closes.

        Blocks until :attr:`~voxkit.stt.base.STTEventType.STREAM_CLOSED` is
        received from ``stt``, then calls :meth:`shutdown` automatically
        (whether it exits normally or via an exception/cancellation).

        Args:
            audio_stream: An async iterator yielding raw audio byte chunks to
                feed to the STT provider, in the encoding/sample rate that
                provider expects.
        """
        await self.stt.connect()
        await self.tts.connect()
        self.tts.synthesize()

        self._background_tasks.append(asyncio.create_task(self.stt.send(audio_stream)))
        self._background_tasks.append(asyncio.create_task(self.stt.receive()))
        self._background_tasks.append(asyncio.create_task(self.__consume_tts_output()))

        try:
            await self.__consume_stt_events()
        finally:
            await self.shutdown()

    async def __consume_stt_events(self) -> None:
        """Background loop: react to each :class:`~voxkit.stt.base.STTEvent` as it arrives from STT."""
        while True:
            event = await self.stt_output_queue.get()

            if event.type == STTEventType.SPEECH_START:
                # Interrupt: user started talking while the agent may still be
                # generating/speaking. Cancel the in-flight turn immediately.
                if self.interrupt:
                    await self.__handle_interrupt()

            elif event.type == STTEventType.FINAL_TRANSCRIPT:
                if event.text and event.text.strip():
                    await self.__handle_user_turn(event.text.strip())

            elif event.type == STTEventType.SPEECH_END:
                pass

            elif event.type == STTEventType.PARTIAL_TRANSCRIPT:
                pass

            elif event.type == STTEventType.STREAM_CLOSED:
                logger.error("VoxkitPipeline: STT stream closed, stopping pipeline")
                return

    async def __handle_interrupt(self) -> None:
        """Cancel the in-flight agent turn (if any) and tell TTS/the client to stop."""
        # Always drain + signal tts_output_queue
        # Draining first, then signaling, preserves ordering: nothing stale
        # can arrive at the client after this INTERRUPT event, since
        # __consume_tts_output is the only thing that ever calls the client
        # callback, and it processes this queue strictly in order.
        await self.__drain(self.tts_output_queue)
        await self.__signal(self.tts_output_queue, TTSEvent(TTSEventType.INTERRUPT))

        if not self._is_bot_speaking:
            # Nothing server-side needs cancelling/reconnecting - the LLM
            # already finished this turn and TTS already sent everything.
            # The client-side notify above is enough to handle any audio
            # still sitting in the client's own playback buffer.
            return

        logger.info("Interrupt detected, cancelling in-flight generation and reconnecting TTS")

        if self._turn_task and not self._turn_task.done():
            self._cancel_event.set()
            self._turn_task.cancel()

        # This is the expensive path (triggers SarvamTTSProvider._reconnect())
        # Only worth paying when the server genuinely believes generation
        # or synthesis is still in flight.
        await self.__drain(self.llm_output_queue)
        await self.__signal(self.llm_output_queue, LLMEvent(LLMEventType.INTERRUPT))

        self._is_bot_speaking = False

    async def __drain(self, queue: "asyncio.Queue") -> None:
        """Discard every item currently sitting in ``queue`` without blocking."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def __handle_user_turn(self, text: str) -> None:
        """Kick off a new agent turn for a finalized user utterance.

        Args:
            text: The finalized transcript text for this turn.
        """
        # Fresh cancel event per turn - the previous one (if any) stays set for
        # the task that's unwinding; this one is what the new turn checks.
        self._cancel_event = asyncio.Event()
        self._is_bot_speaking = True  # Optimistic - sentences will start flowing to TTS momentarily
        self._turn_task = asyncio.create_task(self.__run_agent_turn(text, self._cancel_event))

    async def __run_agent_turn(self, text: str, cancel_event: asyncio.Event) -> None:
        """Stream the agent's reply for ``text`` and forward each sentence to TTS.

        Args:
            text: The user's transcript for this turn.
            cancel_event: Set to abandon this turn early (e.g. on barge-in).
        """
        try:
            async for sentence in self.__stream_agent_sentences(text, cancel_event):
                if cancel_event.is_set():
                    break
                await self.llm_output_queue.put(LLMEvent(LLMEventType.SENTENCE, sentence))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("VoxkitPipeline: agent turn failed")
        finally:
            # Sentinel: tells the TTS provider this turn's sentences are complete.
            # Distinct from LLMEventType.INTERRUPT -- this means "nothing more is
            # coming for now," not "stop what's currently playing." If an
            # interrupt already fired for this same turn, the provider will see
            # INTERRUPT followed by END_OF_TURN back to back -- harmless.
            await self.__signal(self.llm_output_queue, LLMEvent(LLMEventType.END_OF_TURN))

    async def __signal(self, queue: "asyncio.Queue", event: LLMEvent | TTSEvent) -> None:
        """Non-blocking push for control events (``END_OF_TURN``/``INTERRUPT``).

        Usable against either ``llm_output_queue`` or ``tts_output_queue``.
        Control events aren't real content and shouldn't be subject to the
        same backpressure as sentences/audio -- if a queue is bounded and
        full, a plain ``put()`` would suspend waiting for space, which
        defeats the purpose of a signal that needs to land immediately.
        Evicts the oldest item instead of waiting.

        Args:
            queue: The queue to push onto (``llm_output_queue`` or ``tts_output_queue``).
            event: The control event to push.
        """
        while True:
            try:
                queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

    async def __stream_agent_sentences(
        self, text: str, cancel_event: asyncio.Event
    ) -> AsyncIterator[str]:
        """Stream tokens from the LangGraph agent, yielding complete sentences as boundaries are found.

        NOTE: Verify this against your actual LangGraph version. ``stream_mode="messages"``
        is the current pattern for token-level streaming in recent LangGraph releases,
        yielding ``(message_chunk, metadata)`` tuples where ``message_chunk.content`` holds the
        incremental text. If your version streams differently, adjust this loop --
        don't assume this shape is correct without checking.

        Args:
            text: The user's transcript to send to the agent as this turn's input.
            cancel_event: Checked between tokens; stops streaming early when set.

        Yields:
            Each complete sentence/clause as soon as a boundary
            (:data:`SENTENCE_BOUNDARY`) is detected, plus any trailing partial
            sentence once the agent finishes (unless cancelled).
        """
        buffer = ""
        config = {"configurable": {"thread_id": self.thread_id}}

        async for message_chunk, _metadata in self.agent.astream(
            {"messages": [("user", text)]},
            config=config,
            stream_mode="messages",
        ):
            if cancel_event.is_set():
                return

            token = getattr(message_chunk, "content", "") or ""
            if not token:
                continue

            buffer += token
            match = SENTENCE_BOUNDARY.search(buffer)
            if match:
                boundary_idx = match.end()
                sentence, buffer = buffer[:boundary_idx], buffer[boundary_idx:]
                if sentence.strip():
                    yield sentence.strip()

        if buffer.strip() and not cancel_event.is_set():
            yield buffer.strip()

    async def __consume_tts_output(self) -> None:
        """Background loop: forward every :class:`~voxkit.tts.base.TTSEvent` from TTS to ``callback``.

        The pipeline doesn't unpack or transform the event for the client --
        it forwards the full ``TTSEvent`` (type + payload) so the client can
        branch on ``event.type`` itself (``AUDIO`` -> play, ``INTERRUPT`` ->
        stop/clear playback, ``END_OF_TURN`` -> mark the bot's turn as
        finished, etc). Server-side logging still happens here for
        observability, independent of what the client does with it.
        """
        while True:
            event = await self.tts_output_queue.get()

            if event.type == TTSEventType.END_OF_TURN:
                logger.debug("VoxkitPipeline: TTS finished speaking this turn")
                self._is_bot_speaking = False

            elif event.type == TTSEventType.INTERRUPT:
                logger.info("VoxkitPipeline: forwarding barge-in to client")

            elif event.type == TTSEventType.STREAM_CLOSED:
                # The provider already attempts its own reconnect internally
                # before giving up -- by the time this event reaches us, that
                # has either already succeeded (and audio will keep flowing)
                # or the provider's internal task has ended for good. Logged
                # here for observability; still forwarded below in case the
                # client wants to show a connection-issue indicator.
                logger.error(
                    "VoxkitPipeline: TTS reported STREAM_CLOSED -- if this "
                    "doesn't self-resolve, the provider's internal task may "
                    "have died; consider monitoring task health directly."
                )

            await self.callback(event)

    async def shutdown(self) -> None:
        """Cancel all background tasks and close both providers.

        Called automatically by :meth:`run` on exit; safe to call directly
        (e.g. to stop the pipeline early from outside).
        """
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self.stt.close()
        await self.tts.close()
