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


class VoxkitPipeline:
    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        agent: CompiledStateGraph,
        callback: Callable[[TTSEvent], Awaitable[None]],
        thread_id: str = "default",
        interrupt: bool = True,
    ):
        self.stt: STTProvider = stt
        self.tts: TTSProvider = tts
        self.agent: CompiledStateGraph = agent
        self.callback = callback  # receives full TTSEvent -- client decides what to do per event.type
        self.thread_id = thread_id  # Passed to the agent on every turn so checkpointed memory persists
        self.interrupt = interrupt  # if False, SPEECH_START never cancels/barges in on the current turn

        self.stt_output_queue: asyncio.Queue[STTEvent] = self.stt.get_output_queue()
        self.llm_output_queue: asyncio.Queue[LLMEvent] = self.tts.get_input_queue()
        self.tts_output_queue: asyncio.Queue[TTSEvent] = self.tts.get_output_queue()

        self._background_tasks: list[asyncio.Task] = []
        self._turn_task: Optional[asyncio.Task] = None
        self._cancel_event: asyncio.Event = asyncio.Event()
        
        # Tracks whether the bot is actually speaking right now
        self._is_bot_speaking: bool = False

    async def run(self, audio_stream: AsyncIterator[bytes]):
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

    async def __consume_stt_events(self):
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

    async def __handle_interrupt(self):
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

    async def __drain(self, queue: "asyncio.Queue"):
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def __handle_user_turn(self, text: str):
        # Fresh cancel event per turn - the previous one (if any) stays set for
        # the task that's unwinding; this one is what the new turn checks.
        self._cancel_event = asyncio.Event()
        self._is_bot_speaking = True  # Optimistic - sentences will start flowing to TTS momentarily
        self._turn_task = asyncio.create_task(self.__run_agent_turn(text, self._cancel_event))

    async def __run_agent_turn(self, text: str, cancel_event: asyncio.Event):
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

    async def __signal(self, queue: "asyncio.Queue", event):
        """
        Non-blocking push for control events (END_OF_TURN / INTERRUPT), usable
        against either llm_output_queue or tts_output_queue. Control events
        aren't real content and shouldn't be subject to the same backpressure
        as sentences/audio -- if a queue is bounded and full, a plain `put()`
        would suspend waiting for space, which defeats the purpose of a signal
        that needs to land immediately. Evict the oldest item instead of waiting.
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
        """
        Streams tokens from the LangGraph/LangChain agent and yields complete sentences
        as soon as a boundary is detected.

        NOTE: Verify this against your actual LangGraph version. `stream_mode="messages"`
        is the current pattern for token-level streaming in recent LangGraph releases,
        yielding (message_chunk, metadata) tuples where message_chunk.content holds the
        incremental text. If your version streams differently, adjust this loop --
        don't assume this shape is correct without checking.
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

    async def __consume_tts_output(self):
        """
        Drains events from the TTS provider's output queue and forwards each
        one, as-is, to the client callback. The pipeline doesn't unpack or
        transform the event for the client -- it forwards the full TTSEvent
        (type + payload) so the client can branch on event.type itself
        (AUDIO -> play, INTERRUPT -> stop/clear playback, END_OF_TURN -> mark
        the bot's turn as finished, etc). Server-side logging still happens
        here for observability, independent of what the client does with it.
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

    async def shutdown(self):
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self.stt.close()
        await self.tts.close()