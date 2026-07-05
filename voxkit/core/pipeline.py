import asyncio
import logging
import re

from typing import AsyncIterator, Optional

from langgraph.graph.state import CompiledStateGraph

from voxkit.stt import STTEvent, STTEventType, STTProvider
from voxkit.llm import LLMEvent, LLMEventType

logger = logging.getLogger(__name__)

SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\s]|[,;][\s]")

class VoxkitPipeline:
    def __init__(self, stt: STTProvider, agent: CompiledStateGraph, thread_id: str = "default"):
        self.stt: STTProvider = stt
        self.agent: CompiledStateGraph = agent
        self.thread_id = thread_id  # Passed to the agent on every turn so checkpointed memory persists

        self.stt_output_queue: asyncio.Queue[STTEvent] = self.stt.queue
        self.llm_output_queue: asyncio.Queue[LLMEvent] = asyncio.Queue()

        self._background_tasks: list[asyncio.Task] = []
        self._turn_task: Optional[asyncio.Task] = None
        self._cancel_event: asyncio.Event = asyncio.Event()

    async def run(self, audio_stream: AsyncIterator[bytes]):
        await self.stt.connect()

        self._background_tasks.append(asyncio.create_task(self.stt.send(audio_stream)))
        self._background_tasks.append(asyncio.create_task(self.stt.receive()))

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
        if self._turn_task and not self._turn_task.done():
            logger.info("Interrupt detected, cancelling in-flight agent turn")
            self._cancel_event.set()
            self._turn_task.cancel()

            # Drain anything already queued for TTS - it's stale now.
            while not self.llm_output_queue.empty():
                try:
                    self.llm_output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Wake up a TTS consumer that might be blocked on queue.get() so it
            # actively stops whatever it's synthesizing/playing right now,
            # rather than just idling until the next sentence.
            await self.__signal(LLMEvent(LLMEventType.INTERRUPT))

    async def __handle_user_turn(self, text: str):
        # Fresh cancel event per turn - the previous one (if any) stays set for
        # the task that's unwinding; this one is what the new turn checks.
        self._cancel_event = asyncio.Event()
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
            # Sentinel: tells the TTS consumer this turn's sentences are complete.
            # Distinct from TTSEventType.INTERRUPT -- this means "nothing more is
            # coming for now," not "stop what's currently playing." If an
            # interrupt already fired for this same turn, the consumer will see
            # INTERRUPT followed by END_OF_TURN back to back -- harmless, since
            # both are no-ops for a consumer that isn't currently mid-sentence.
            await self.__signal(LLMEvent(LLMEventType.END_OF_TURN))

    async def __signal(self, event: LLMEvent):
        """
        Non-blocking push for control events (END_OF_TURN / INTERRUPT). These
        aren't real content and shouldn't be subject to the same backpressure
        as sentences -- if the queue is bounded (e.g. maxsize=2) and full, a
        plain `put()` would suspend waiting for space, which defeats the
        purpose of a signal that needs to land immediately. Evict the oldest
        item instead of waiting.
        """
        while True:
            try:
                self.llm_output_queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                try:
                    self.llm_output_queue.get_nowait()
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

    async def shutdown(self):
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self.stt.close()