import asyncio
import base64
import logging

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse, ErrorResponse

from voxkit.llm import LLMEvent, LLMEventType
from voxkit.tts import TTSOptions, TTSProvider, TTSEvent, TTSEventType

logger = logging.getLogger(__name__)


class SarvamTTSOptions(TTSOptions):
    api_key: str
    model: str
    target_language_code: str
    speaker: str
    send_completion_event: bool = True
    output_audio_codec: str = "linear16"
    speech_sample_rate: int = 24000


class SarvamTTSProvider(TTSProvider):
    def __init__(self, options: SarvamTTSOptions):
        super().__init__()

        self.options = options

        self.client = AsyncSarvamAI(api_subscription_key=options.api_key)
        self.ws = None
        self._ctx = None

        self._tasks: list[asyncio.Task] = []
        self._closed = False

    async def connect(self):
        self._ctx = self.client.text_to_speech_streaming.connect(
            model=self.options.model,
            send_completion_event=self.options.send_completion_event,
        )
        self.ws = await self._ctx.__aenter__()
        await self.ws.configure(
            target_language_code=self.options.target_language_code,
            speaker=self.options.speaker,
            output_audio_codec=self.options.output_audio_codec,
            speech_sample_rate=self.options.speech_sample_rate,
        )

    def synthesize(self):
        """Spins up the internal send/receive loops. Call after connect()."""
        self._tasks.append(asyncio.create_task(self._send()))
        self._tasks.append(asyncio.create_task(self._receive_with_reconnect()))

    async def _reconnect(self):
        """
        Closes the current socket and opens a fresh one. Only the send-side
        connection reference is swapped here - the running receive loop
        (bound to the old socket) is expected to end on its own once that
        socket closes, and _receive_with_reconnect() below is what notices that
        and restarts it against whatever connection is current.
        """
        if self._ctx is not None:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception:
                logger.exception("SarvamTTSProvider: error closing socket during reconnect")
        await self.connect()

    async def _send(self):
        if not self.ws:
            raise RuntimeError("SarvamTTSProvider: _send started before connect()")

        while True:
            event = await self.input.get()
            try:
                if event.type == LLMEventType.SENTENCE:
                    # Await directly - do NOT create_task this. convert() calls
                    # must stay strictly ordered on the socket; fire-and-forget
                    # tasks can interleave and send sentences out of order.
                    await self.ws.convert(event.text)

                elif event.type == LLMEventType.END_OF_TURN:
                    await self.ws.flush()

                elif event.type == LLMEventType.INTERRUPT:
                    logger.info("SarvamTTSProvider: interrupt received, closing and reconnecting")
                    await self._reconnect()

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("SarvamTTSProvider: send loop failed")

    async def _receive(self):
        """
        One pass over the currently-live socket. Ends when that socket
        closes (either due to _reconnect() above, or an unexpected drop).
        """
        async for message in self.ws:
            if isinstance(message, AudioOutput):
                audio_b64 = message.data.audio  # Base64 encoded audio bytes
                await self.output.put(TTSEvent(TTSEventType.AUDIO, audio_b64))

            elif isinstance(message, EventResponse):
                logger.debug(f"Received completion event: {message.data.event_type}")
                if message.data.event_type == "final":
                    await self.output.put(TTSEvent(TTSEventType.END_OF_TURN))

            elif isinstance(message, ErrorResponse):
                logger.error(f"SarvamTTSProvider received error response: {message.data.message}")

    async def _receive_with_reconnect(self):
        """
        Wraps _receive() in an outer retry loop. When _reconnect() (above)
        swaps self.ws for a new connection, the _receive() pass currently
        running is still bound to the old socket object and simply ends once
        it closes -- it does not follow the swap. This supervisor is what
        notices that and calls _receive() again, which reads self.ws
        fresh each time, so it naturally attaches to whatever connection is
        current now. This is what lets reconnect-on-interrupt stay entirely
        internal to this provider.
        """
        while not self._closed:
            try:
                await self._receive()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("SarvamTTSProvider: receive loop failed unexpectedly")
                await self.output.put(TTSEvent(TTSEventType.STREAM_CLOSED))
                if self._closed:
                    return
                # Unexpected drop, not a controlled reconnect - re-establish
                # the connection ourselves before looping, since nothing else
                # triggered a reconnect in this case.
                try:
                    await self._reconnect()
                except Exception:
                    logger.exception("SarvamTTSProvider: reconnect after failure also failed")
                    return

    async def close(self):
        self._closed = True
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
            self.ws = None