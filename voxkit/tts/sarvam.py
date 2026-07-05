import asyncio
import base64
import logging
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
from voxkit.llm import LLMEvent, LLMEventType
from voxkit.tts import TTSOptions, TTSProvider, TTSEvent, TTSEventType

logger = logging.getLogger(__name__)

class SarvamTTSOptions(TTSOptions):
    api_key: str
    model: str
    target_language_code: str
    speaker: str
    send_completion_event: bool = True


class SarvamTTSProvider(TTSProvider):
    def __init__(self, options: SarvamTTSOptions, queue: asyncio.Queue[LLMEvent]):
        self.options = options
        self.queue = queue
        self.client = AsyncSarvamAI(api_subscription_key=options.api_key)
        self.ws = None

    async def connect(self):
        self._ctx = self.client.text_to_speech_streaming.connect(
            model=self.options.model,
            send_completion_event=self.options.send_completion_event
        )
        self.ws = await self._ctx.__aenter__()
        self.ws.configure(target_language_code=self.options.target_language_code, speaker=self.options.speaker)

    async def send(self):
        if not self.ws:
            raise RuntimeError("SarvamTTSProvider.send() called before connect()")
        
        while True:
            event = await self.queue.get()
            if event.type == LLMEventType.SENTENCE or event.type == LLMEventType.END_OF_TURN:
                asyncio.create_task(self.ws.convert(event.text))
                if event.type == LLMEventType.END_OF_TURN:
                    await self.ws.flush()
            elif event.type == LLMEventType.INTERRUPT:
                self.ws.close()


    async def receive(self):
        if not self.ws:
            raise RuntimeError("SarvamTTSProvider.receive() called before connect()")
        
        async for message in self.ws:
            if isinstance(message, AudioOutput):
                # Audio is in b64 encoded
                yield TTSEvent(TTSEventType.AUDIO, message.data.audio)
            elif isinstance(message, EventResponse):
                logger.debug(f"Received completion event: {message.data.event_type}")
                if message.data.event_type == "final":
                    yield TTSEvent(TTSEventType.END_OF_TURN)