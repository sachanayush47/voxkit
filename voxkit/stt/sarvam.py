import asyncio
import base64
import logging

from voxkit.stt import STTOptions, STTProvider, STTEventType, STTEvent
from sarvamai import AsyncSarvamAI
from typing import AsyncIterator


logger = logging.getLogger(__name__)

class SarvamSTTOptions(STTOptions):
    api_key: str
    model: str
    mode: str
    language_code: str | None = None
    high_vad_sensitivity: bool = True
    vad_signals: bool = True
    encoding: str = "audio/wav"
    input_audio_codec: str = "pcm_s16le"
    sample_rate: int = 16000


class SarvamSTTProvider(STTProvider):
    def __init__(self, options: SarvamSTTOptions):
        super().__init__()
        self.options = options
        self.client = AsyncSarvamAI(api_subscription_key=options.api_key)
        self.ws = None

        self._ctx = None

    async def connect(self):
        self._ctx = self.client.speech_to_text_streaming.connect(
            language_code=self.options.language_code,
            model=self.options.model,
            mode=self.options.mode,
            sample_rate=self.options.sample_rate,
            input_audio_codec=self.options.input_audio_codec,
            high_vad_sensitivity=self.options.high_vad_sensitivity,
            vad_signals=self.options.vad_signals,
            api_subscription_key=self.options.api_key,
        )
        self.ws = await self._ctx.__aenter__()

    async def send(self, audio_stream: AsyncIterator[bytes]):
        if not self.ws:
            raise RuntimeError("SarvamSTTProvider.send() called before connect()")
        
        try:
            async for chunk in audio_stream:
                await self.ws.transcribe(
                    audio=base64.b64encode(chunk).decode("utf-8"),
                    encoding=self.options.encoding,
                    sample_rate=self.options.sample_rate
                )
        except asyncio.CancelledError:
            raise  # Normal shutdown path - don't report as a stream failure
        except Exception:
            logger.exception("SarvamSTTProvider: audio send failed")
            await self.output.put(STTEvent(STTEventType.STREAM_CLOSED))

    async def receive(self):
        if not self.ws:
            await self.connect()

        try:
            async for message in self.ws:
                if message.type == "events":
                    signal = message.data.signal_type
                    logger.info(f"Voice activity: {signal}")
                    if signal == "START_SPEECH":
                        await self.output.put(STTEvent(STTEventType.SPEECH_START))
                    elif signal == "END_SPEECH":
                        await self.output.put(STTEvent(STTEventType.SPEECH_END))
                    else:
                        logger.warning(f"Unknown VAD signal_type: {signal}")

                elif message.type == "data":
                    # No partial variant shown in Sarvam's docs or sample -- always final.
                    logger.debug(f"Transcript: {message.data.transcript}")
                    await self.output.put(STTEvent(STTEventType.FINAL_TRANSCRIPT, message.data.transcript))

                else:
                    logger.warning(f"Unknown message type: {message.type}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("SarvamSTTProvider: receive loop failed")
            await self.output.put(STTEvent(STTEventType.STREAM_CLOSED))

    async def close(self):
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
            self.ws = None