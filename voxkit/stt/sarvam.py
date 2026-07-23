"""Sarvam AI speech-to-text provider.

Wraps Sarvam's streaming speech-to-text websocket
(``client.speech_to_text_streaming``) as an :class:`~voxkit.stt.base.STTProvider`.
Requires the ``sarvamai`` package and a Sarvam API subscription key.
"""

import asyncio
import base64
import logging

from voxkit.stt import STTOptions, STTProvider, STTEventType, STTEvent
from sarvamai import AsyncSarvamAI
from typing import AsyncIterator


logger = logging.getLogger(__name__)


class SarvamSTTOptions(STTOptions):
    """Configuration for :class:`SarvamSTTProvider`.

    Attributes:
        api_key: Sarvam API subscription key.
        model: Sarvam STT model name, e.g. ``"saaras:v3"``.
        mode: Sarvam streaming mode, e.g. ``"transcribe"``.
        language_code: BCP-47 language code (e.g. ``"en-IN"``), or ``None`` to
            let Sarvam auto-detect the language.
        high_vad_sensitivity: Whether to use Sarvam's high-sensitivity voice
            activity detection.
        vad_signals: Whether Sarvam should emit ``START_SPEECH``/``END_SPEECH``
            voice-activity events on the stream (mapped to
            :attr:`~voxkit.stt.base.STTEventType.SPEECH_START` /
            :attr:`~voxkit.stt.base.STTEventType.SPEECH_END`).
        encoding: MIME-style encoding of the audio chunks passed to
            :meth:`SarvamSTTProvider.send`, e.g. ``"audio/wav"``.
        input_audio_codec: Raw sample codec of the audio, e.g. ``"pcm_s16le"``.
        sample_rate: Audio sample rate in Hz, e.g. ``16000``.
    """

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
    """Streams microphone audio to Sarvam and emits :class:`~voxkit.stt.base.STTEvent` in response.

    Example:
        >>> options = SarvamSTTOptions(api_key="...", model="saaras:v3", mode="transcribe")
        >>> stt = SarvamSTTProvider(options)
        >>> await stt.connect()
        >>> # concurrently: await stt.send(audio_stream) and await stt.receive()
    """

    def __init__(self, options: SarvamSTTOptions) -> None:
        """Create the provider. Call :meth:`connect` before using it.

        Args:
            options: Sarvam-specific configuration.
        """
        super().__init__()
        self.options = options
        self.client = AsyncSarvamAI(api_subscription_key=options.api_key)
        self.ws = None
        """The live streaming socket, set by :meth:`connect`. ``None`` until then."""

        self._ctx = None

    async def connect(self) -> None:
        """Open the Sarvam speech-to-text streaming websocket."""
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

    async def send(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Forward audio chunks from ``audio_stream`` to Sarvam over the open socket.

        Must be called after :meth:`connect`. On failure, pushes
        :attr:`~voxkit.stt.base.STTEventType.STREAM_CLOSED` onto :attr:`output`
        instead of raising.

        Args:
            audio_stream: An async iterator yielding raw audio byte chunks
                matching :attr:`SarvamSTTOptions.encoding` and
                :attr:`SarvamSTTOptions.sample_rate`.

        Raises:
            RuntimeError: If called before :meth:`connect`.
        """
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

    async def receive(self) -> None:
        """Read messages from Sarvam and push translated :class:`~voxkit.stt.base.STTEvent` onto :attr:`output`.

        Calls :meth:`connect` itself if the socket isn't open yet. Runs until
        the socket closes or errors, at which point
        :attr:`~voxkit.stt.base.STTEventType.STREAM_CLOSED` is pushed onto
        :attr:`output`.
        """
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

    async def close(self) -> None:
        """Close the Sarvam websocket, if open. Safe to call more than once."""
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
            self.ws = None
