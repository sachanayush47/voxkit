"""Sarvam AI text-to-speech provider.

Wraps Sarvam's streaming text-to-speech websocket
(``client.text_to_speech_streaming``) as a :class:`~voxkit.tts.base.TTSProvider`.
Requires the ``sarvamai`` package and a Sarvam API subscription key.

Sarvam's TTS socket has no server-side "cancel" message, so barge-in is
implemented client-side here by closing the socket and opening a fresh one
(see :meth:`SarvamTTSProvider._reconnect`) whenever an
:attr:`~voxkit.llm.base.LLMEventType.INTERRUPT` arrives.
"""

import asyncio
import logging
from typing import Any, Literal

from pydantic import Field
from sarvamai import AsyncSarvamAI, AudioOutput, ErrorResponse, EventResponse
from sarvamai.types import ConfigureConnection, ConfigureConnectionData

from voxkit.llm import LLMEvent, LLMEventType
from voxkit.tts import TTSEvent, TTSEventType, TTSOptions, TTSProvider

logger = logging.getLogger(__name__)

_RECONNECT_ATTEMPTS = 5
_RECONNECT_DELAY = 1.0

SarvamTTSModel = Literal["bulbul:v2", "bulbul:v3"]
"""Sarvam streaming TTS models. ``bulbul:v2`` supports pitch/loudness; ``bulbul:v3`` supports temperature."""

SarvamTTSLanguageCode = Literal[
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
]
"""BCP-47 codes Sarvam can synthesize in."""

SarvamTTSSpeaker = Literal[
    # bulbul:v2
    "anushka",
    "abhilash",
    "manisha",
    "vidya",
    "arya",
    "karun",
    "hitesh",
    # bulbul:v3
    "aditya",
    "ritu",
    "priya",
    "neha",
    "rahul",
    "pooja",
    "rohan",
    "simran",
    "kavya",
    "amit",
    "dev",
    "ishita",
    "shreya",
    "ratan",
    "varun",
    "manan",
    "sumit",
    "roopa",
    "kabir",
    "aayan",
    "shubh",
    "ashutosh",
    "advait",
    "amelia",
    "sophia",
]
"""Sarvam voices. A speaker only works with the model version it belongs to (see the groups above)."""

SarvamTTSAudioCodec = Literal["linear16", "mulaw", "alaw", "opus", "flac", "aac", "wav", "mp3"]
"""Output audio codecs. ``linear16`` is raw PCM, which is what most playback paths want."""

SarvamTTSAudioBitrate = Literal["32k", "64k", "96k", "128k", "192k"]
"""Output bitrate. Only meaningful for the compressed codecs (``mp3``, ``aac``, ``opus``)."""

SarvamTTSSampleRate = Literal[8000, 16000, 22050, 24000]
"""Output sample rates Sarvam supports. Defaults to 22050 on ``bulbul:v2`` and 24000 on ``bulbul:v3``."""


class SarvamTTSOptions(TTSOptions):
    """Configuration for :class:`SarvamTTSProvider`.

    Defaults mirror the ``sarvamai`` SDK's own defaults, so constructing this
    with only the required fields behaves the same as calling the SDK's
    ``configure()`` helper with no extra arguments. The two exceptions are
    :attr:`output_audio_codec` and :attr:`speech_sample_rate`, which default to
    raw 24 kHz PCM (what voxkit's playback path wants) rather than the SDK's
    22.05 kHz MP3.

    Several knobs are model-specific, and Sarvam silently ignores rather than
    rejects the ones that don't apply: :attr:`pitch` and :attr:`loudness` are
    ignored by ``bulbul:v3``, while :attr:`temperature` and :attr:`dict_id` are
    ignored by ``bulbul:v2``. A field left ``None`` is omitted from the config
    message entirely, leaving Sarvam's server-side default in effect.

    Attributes:
        api_key: Sarvam API subscription key.
        model: Sarvam TTS model name, e.g. ``"bulbul:v3"``.
        target_language_code: BCP-47 language code to synthesize in, e.g. ``"en-IN"``.
        speaker: Sarvam speaker/voice name, e.g. ``"priya"``. Must belong to
            the chosen :attr:`model`.
        send_completion_event: Whether Sarvam should send an ``EventResponse``
            with ``event_type == "final"`` when synthesis for a flushed
            request completes (mapped to
            :attr:`~voxkit.tts.base.TTSEventType.END_OF_TURN`).
        output_audio_codec: Output audio codec. Defaults to ``"linear16"``
            (raw PCM); the SDK's own default is ``"mp3"``.
        output_audio_bitrate: Output bitrate for compressed codecs. Only
            meaningful for ``mp3``/``aac``/``opus``.
        speech_sample_rate: Output audio sample rate in Hz. Defaults to
            ``24000``, which is ``bulbul:v3``'s native rate; the SDK's own
            default is ``22050``.
        pace: Speech speed. ``1.0`` is normal; the usable range is 0.3-3.0 on
            ``bulbul:v2`` and 0.5-2.0 on ``bulbul:v3``.
        pitch: Voice pitch, roughly -0.75 to 0.75, ``0.0`` being the voice's
            natural pitch. ``bulbul:v2`` only.
        loudness: Output loudness, roughly 0.3 to 3.0, ``1.0`` being normal.
            ``bulbul:v2`` only.
        temperature: Synthesis randomness, roughly 0.01 to 1.0. Lower is more
            deterministic and consistent across turns. ``bulbul:v3`` only.
        enable_preprocessing: Whether to normalize English words and numeric
            entities (numbers, dates, ...) before synthesis. Worth turning on
            for mixed-language text. Always on for ``bulbul:v3``.
        dict_id: ID of a pronunciation dictionary (created via Sarvam's
            ``/text-to-speech/pronunciation-dictionary`` endpoints) to apply
            during synthesis. ``None`` means no dictionary. ``bulbul:v3`` only.
        min_buffer_size: Minimum number of buffered characters that triggers a
            flush to the model. Lower values cut first-audio latency at the
            cost of more, smaller requests.
        max_chunk_length: Maximum length Sarvam will split a sentence at.
    """

    api_key: str
    model: SarvamTTSModel
    target_language_code: SarvamTTSLanguageCode
    speaker: SarvamTTSSpeaker
    send_completion_event: bool = True

    output_audio_codec: SarvamTTSAudioCodec = "linear16"
    output_audio_bitrate: SarvamTTSAudioBitrate = "128k"
    speech_sample_rate: SarvamTTSSampleRate = 24000

    pace: float = 1.0
    pitch: float = 0.0
    loudness: float = 1.0
    temperature: float = 0.6

    enable_preprocessing: bool = False
    dict_id: str | None = None
    min_buffer_size: int = Field(default=50, ge=1)
    max_chunk_length: int = Field(default=150, ge=1)


class SarvamTTSProvider(TTSProvider):
    """Synthesizes agent sentences to audio via Sarvam's streaming TTS websocket.

    Example:
        >>> options = SarvamTTSOptions(api_key="...", model="bulbul:v3",
        ...                             target_language_code="en-IN", speaker="priya")
        >>> tts = SarvamTTSProvider(options)
        >>> await tts.connect()
        >>> tts.synthesize()
        >>> # feed sentences: await tts.get_input_queue().put(LLMEvent(LLMEventType.SENTENCE, "Hi!"))
        >>> # read audio: event = await tts.get_output_queue().get()
    """

    def __init__(self, options: SarvamTTSOptions) -> None:
        """Create the provider. Call :meth:`connect` then :meth:`synthesize` before using it.

        Args:
            options: Sarvam-specific configuration.
        """
        super().__init__()

        self.options = options

        self.client = AsyncSarvamAI(api_subscription_key=options.api_key)
        self.ws = None
        """The live streaming socket, set by :meth:`connect`. ``None`` until then."""
        self._ctx = None

        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._connected = asyncio.Event()

    def _config_message(self) -> ConfigureConnection:
        """Build the ``config`` message sent as the first frame after connecting.

        Most fields carry the SDK's own defaults, so they are always sent.
        Anything that is nonetheless ``None`` (``dict_id``, unless set) is left
        out of the message rather than sent as a null, so Sarvam's server-side
        default stays in effect.

        Returns:
            The config message to send on the socket.
        """
        options = self.options
        optional: dict[str, Any] = {
            "output_audio_bitrate": options.output_audio_bitrate,
            "pace": options.pace,
            "pitch": options.pitch,
            "loudness": options.loudness,
            "temperature": options.temperature,
            "enable_preprocessing": options.enable_preprocessing,
            "dict_id": options.dict_id,
            "min_buffer_size": options.min_buffer_size,
            "max_chunk_length": options.max_chunk_length,
        }
        return ConfigureConnection(
            data=ConfigureConnectionData(
                model=options.model,
                target_language_code=options.target_language_code,
                speaker=options.speaker,
                output_audio_codec=options.output_audio_codec,
                speech_sample_rate=options.speech_sample_rate,
                **{key: value for key, value in optional.items() if value is not None},
            )
        )

    async def connect(self) -> None:
        """Open the Sarvam text-to-speech streaming websocket and send the initial config message."""
        self._ctx = self.client.text_to_speech_streaming.connect(
            model=self.options.model,
            send_completion_event="true" if self.options.send_completion_event else "false",
        )
        self.ws = await self._ctx.__aenter__()
        # Not ws.configure(): it lacks `temperature` and injects pitch/loudness defaults bulbul:v3 rejects.
        await self.ws._send_model(self._config_message())
        self._connected.set()

    def synthesize(self) -> None:
        """Spin up the internal send/receive loops as background tasks. Call after :meth:`connect`."""
        self._tasks.append(asyncio.create_task(self._send()))
        self._tasks.append(asyncio.create_task(self._receive_with_reconnect()))

    async def _reconnect(self) -> bool:
        """Replace the current socket with a fresh one, retrying a few times.

        Returns:
            ``True`` once connected, ``False`` if every attempt failed or the
            provider was closed.
        """
        await self._close()
        for attempt in range(1, _RECONNECT_ATTEMPTS + 1):
            if self._closed:
                return False
            try:
                await self.connect()
                return True
            except Exception:
                logger.warning("SarvamTTSProvider: reconnect attempt %d failed", attempt, exc_info=True)
                await asyncio.sleep(_RECONNECT_DELAY)
        logger.error("SarvamTTSProvider: giving up after %d reconnect attempts", _RECONNECT_ATTEMPTS)
        return False

    async def _close(self) -> None:
        """Close the current socket, ignoring errors from an already-dead connection."""
        self._connected.clear()
        ctx, self._ctx, self.ws = self._ctx, None, None
        if ctx is not None:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                logger.debug("SarvamTTSProvider: error closing socket", exc_info=True)

    async def _send(self) -> None:
        """Background loop: pull :class:`~voxkit.llm.base.LLMEvent` off :attr:`input` and act on them.

        Raises:
            RuntimeError: If started before :meth:`connect`.
        """
        if not self.ws:
            raise RuntimeError("SarvamTTSProvider: _send started before connect()")

        while True:
            event: LLMEvent = await self.input.get()
            try:
                if event.type == LLMEventType.INTERRUPT:
                    # A fresh socket is the only way to drop in-flight synthesis. If ws is
                    # None, the receive loop is already reconnecting after a drop.
                    if self.ws is not None:
                        logger.debug("SarvamTTSProvider: interrupt, reconnecting to drop in-flight synthesis")
                        await self._reconnect()
                    continue

                await self._connected.wait()
                if event.type == LLMEventType.SENTENCE:
                    # Awaited, not create_task'd: sentences must stay ordered on the socket.
                    await self.ws.convert(event.text)
                elif event.type == LLMEventType.END_OF_TURN:
                    await self.ws.flush()

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("SarvamTTSProvider: failed to send %s", event.type.name, exc_info=True)

    async def _receive(self, ws: Any) -> None:
        """Read ``ws`` until it closes, translating messages into :class:`~voxkit.tts.base.TTSEvent`.

        Args:
            ws: The socket to read.
        """
        async for message in ws:
            if isinstance(message, AudioOutput):
                await self.output.put(TTSEvent(TTSEventType.AUDIO, message.data.audio))

            elif isinstance(message, EventResponse):
                logger.debug("SarvamTTSProvider: event %s", message.data.event_type)
                if message.data.event_type == "final":
                    await self.output.put(TTSEvent(TTSEventType.END_OF_TURN))

            elif isinstance(message, ErrorResponse):
                logger.error("SarvamTTSProvider: error response: %s", message.data.message)

    async def _receive_with_reconnect(self) -> None:
        """Keep reading whichever socket is current, reconnecting if it drops.

        An interrupt swaps the socket from :meth:`_send`; the read of the old
        socket then ends, and this loop simply moves on to the new one. If the
        *current* socket drops, this loop reconnects, emitting
        :attr:`~voxkit.tts.base.TTSEventType.STREAM_CLOSED` only if that fails.
        """
        while not self._closed:
            await self._connected.wait()
            ws = self.ws
            try:
                await self._receive(ws)
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.ws is ws and not self._closed:
                    logger.warning("SarvamTTSProvider: connection lost, reconnecting", exc_info=True)
            if self.ws is ws and not self._closed and not await self._reconnect():
                await self.output.put(TTSEvent(TTSEventType.STREAM_CLOSED))
                return

    async def close(self) -> None:
        """Cancel the internal send/receive tasks and close the socket. Safe to call more than once."""
        self._closed = True
        for task in self._tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._close()
