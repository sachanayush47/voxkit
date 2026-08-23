"""Sarvam AI speech-to-text provider.

Wraps Sarvam's streaming speech-to-text websocket
(``client.speech_to_text_streaming``) as an :class:`~voxkit.stt.base.STTProvider`.
Requires the ``sarvamai`` package and a Sarvam API subscription key.
"""

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import Field
from sarvamai import AsyncSarvamAI

from voxkit.stt import STTEvent, STTEventType, STTOptions, STTProvider

logger = logging.getLogger(__name__)

SarvamSTTModel = Literal["saaras:v3", "saarika:v2.5"]
"""Sarvam streaming STT models. ``saarika:v2.5`` is legacy; ``saaras:v3`` is recommended."""

SarvamSTTMode = Literal["transcribe", "translate", "verbatim", "translit", "codemix"]
"""Output style for the transcript. Only honoured by ``saaras:v3``."""

SarvamSTTLanguageCode = Literal[
    "unknown",
    "en-IN",
    "hi-IN",
    "bn-IN",
    "gu-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
]
"""BCP-47 codes Sarvam accepts. The last twelve are ``saaras:v3``-only; ``"unknown"`` auto-detects."""

SarvamSTTInputAudioCodec = Literal["wav", "pcm_s16le", "pcm_l16", "pcm_raw"]
"""Raw sample codec of the audio chunks sent to Sarvam."""


def _flag(value: bool | None) -> str | None:
    """Render a boolean the way Sarvam's websocket query parameters expect it.

    Args:
        value: The flag, or ``None`` to leave it unset.

    Returns:
        ``"true"``/``"false"``, or ``None`` if ``value`` was ``None`` (in which
        case the parameter is dropped and Sarvam's own default applies).
    """
    if value is None:
        return None
    return "true" if value else "false"


def _num(value: float | int | None) -> str | None:
    """Render a numeric websocket query parameter as a string, preserving "unset".

    Args:
        value: The number, or ``None`` to leave it unset.

    Returns:
        The stringified number, or ``None`` if ``value`` was ``None``.
    """
    if value is None:
        return None
    return str(value)


class SarvamSTTOptions(STTOptions):
    """Configuration for :class:`SarvamSTTProvider`.

    Defaults mirror the ``sarvamai`` SDK's own defaults. The SDK leaves every
    VAD tuning knob unset (there are no client-side values to mirror -- the
    numbers live server-side), so those default to ``None`` here too: they are
    not sent at all, and Sarvam's server-side default or the
    :attr:`high_vad_sensitivity` preset applies. Set one to override it.

    :attr:`vad_signals` is the one deliberate departure: it defaults to ``True``
    because :class:`~voxkit.core.pipeline.VoxkitPipeline` needs ``SPEECH_START``
    to detect barge-in, and voxkit bundles no VAD of its own.

    Attributes:
        api_key: Sarvam API subscription key.
        model: Sarvam STT model name, e.g. ``"saaras:v3"``.
        mode: Transcript style — ``"transcribe"``, ``"translate"``,
            ``"verbatim"``, ``"translit"`` or ``"codemix"``. Only applies to
            ``saaras:v3``.
        language_code: BCP-47 language code (e.g. ``"en-IN"``). Defaults to
            ``"unknown"``, which makes Sarvam auto-detect the language and
            report it back on each transcript; ``None`` is equivalent.
        encoding: MIME-style encoding of the audio chunks passed to
            :meth:`SarvamSTTProvider.send`, e.g. ``"audio/wav"``.
        input_audio_codec: Raw sample codec of the audio, e.g. ``"pcm_s16le"``.
        sample_rate: Audio sample rate in Hz. Only ``8000`` and ``16000`` are
            supported as connection-level values; ``8000`` is available *only*
            this way.
        vad_signals: Whether Sarvam should emit ``START_SPEECH``/``END_SPEECH``
            voice-activity events on the stream (mapped to
            :attr:`~voxkit.stt.base.STTEventType.SPEECH_START` /
            :attr:`~voxkit.stt.base.STTEventType.SPEECH_END`).
        flush_signal: Whether Sarvam should accept flush signals that force any
            buffered audio to be finalized (see
            :meth:`SarvamSTTProvider.flush`). ``None`` leaves it to Sarvam.
        high_vad_sensitivity: Whether to use Sarvam's high-sensitivity voice
            activity detection preset. The individual thresholds below override
            whatever this preset sets.
        positive_speech_threshold: VAD probability (0.0-1.0) above which a
            frame counts as speech.
        negative_speech_threshold: VAD probability (0.0-1.0) below which a
            frame counts as silence.
        min_speech_frames: Consecutive speech frames required to open a speech
            segment.
        first_turn_min_speech_frames: Same as :attr:`min_speech_frames`, but
            for the very first user turn only.
        negative_frames_count: Silence frames needed within
            :attr:`negative_frames_window` to close a speech segment.
        negative_frames_window: Sliding window size, in frames, that
            :attr:`negative_frames_count` is counted over.
        start_speech_volume_threshold: Volume in dB below which audio is
            treated as too quiet to be speech. Unset means no volume gate.
        interrupt_min_speech_frames: Speech frames required to register a
            barge-in — the knob to reach for when interrupts fire too eagerly
            or too reluctantly.
        pre_speech_pad_frames: Frames to prepend before the detected speech
            onset so the start of the utterance isn't clipped.
        num_initial_ignored_frames: Leading frames to discard outright at
            connection start, e.g. to drop connection-setup noise.
    """

    api_key: str
    model: SarvamSTTModel
    mode: SarvamSTTMode
    language_code: SarvamSTTLanguageCode | None = "unknown"
    encoding: str = "audio/wav"
    input_audio_codec: SarvamSTTInputAudioCodec = "pcm_s16le"
    sample_rate: int = 16000

    vad_signals: bool = True
    flush_signal: bool | None = None

    high_vad_sensitivity: bool = True
    positive_speech_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    negative_speech_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_speech_frames: int | None = Field(default=None, ge=0)
    first_turn_min_speech_frames: int | None = Field(default=None, ge=0)
    negative_frames_count: int | None = Field(default=None, ge=0)
    negative_frames_window: int | None = Field(default=None, ge=0)
    start_speech_volume_threshold: float | None = None
    interrupt_min_speech_frames: int | None = Field(default=None, ge=0)
    pre_speech_pad_frames: int | None = Field(default=None, ge=0)
    num_initial_ignored_frames: int | None = Field(default=None, ge=0)


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

    def _connect_params(self) -> dict[str, Any]:
        """Build the websocket query parameters from :attr:`options`.

        Anything left ``None`` on the options object is omitted entirely rather
        than sent as a null, so Sarvam's server-side defaults stay in effect.

        Returns:
            Keyword arguments for ``speech_to_text_streaming.connect()``.
        """
        options = self.options
        params: dict[str, Any] = {
            "model": options.model,
            "mode": options.mode,
            "sample_rate": _num(options.sample_rate),
            "input_audio_codec": options.input_audio_codec,
            "vad_signals": _flag(options.vad_signals),
            "flush_signal": _flag(options.flush_signal),
            "high_vad_sensitivity": _flag(options.high_vad_sensitivity),
            "positive_speech_threshold": _num(options.positive_speech_threshold),
            "negative_speech_threshold": _num(options.negative_speech_threshold),
            "min_speech_frames": _num(options.min_speech_frames),
            "first_turn_min_speech_frames": _num(options.first_turn_min_speech_frames),
            "negative_frames_count": _num(options.negative_frames_count),
            "negative_frames_window": _num(options.negative_frames_window),
            "start_speech_volume_threshold": _num(options.start_speech_volume_threshold),
            "interrupt_min_speech_frames": _num(options.interrupt_min_speech_frames),
            "pre_speech_pad_frames": _num(options.pre_speech_pad_frames),
            "num_initial_ignored_frames": _num(options.num_initial_ignored_frames),
            "api_subscription_key": options.api_key,
        }
        # language_code is a required keyword on connect(), so it always goes
        # through - the SDK drops it from the query string when it's None,
        # which is what makes Sarvam auto-detect the language.
        return {
            "language_code": options.language_code,
            **{key: value for key, value in params.items() if value is not None},
        }

    async def connect(self) -> None:
        """Open the Sarvam speech-to-text streaming websocket."""
        self._ctx = self.client.speech_to_text_streaming.connect(**self._connect_params())
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
                    sample_rate=self.options.sample_rate,
                )
        except asyncio.CancelledError:
            raise  # Normal shutdown path - don't report as a stream failure
        except Exception:
            logger.exception("SarvamSTTProvider: audio send failed")
            await self.output.put(STTEvent(STTEventType.STREAM_CLOSED))

    async def flush(self) -> None:
        """Ask Sarvam to finalize whatever audio it has buffered, without waiting for VAD.

        Requires :attr:`SarvamSTTOptions.flush_signal` to be enabled on the
        connection. Useful when the caller knows the utterance is over (the
        user hung up, pressed a button, ...) and doesn't want to wait for an
        ``END_SPEECH`` that may never arrive.

        Raises:
            RuntimeError: If called before :meth:`connect`.
        """
        if not self.ws:
            raise RuntimeError("SarvamSTTProvider.flush() called before connect()")

        await self.ws.flush()

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

                elif message.type == "error":
                    logger.error(
                        f"SarvamSTTProvider received error response ({message.data.code}): {message.data.error}"
                    )

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
