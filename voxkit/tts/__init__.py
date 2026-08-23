"""Text-to-speech providers and the :class:`~voxkit.tts.base.TTSProvider` interface."""

from voxkit.tts.base import TTSEvent, TTSEventType, TTSOptions, TTSProvider
from voxkit.tts.sarvam import (
    SarvamTTSAudioBitrate,
    SarvamTTSAudioCodec,
    SarvamTTSLanguageCode,
    SarvamTTSModel,
    SarvamTTSOptions,
    SarvamTTSProvider,
    SarvamTTSSampleRate,
    SarvamTTSSpeaker,
)

__all__ = [
    "SarvamTTSAudioBitrate",
    "SarvamTTSAudioCodec",
    "SarvamTTSLanguageCode",
    "SarvamTTSModel",
    "SarvamTTSOptions",
    "SarvamTTSProvider",
    "SarvamTTSSampleRate",
    "SarvamTTSSpeaker",
    "TTSEvent",
    "TTSEventType",
    "TTSOptions",
    "TTSProvider",
]
