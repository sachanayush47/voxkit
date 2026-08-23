"""Speech-to-text providers and the :class:`~voxkit.stt.base.STTProvider` interface."""

from voxkit.stt.base import STTEvent, STTEventType, STTOptions, STTProvider
from voxkit.stt.sarvam import (
    SarvamSTTInputAudioCodec,
    SarvamSTTLanguageCode,
    SarvamSTTMode,
    SarvamSTTModel,
    SarvamSTTOptions,
    SarvamSTTProvider,
)

__all__ = [
    "STTEvent",
    "STTEventType",
    "STTOptions",
    "STTProvider",
    "SarvamSTTInputAudioCodec",
    "SarvamSTTLanguageCode",
    "SarvamSTTMode",
    "SarvamSTTModel",
    "SarvamSTTOptions",
    "SarvamSTTProvider",
]
