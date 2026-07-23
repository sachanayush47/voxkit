"""Speech-to-text providers and the :class:`~voxkit.stt.base.STTProvider` interface."""

from voxkit.stt.base import STTOptions, STTProvider, STTEventType, STTEvent
from voxkit.stt.sarvam import SarvamSTTOptions, SarvamSTTProvider

__all__ = ["STTOptions", "SarvamSTTOptions", "STTProvider", "SarvamSTTProvider", "STTEventType", "STTEvent"]
