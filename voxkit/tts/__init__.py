"""Text-to-speech providers and the :class:`~voxkit.tts.base.TTSProvider` interface."""

from voxkit.tts.base import TTSOptions, TTSProvider, TTSEventType, TTSEvent
from voxkit.tts.sarvam import SarvamTTSOptions, SarvamTTSProvider

__all__ = ["TTSOptions", "TTSProvider", "TTSEventType", "TTSEvent", "SarvamTTSOptions", "SarvamTTSProvider"]
