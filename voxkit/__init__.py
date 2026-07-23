"""voxkit: a thin, event-driven library for building real-time voice agents.

voxkit wires a streaming speech-to-text provider, any LangGraph agent, and a
streaming text-to-speech provider into one turn-taking pipeline
(:class:`~voxkit.core.pipeline.VoxkitPipeline`), handling audio-in/audio-out
plumbing and barge-in (interrupt) so you don't have to.

It is intentionally thin: no frame-based transport bus, no call/room
management, no bundled VAD -- STT/TTS providers report voice activity
themselves. If you already know LangChain/LangGraph, you already know how to
build the "brain" of a voxkit voice agent.

Quick start:
    >>> from voxkit import VoxkitPipeline
    >>> from voxkit.stt import SarvamSTTOptions, SarvamSTTProvider
    >>> from voxkit.tts import SarvamTTSOptions, SarvamTTSProvider
    >>> from langchain.agents import create_agent
    >>>
    >>> stt = SarvamSTTProvider(SarvamSTTOptions(api_key="...", model="saaras:v3", mode="transcribe"))
    >>> tts = SarvamTTSProvider(SarvamTTSOptions(api_key="...", model="bulbul:v3",
    ...                                          target_language_code="en-IN", speaker="priya"))
    >>> agent = create_agent(model=..., tools=[])
    >>>
    >>> async def handle_tts_event(event):
    ...     ...  # play event.audio, handle INTERRUPT, etc.
    >>>
    >>> pipeline = VoxkitPipeline(stt, tts, agent, handle_tts_event)
    >>> await pipeline.run(microphone_stream())

See the ``stt``, ``tts``, and ``llm`` subpackages for the provider
interfaces and event types, and :class:`~voxkit.core.pipeline.VoxkitPipeline`
for the orchestrator itself.
"""

from voxkit.core import VoxkitPipeline
from voxkit.llm import LLMEvent, LLMEventType
from voxkit.stt import (
    SarvamSTTOptions,
    SarvamSTTProvider,
    STTEvent,
    STTEventType,
    STTOptions,
    STTProvider,
)
from voxkit.tts import (
    SarvamTTSOptions,
    SarvamTTSProvider,
    TTSEvent,
    TTSEventType,
    TTSOptions,
    TTSProvider,
)

__version__ = "0.0.1"

__all__ = [
    "VoxkitPipeline",
    "LLMEvent",
    "LLMEventType",
    "STTEvent",
    "STTEventType",
    "STTOptions",
    "STTProvider",
    "SarvamSTTOptions",
    "SarvamSTTProvider",
    "TTSEvent",
    "TTSEventType",
    "TTSOptions",
    "TTSProvider",
    "SarvamTTSOptions",
    "SarvamTTSProvider",
]
