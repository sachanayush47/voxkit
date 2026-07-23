"""Event types exchanged between the agent-turn runner and the TTS provider.

voxkit does not wrap the LLM/agent itself behind a provider interface --
any LangGraph ``CompiledStateGraph`` is accepted directly by
:class:`~voxkit.core.pipeline.VoxkitPipeline`. This module only defines the
small message type used to hand agent output (sentence-by-sentence) to a
:class:`~voxkit.tts.base.TTSProvider`.
"""

from dataclasses import dataclass
from enum import Enum, auto


class LLMEventType(Enum):
    """Kind of :class:`LLMEvent` flowing from the agent-turn runner to TTS."""

    SENTENCE = auto()
    """A complete sentence of agent output ready to synthesize. ``LLMEvent.text`` carries the content."""

    END_OF_TURN = auto()
    """The agent finished generating normally. Nothing more is coming for this turn; no action needed."""

    INTERRUPT = auto()
    """The user barged in. Stop whatever is currently synthesizing/playing, right now."""


@dataclass
class LLMEvent:
    """A single event in the agent-output stream consumed by a :class:`~voxkit.tts.base.TTSProvider`.

    Args:
        type: What kind of event this is.
        text: The sentence text. Only set when ``type`` is :attr:`LLMEventType.SENTENCE`.
    """

    type: LLMEventType
    text: str | None = None
