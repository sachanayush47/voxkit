from dataclasses import dataclass
from enum import Enum, auto


class LLMEventType(Enum):
    SENTENCE = auto()      # A sentence of text ready to synthesize - text carries the content
    END_OF_TURN = auto()   # Agent finished normally - nothing more coming right now, no action needed
    INTERRUPT = auto()     # Interrupt - stop whatever is currently synthesizing/playing, right now


@dataclass
class LLMEvent:
    type: LLMEventType
    text: str | None = None
