from enum import Enum, auto

class TTSEventType(Enum):
    SENTENCE = auto()      # A sentence of text ready to synthesize - text carries the content
    END_OF_TURN = auto()   # Agent finished normally - nothing more coming right now, no action needed
    INTERRUPT = auto()     # Interrupt - stop whatever is currently synthesizing/playing, right now

class TTSEvent:
    type: TTSEventType
    text: str | None = None