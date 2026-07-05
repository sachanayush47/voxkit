import asyncio

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator

from pydantic import BaseModel

from voxkit.llm import LLMEvent


class TTSEventType(Enum):
    AUDIO = auto()         # A sentence of text ready to synthesize - text carries the content
    END_OF_TURN = auto()   # Agent finished normally - nothing more coming right now, no action needed
    INTERRUPT = auto()     # Interrupt - stop whatever is currently synthesizing/playing, right now
    STREAM_CLOSED = auto() # The TTS stream died (error or clean close or interrupt)


@dataclass
class TTSEvent:
    type: TTSEventType
    audio: str | None = None


class TTSProvider(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.input: asyncio.Queue[LLMEvent] = asyncio.Queue()
        self.output: asyncio.Queue[TTSEvent] = asyncio.Queue()

    @abstractmethod
    def synthesize(self) -> None: ...

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    def get_input_queue(self) -> asyncio.Queue[LLMEvent]:
        return self.input

    def get_output_queue(self) -> asyncio.Queue[TTSEvent]:
        return self.output


class TTSOptions(BaseModel):
    pass

