from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import AsyncIterator
import asyncio
from pydantic import BaseModel
from dataclasses import dataclass
from voxkit.llm import LLMEvent


class TTSProvider(ABC):
    queue: asyncio.Queue[LLMEvent]

    @abstractmethod
    async def send(self) -> None: ...

    @abstractmethod
    async def receive(self) -> AsyncIterator[str]: ...

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class TTSOptions(BaseModel):
    pass


class TTSEventType(Enum):
    AUDIO = auto()      # A sentence of text ready to synthesize - text carries the content
    END_OF_TURN = auto()   # Agent finished normally - nothing more coming right now, no action needed
    INTERRUPT = auto()     # Interrupt - stop whatever is currently synthesizing/playing, right now


@dataclass
class TTSEvent:
    type: TTSEventType
    audio: bytes
