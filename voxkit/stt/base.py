import asyncio

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator

from pydantic import BaseModel


class STTEventType(Enum):
    SPEECH_START = auto()       # VAD detected user started talking -- use for barge-in
    SPEECH_END = auto()         # VAD detected user stopped talking -- utterance boundary
    PARTIAL_TRANSCRIPT = auto() # Interim result, not yet finalized
    FINAL_TRANSCRIPT = auto()   # Finalized result, safe to hand to the LLM
    STREAM_CLOSED = auto()      # The STT stream died (error or clean close) -- producer done forever


@dataclass
class STTEvent:
    type: STTEventType
    text: str | None = None


class STTProvider(ABC):
    queue: asyncio.Queue[STTEvent]

    @abstractmethod
    async def send(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]: ...

    @abstractmethod
    async def receive(self) -> AsyncIterator[str]: ...

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class Options(BaseModel):
    pass

