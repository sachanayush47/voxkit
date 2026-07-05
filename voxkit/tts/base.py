from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import AsyncIterator
import asyncio
from pydantic import BaseModel

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