from abc import ABC, abstractmethod
from typing import AsyncIterator


class STTProvider(ABC):

    @abstractmethod
    async def stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]: ...
