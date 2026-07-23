"""The speech-to-text provider interface.

Every STT backend (e.g. :class:`~voxkit.stt.sarvam.SarvamSTTProvider`)
implements :class:`STTProvider`. :class:`~voxkit.core.pipeline.VoxkitPipeline`
only talks to providers through this interface, so any backend that
implements it can be dropped into the pipeline unchanged.
"""

import asyncio

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator

from pydantic import BaseModel


class STTEventType(Enum):
    """Kind of :class:`STTEvent` produced by an :class:`STTProvider`."""

    SPEECH_START = auto()
    """Voice activity detection fired: the user started talking. Used for barge-in."""

    SPEECH_END = auto()
    """Voice activity detection fired: the user stopped talking (utterance boundary)."""

    PARTIAL_TRANSCRIPT = auto()
    """An interim transcript that may still change. Not yet safe to hand to the LLM."""

    FINAL_TRANSCRIPT = auto()
    """A finalized transcript. ``STTEvent.text`` holds the recognized text; safe to hand to the LLM."""

    STREAM_CLOSED = auto()
    """The STT stream died, whether from an error or a clean close. The provider is done producing events."""


@dataclass
class STTEvent:
    """A single event emitted by an :class:`STTProvider` on its output queue.

    Args:
        type: What kind of event this is.
        text: The recognized text. Only set for :attr:`STTEventType.FINAL_TRANSCRIPT`
            and :attr:`STTEventType.PARTIAL_TRANSCRIPT`.
    """

    type: STTEventType
    text: str | None = None


class STTProvider(ABC):
    """Base class for streaming speech-to-text backends.

    Implementations wrap a vendor SDK/websocket and translate its messages
    into :class:`STTEvent` instances pushed onto :attr:`output`. The typical
    lifecycle, as driven by :class:`~voxkit.core.pipeline.VoxkitPipeline`, is::

        await provider.connect()
        # concurrently:
        await provider.send(audio_stream)   # producer: mic audio -> provider
        await provider.receive()            # consumer: provider -> output queue
        ...
        await provider.close()
    """

    def __init__(self) -> None:
        super().__init__()
        self.output: asyncio.Queue[STTEvent] = asyncio.Queue()
        """Queue of :class:`STTEvent` produced by :meth:`receive`. Consumed via :meth:`get_output_queue`."""

    @abstractmethod
    async def send(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream raw audio chunks from ``audio_stream`` to the STT backend.

        Runs concurrently with :meth:`receive` for the lifetime of the
        connection. Must return (not raise) on ``asyncio.CancelledError`` from
        normal pipeline shutdown, and should push
        :attr:`STTEventType.STREAM_CLOSED` onto :attr:`output` if sending fails
        unexpectedly.

        Args:
            audio_stream: An async iterator yielding raw audio byte chunks
                (encoding/sample rate as configured on the provider's options).
        """
        ...

    @abstractmethod
    async def receive(self) -> AsyncIterator[str]:
        """Read messages from the STT backend and push :class:`STTEvent` onto :attr:`output`.

        Runs concurrently with :meth:`send` for the lifetime of the
        connection. Must keep running until the stream closes, pushing
        :attr:`STTEventType.STREAM_CLOSED` when it does (whether from a clean
        close or an error).
        """
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Open the connection to the STT backend. Must be called before :meth:`send`/:meth:`receive`."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Tear down the connection and release any underlying resources. Safe to call more than once."""
        ...

    def get_output_queue(self) -> asyncio.Queue[STTEvent]:
        """Return the queue that :class:`STTEvent` instances are pushed onto."""
        return self.output


class STTOptions(BaseModel):
    """Base class for provider-specific STT configuration.

    Each provider defines its own subclass (e.g.
    :class:`~voxkit.stt.sarvam.SarvamSTTOptions`) declaring the fields it
    needs (API key, model, sample rate, ...).
    """

    pass
