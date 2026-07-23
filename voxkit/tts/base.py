"""The text-to-speech provider interface.

Every TTS backend (e.g. :class:`~voxkit.tts.sarvam.SarvamTTSProvider`)
implements :class:`TTSProvider`. :class:`~voxkit.core.pipeline.VoxkitPipeline`
only talks to providers through this interface, so any backend that
implements it can be dropped into the pipeline unchanged.
"""

import asyncio

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator

from pydantic import BaseModel

from voxkit.llm import LLMEvent


class TTSEventType(Enum):
    """Kind of :class:`TTSEvent` produced by a :class:`TTSProvider`."""

    AUDIO = auto()
    """A chunk of synthesized audio is ready. ``TTSEvent.audio`` carries the (base64-encoded) bytes."""

    END_OF_TURN = auto()
    """Synthesis for the current turn finished normally; nothing more is coming right now."""

    INTERRUPT = auto()
    """Barge-in: stop playing whatever audio has already been sent to the client, right now."""

    STREAM_CLOSED = auto()
    """The TTS stream died, whether from an error, a clean close, or an interrupt-triggered reconnect."""


@dataclass
class TTSEvent:
    """A single event emitted by a :class:`TTSProvider` on its output queue.

    Args:
        type: What kind of event this is.
        audio: Base64-encoded audio bytes. Only set for :attr:`TTSEventType.AUDIO`.
    """

    type: TTSEventType
    audio: str | None = None


class TTSProvider(ABC):
    """Base class for streaming text-to-speech backends.

    Implementations wrap a vendor SDK/websocket, consume :class:`~voxkit.llm.base.LLMEvent`
    (sentences from the agent) off :attr:`input`, and push
    synthesized-audio :class:`TTSEvent` onto :attr:`output`. The typical
    lifecycle, as driven by :class:`~voxkit.core.pipeline.VoxkitPipeline`, is::

        await provider.connect()
        provider.synthesize()   # spins up its own internal send/receive tasks
        # meanwhile, LLMEvent instances are pushed onto provider.get_input_queue()
        # and TTSEvent instances are read off provider.get_output_queue()
        ...
        await provider.close()
    """

    def __init__(self) -> None:
        super().__init__()
        self.input: asyncio.Queue[LLMEvent] = asyncio.Queue()
        """Queue of :class:`~voxkit.llm.base.LLMEvent` to synthesize. Fed via :meth:`get_input_queue`."""
        self.output: asyncio.Queue[TTSEvent] = asyncio.Queue()
        """Queue of :class:`TTSEvent` produced by synthesis. Consumed via :meth:`get_output_queue`."""

    @abstractmethod
    def synthesize(self) -> None:
        """Start the provider's internal synthesis loop(s). Call after :meth:`connect`.

        Unlike :meth:`~voxkit.stt.base.STTProvider.send`/``receive``, this is
        not awaited directly by the caller -- implementations typically
        schedule their own background tasks here that read from :attr:`input`
        and write to :attr:`output` for the lifetime of the provider.
        """
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Open the connection to the TTS backend. Must be called before :meth:`synthesize`."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Tear down the connection and release any underlying resources. Safe to call more than once."""
        ...

    def get_input_queue(self) -> asyncio.Queue[LLMEvent]:
        """Return the queue that agent-generated :class:`~voxkit.llm.base.LLMEvent` should be pushed onto."""
        return self.input

    def get_output_queue(self) -> asyncio.Queue[TTSEvent]:
        """Return the queue that synthesized :class:`TTSEvent` are pushed onto."""
        return self.output


class TTSOptions(BaseModel):
    """Base class for provider-specific TTS configuration.

    Each provider defines its own subclass (e.g.
    :class:`~voxkit.tts.sarvam.SarvamTTSOptions`) declaring the fields it
    needs (API key, model, voice, sample rate, ...).
    """

    pass
