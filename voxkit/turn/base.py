"""The :class:`EndOfTurnDetector` interface for end-of-turn checks.

STT providers decide when speech *stops* purely from silence, so a user who
pauses mid-thought ("I'd like to book for... um...") gets their half-sentence
finalized and answered. An end-of-turn detector adds a second opinion: before
:class:`~voxkit.core.pipeline.VoxkitPipeline` hands a finalized transcript to
the agent, it asks the detector whether the user actually sounds finished.
"""

from abc import ABC, abstractmethod


class EndOfTurnDetector(ABC):
    """Decides whether the user has finished their turn.

    Plug one into :attr:`~voxkit.core.pipeline.PipelineConfig.end_of_turn`.
    The pipeline then:

    - passes every incoming audio chunk to :meth:`push_audio` (the same bytes
      it forwards to the STT provider),
    - calls :meth:`start_turn` when the user starts a new turn, and
    - awaits :meth:`is_end_of_turn` on each finalized transcript.

    When a turn is judged incomplete, the pipeline holds the transcript and
    waits up to :attr:`~voxkit.core.pipeline.PipelineConfig.end_of_turn_timeout`
    seconds for the user to continue; their next transcript is appended and
    checked again.
    """

    @abstractmethod
    def push_audio(self, chunk: bytes) -> None:
        """Receive one chunk of the raw input audio stream.

        Called for every chunk, in order, on the event loop -- so keep it
        cheap (append to a buffer; don't run inference here).

        Args:
            chunk: Raw audio bytes, exactly as passed to
                :meth:`~voxkit.core.pipeline.VoxkitPipeline.run`.
        """

    @abstractmethod
    def start_turn(self) -> None:
        """Mark the start of a new user turn.

        Called on :attr:`~voxkit.stt.base.STTEventType.SPEECH_START` when no
        transcript is being held -- i.e. the user is starting fresh, not
        continuing a turn judged incomplete. Audio from earlier turns should
        no longer influence the next verdict.
        """

    @abstractmethod
    async def is_end_of_turn(self, transcript: str) -> bool:
        """Judge whether the user has finished speaking.

        Called once per finalized transcript, directly on the
        response-latency path -- keep it fast. If it raises, the pipeline
        logs the error and treats the turn as complete, so a broken detector
        never leaves the user unanswered.

        Args:
            transcript: Everything the user has said this turn so far.
                Audio-based detectors may ignore it.

        Returns:
            ``True`` if the turn is complete and the agent should respond,
            ``False`` if the user is likely to continue.
        """
