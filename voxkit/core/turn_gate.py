"""Deciding when a user's turn is over and should go to the agent."""

import asyncio
import logging
from collections.abc import Callable

from voxkit.turn import EndOfTurnDetector

logger = logging.getLogger(__name__)


class EndOfTurnGate:
    """Holds finalized transcripts until the user's turn is judged complete.

    Without a detector, every transcript is committed immediately (merged with
    any restored text). With one,
    the gate runs this state machine:

    - **Transcript arrives** -> append it to the held text and ask the
      detector. Complete -> commit. Incomplete -> wait ``timeout`` seconds.
    - **Speech starts** -> pause the wait (the user is continuing). If
      nothing is held, tell the detector a fresh turn has started.
    - **Speech ends without a transcript** (e.g. noise) -> resume the wait.
    - **Wait expires** -> commit whatever is held.

    If a committed turn's reply is cancelled before it was spoken, the
    pipeline hands its text back via :meth:`restore`, so it merges with the
    user's continuation instead of being lost.

    A detector error counts as "complete", so the user is never left
    unanswered.
    """

    def __init__(
        self,
        detector: EndOfTurnDetector | None,
        timeout: float,
        commit: Callable[[str], None],
    ) -> None:
        """Create the gate.

        Args:
            detector: The end-of-turn check, or ``None`` to commit every
                transcript immediately.
            timeout: Seconds to wait after an incomplete verdict before
                committing anyway.
            commit: Called (synchronously) with the full text of each
                completed user turn.
        """
        self.detector = detector
        self.timeout = timeout
        self._commit = commit
        self._held = ""
        self._task: asyncio.Task | None = None

    def push_audio(self, chunk: bytes) -> None:
        """Forward an input audio chunk to the detector, if any.

        Args:
            chunk: Raw audio bytes from the input stream.
        """
        if self.detector is not None:
            self.detector.push_audio(chunk)

    def on_speech_start(self) -> None:
        """Handle the user starting to speak."""
        if self.detector is not None and not self._held:
            self.detector.start_turn()
        self._cancel()

    def on_speech_end(self) -> None:
        """Handle the user going quiet."""
        if self._held and (self._task is None or self._task.done()):
            self._start(check=False)

    def on_transcript(self, text: str) -> None:
        """Handle a finalized transcript.

        Args:
            text: The transcript text (already stripped, non-empty).
        """
        self._held = f"{self._held} {text}".strip()
        if self.detector is None:
            self._cancel()
            text, self._held = self._held, ""
            self._commit(text)
            return
        self._start(check=True)

    def restore(self, text: str) -> None:
        """Put back the text of a committed turn whose reply was cancelled before it was spoken.

        It is placed ahead of anything already held and is committed together
        with the user's next words (or after the timeout if none come).

        Args:
            text: The cancelled turn's text.
        """
        self._held = f"{text} {self._held}".strip()

    def close(self) -> None:
        """Abandon any pending decision. Held text is discarded."""
        self._cancel()
        self._held = ""

    def _cancel(self) -> None:
        """Stop any in-flight check or wait, keeping the held text."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def _start(self, check: bool) -> None:
        """(Re)start deciding when to commit the held text.

        Args:
            check: Run the detector first; ``False`` goes straight to the wait.
        """
        self._cancel()
        self._task = asyncio.create_task(self._resolve(check))

    async def _resolve(self, check: bool) -> None:
        """Commit the held text once judged complete or once the wait expires.

        Args:
            check: Run the detector before waiting.
        """
        complete = False
        if check and self.detector is not None:
            try:
                complete = await self.detector.is_end_of_turn(self._held)
            except Exception:
                logger.exception("EndOfTurnGate: detector failed, treating turn as complete")
                complete = True
        if not complete:
            if check:
                logger.debug("EndOfTurnGate: holding %r, sounds incomplete", self._held)
            logger.debug("EndOfTurnGate: waiting %.1fs for the user to continue", self.timeout)
            await asyncio.sleep(self.timeout)

        # No await between taking the text and committing it, so cancellation can't split them.
        text, self._held = self._held, ""
        if text:
            self._commit(text)
