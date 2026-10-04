"""End-of-turn detection: deciding whether the user has actually finished speaking."""

from voxkit.turn.base import EndOfTurnDetector
from voxkit.turn.smart_turn import SMART_TURN_FILENAME, SMART_TURN_REPO, PipecatSmartTurnDetector

__all__ = ["SMART_TURN_FILENAME", "SMART_TURN_REPO", "EndOfTurnDetector", "PipecatSmartTurnDetector"]
