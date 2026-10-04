"""An :class:`~voxkit.turn.base.EndOfTurnDetector` backed by pipecat's Smart Turn v3 model.

`Smart Turn <https://github.com/pipecat-ai/smart-turn>`_ is a small
(~8 MB) audio classifier that listens to the end of an utterance -- intonation,
trailing fillers, cut-off words -- and predicts whether the speaker has
finished. It runs locally on CPU via ONNX Runtime in tens of milliseconds and
needs no API key.

Requires the ``smart-turn`` extra: ``pip install "voxkit[smart-turn]"``.
"""

import asyncio
import logging
from typing import Literal

from voxkit.turn.base import EndOfTurnDetector

logger = logging.getLogger(__name__)

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
"""Hugging Face repo the model is downloaded from."""

SMART_TURN_FILENAME = "smart-turn-v3.2-cpu.onnx"
"""Model file within :data:`SMART_TURN_REPO` (the CPU-optimized v3.2 build)."""


_MODEL_SAMPLE_RATE = 16000
_WINDOW_SECONDS = 8
_PRE_ROLL_SECONDS = 1.0


class PipecatSmartTurnDetector(EndOfTurnDetector):
    """Predicts end-of-turn from the user's audio with Smart Turn v3.

    Buffers the input audio since the user's turn started and, on each
    finalized transcript, scores the last 8 seconds of it. A score at or
    above ``threshold`` means the turn is complete.

    Input audio must be mono 16-bit little-endian PCM (``pcm_s16le``) -- the
    same format :class:`~voxkit.stt.sarvam.SarvamSTTProvider` takes by
    default. 8 kHz audio is upsampled to the 16 kHz the model expects.

    The model is downloaded from Hugging Face on first use and cached;
    construction loads it synchronously, so create the detector once at
    startup.

    Example:
        >>> detector = PipecatSmartTurnDetector(sample_rate=16000)
        >>> config = PipelineConfig(end_of_turn=detector)
    """

    def __init__(
        self,
        sample_rate: Literal[8000, 16000] = 16000,
        threshold: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        """Load the Smart Turn model.

        Args:
            sample_rate: Sample rate of the PCM audio fed to the pipeline.
                Must match the STT provider's input.
            threshold: Completion probability (0.0-1.0) at or above which the
                turn counts as finished. Raise it to wait more readily for
                the user to continue; lower it to respond sooner.
            model_path: Path to a local Smart Turn ONNX file. Defaults to
                downloading :data:`SMART_TURN_FILENAME` from
                :data:`SMART_TURN_REPO`.

        Raises:
            ValueError: If ``sample_rate`` or ``threshold`` is out of range.
            ImportError: If the ``smart-turn`` extra isn't installed.
        """
        if sample_rate not in (8000, 16000):
            raise ValueError(f"sample_rate must be 8000 or 16000, got {sample_rate}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import WhisperFeatureExtractor
        except ImportError as e:
            msg = 'PipecatSmartTurnDetector requires the smart-turn extra: pip install "voxkit[smart-turn]"'
            raise ImportError(msg) from e

        self.sample_rate = sample_rate
        self.threshold = threshold

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = model_path or hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILENAME)
        self._session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=_WINDOW_SECONDS)

        self._bytes_per_second = sample_rate * 2
        self._audio = bytearray()

    def push_audio(self, chunk: bytes) -> None:
        """Append ``chunk`` to the turn buffer, keeping only the last 8 seconds.

        Args:
            chunk: Raw ``pcm_s16le`` audio bytes.
        """
        self._audio += chunk
        overflow = len(self._audio) - _WINDOW_SECONDS * self._bytes_per_second
        if overflow > 0:
            del self._audio[: overflow + overflow % 2]

    def start_turn(self) -> None:
        """Drop buffered audio from before this turn, keeping a short pre-roll."""
        keep = int(_PRE_ROLL_SECONDS * self._bytes_per_second)
        keep -= keep % 2
        if len(self._audio) > keep:
            del self._audio[:-keep]

    async def is_end_of_turn(self, transcript: str) -> bool:
        """Score the buffered turn audio with Smart Turn.

        Args:
            transcript: Ignored -- the verdict comes from audio alone.

        Returns:
            ``True`` if the completion probability is at least ``threshold``
            (or there's no audio to judge), ``False`` otherwise.
        """
        audio = bytes(self._audio[: len(self._audio) - len(self._audio) % 2])
        if not audio:
            return True
        probability = await asyncio.to_thread(self.predict, audio)
        logger.debug("PipecatSmartTurnDetector: %r -> completion probability %.2f", transcript, probability)
        return probability >= self.threshold

    def predict(self, audio: bytes) -> float:
        """Return Smart Turn's completion probability for ``audio``.

        Blocking (runs ONNX inference); :meth:`is_end_of_turn` calls it off
        the event loop. Mirrors pipecat's reference ``predict_endpoint``.

        Args:
            audio: ``pcm_s16le`` audio at :attr:`sample_rate`. Only the last
                8 seconds are used; shorter audio is zero-padded.

        Returns:
            Probability (0.0-1.0) that the speaker has finished their turn.
        """
        import numpy as np

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if self.sample_rate != _MODEL_SAMPLE_RATE:
            positions = np.arange(0, len(samples), self.sample_rate / _MODEL_SAMPLE_RATE)
            samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
        samples = samples[-_WINDOW_SECONDS * _MODEL_SAMPLE_RATE :]

        features = self._feature_extractor(
            samples,
            sampling_rate=_MODEL_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_WINDOW_SECONDS * _MODEL_SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        return float(self._session.run(None, {"input_features": features})[0].flatten()[0])
