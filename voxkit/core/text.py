"""Turning streamed agent tokens into speakable sentences.

:class:`SentenceSegmenter` cuts a token stream into sentence/clause-sized
pieces so TTS can start before the agent finishes, and :func:`to_speakable`
strips the markdown that chat models like to emit (``**bold**``, tables,
``<br>``) so the TTS engine reads words, not syntax.
"""

import re

_BOUNDARY = re.compile(r"(?<=[.!?;:,])\s+|\n+")
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_LIST_MARKER = re.compile(r"^\s*[-+*]\s+", re.MULTILINE)
_MD_SYMBOLS = re.compile(r"[*_`#|~>•·▪]+|-{2,}")
_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_WORD = re.compile(r"\w")


def to_speakable(text: str) -> str:
    """Strip markdown/HTML from ``text`` so it reads naturally aloud.

    Args:
        text: A sentence of agent output.

    Returns:
        The cleaned text, or ``""`` if nothing speakable (no letter or digit)
        is left -- e.g. a markdown table separator like ``|---|``. TTS engines
        reject such input, so callers should skip empty results.
    """
    text = _HTML_TAG.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _LIST_MARKER.sub("", text)
    text = _MD_SYMBOLS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text if _WORD.search(text) else ""


class SentenceSegmenter:
    """Accumulates streamed tokens and emits complete, speakable sentences.

    A piece ends at sentence or clause punctuation (``. ! ? ; : ,``) followed
    by whitespace, or at a newline. Pieces shorter than ``min_chars`` are
    merged with the next one, so abbreviations ("Dr. Smith") and short list
    items don't become choppy standalone TTS requests.

    Example:
        >>> segmenter = SentenceSegmenter()
        >>> segmenter.push("Hello there, how are")
        []
        >>> segmenter.push(" you today? I'm fine")
        ['Hello there, how are you today?']
        >>> segmenter.flush()
        ["I'm fine"]
    """

    def __init__(self, min_chars: int = 20) -> None:
        """Create an empty segmenter.

        Args:
            min_chars: Minimum length of an emitted piece; shorter pieces
                are held and merged with what follows.
        """
        self.min_chars = min_chars
        self._buffer = ""
        self._carry = ""

    def push(self, token: str) -> list[str]:
        """Add a token and return any sentences it completed.

        Args:
            token: The next chunk of streamed text.

        Returns:
            Completed, speakable sentences in order (often empty).
        """
        self._buffer += token
        *complete, self._buffer = _BOUNDARY.split(self._buffer)
        sentences = []
        for piece in complete:
            self._carry = f"{self._carry} {piece}".strip()
            if len(to_speakable(self._carry)) >= self.min_chars:
                sentences.extend(self._take_carry())
        return sentences

    def flush(self) -> list[str]:
        """Return whatever text is left once the stream has ended.

        Returns:
            The trailing sentence, if it contains anything speakable.
        """
        self._carry = f"{self._carry} {self._buffer}".strip()
        self._buffer = ""
        return self._take_carry()

    def _take_carry(self) -> list[str]:
        """Empty the carry buffer, returning it if speakable."""
        speakable, self._carry = to_speakable(self._carry), ""
        return [speakable] if speakable else []
