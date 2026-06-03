"""Shared dataclasses for the pipeline.

Times are seconds from the start of the audio throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    """One recognized word with its time span."""

    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass(frozen=True)
class Turn:
    """One span of speech attributed to a single speaker."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Segment:
    """A run of consecutive words by one speaker, for display and subtitles."""

    start: float
    end: float
    speaker: str | None
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Region:
    """A span of detected speech (no speaker attached yet)."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start
