"""Build synthetic multi-speaker mixtures with exact ground truth.

Real conversational corpora with diarization labels are either gated or
hand-annotated with their own error bars. Concatenating single-speaker
utterances from a clean read-speech corpus gives a benchmark where the turn
boundaries and transcripts are known exactly, the construction is seeded and
reproducible, and nothing requires credentials to download.

The trade-off is stated up front: these mixtures have no overlapped speech,
no channel mismatch between speakers, and read-speech acoustics, so scores
on them are an upper bound on conversational performance, useful for
comparing systems under identical conditions rather than for quoting as
real-world accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Turn


@dataclass
class Utterance:
    """One single-speaker source clip."""

    samples: np.ndarray  # float32 mono
    sample_rate: int
    speaker: str
    transcript: str


@dataclass
class Mixture:
    """A constructed conversation with exact reference labels."""

    samples: np.ndarray
    sample_rate: int
    turns: list[Turn]
    transcripts: list[tuple[str, str]]  # (speaker, transcript) in spoken order

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def reference_text(self) -> str:
        return " ".join(text for _, text in self.transcripts)


def build_mixture(
    utterances: list[Utterance],
    gap_range: tuple[float, float] = (0.4, 1.2),
    seed: int = 0,
) -> Mixture:
    """Concatenate utterances in order with seeded silence gaps between them."""
    if not utterances:
        raise ValueError("need at least one utterance")
    rates = {u.sample_rate for u in utterances}
    if len(rates) != 1:
        raise ValueError(f"utterances must share one sample rate, got {sorted(rates)}")
    sr = rates.pop()

    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    turns: list[Turn] = []
    transcripts: list[tuple[str, str]] = []
    cursor = 0.0
    for i, utt in enumerate(utterances):
        if i > 0:
            gap = float(rng.uniform(*gap_range))
            pieces.append(np.zeros(int(round(gap * sr)), dtype=np.float32))
            cursor += gap
        samples = np.asarray(utt.samples, dtype=np.float32)
        pieces.append(samples)
        duration = len(samples) / sr
        turns.append(Turn(cursor, cursor + duration, utt.speaker))
        transcripts.append((utt.speaker, utt.transcript))
        cursor += duration

    return Mixture(np.concatenate(pieces), sr, turns, transcripts)


def interleave_speakers(
    by_speaker: dict[str, list[Utterance]], seed: int = 0
) -> list[Utterance]:
    """Order utterances so adjacent turns usually change speaker.

    A round-robin draw with seeded speaker order: the hard case for
    diarization is the turn boundary, so the benchmark should be mostly
    boundaries rather than long single-speaker blocks.
    """
    rng = np.random.default_rng(seed)
    queues = {spk: list(utts) for spk, utts in by_speaker.items() if utts}
    order: list[Utterance] = []
    last: str | None = None
    while queues:
        choices = [s for s in queues if s != last] or list(queues)
        speaker = choices[int(rng.integers(len(choices)))]
        order.append(queues[speaker].pop(0))
        if not queues[speaker]:
            del queues[speaker]
        last = speaker
    return order
