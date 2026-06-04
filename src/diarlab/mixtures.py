"""Build synthetic multi-speaker mixtures with exact ground truth.

Real conversational corpora with diarization labels are either gated or
hand-annotated with their own error bars. Concatenating single-speaker
utterances from a clean read-speech corpus gives a benchmark where the turn
boundaries and transcripts are known exactly, the construction is seeded and
reproducible, and nothing requires credentials to download.

The trade-off is stated up front: the default mixtures have no overlapped
speech, no channel mismatch between speakers, and read-speech acoustics, so
scores on them are an upper bound on conversational performance, useful for
comparing systems under identical conditions rather than for quoting as
real-world accuracy. `overlap_prob` adds the first missing stressor: seeded
partial overlap at speaker changes, where the next speaker starts before
the current one finishes and the waveforms sum.
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
    overlap_prob: float = 0.0,
    overlap_range: tuple[float, float] = (0.5, 1.5),
) -> Mixture:
    """Place utterances on a timeline with seeded gaps, optionally overlapping.

    With `overlap_prob` zero this is plain concatenation with silence gaps.
    Otherwise each transition to a *different* speaker overlaps with that
    probability: the next utterance starts before the current one ends and
    the waveforms sum over the contested span. The overlap is capped at
    half the shorter utterance so both turns keep an uncontested core, and
    consecutive turns by the same speaker never overlap (a speaker does not
    talk over themselves). Reference turns record the true placement, so
    the ground truth contains real overlapped speech.
    """
    if not utterances:
        raise ValueError("need at least one utterance")
    rates = {u.sample_rate for u in utterances}
    if len(rates) != 1:
        raise ValueError(f"utterances must share one sample rate, got {sorted(rates)}")
    sr = rates.pop()

    rng = np.random.default_rng(seed)
    # Sample offsets accumulate in integers exactly as concatenation would,
    # so the overlap_prob=0 path stays byte-identical to the original
    # construction; turn times accumulate in float seconds as before.
    placements: list[tuple[int, np.ndarray]] = []
    turns: list[Turn] = []
    transcripts: list[tuple[str, str]] = []
    prev_end = 0.0
    prev_lo = prev_len = 0
    for i, utt in enumerate(utterances):
        samples = np.asarray(utt.samples, dtype=np.float32)
        duration = len(samples) / sr
        if i == 0:
            start, lo = 0.0, 0
        else:
            prev = turns[-1]
            # draw the overlap decision only when overlap is enabled, so the
            # overlap_prob=0 path consumes the same rng sequence as the
            # original construction and the seeded mixtures stay unchanged
            overlap_here = (
                overlap_prob > 0
                and utt.speaker != prev.speaker
                and float(rng.uniform()) < overlap_prob
            )
            if overlap_here:
                cap = min(prev.duration, duration) / 2.0
                overlap = min(float(rng.uniform(*overlap_range)), cap)
                start = prev_end - overlap
                lo = prev_lo + prev_len - int(round(overlap * sr))
            else:
                gap = float(rng.uniform(*gap_range))
                start = prev_end + gap
                lo = prev_lo + prev_len + int(round(gap * sr))
        placements.append((lo, samples))
        turns.append(Turn(start, start + duration, utt.speaker))
        transcripts.append((utt.speaker, utt.transcript))
        prev_end = start + duration
        prev_lo, prev_len = lo, len(samples)

    mix = np.zeros(max(lo + len(s) for lo, s in placements), dtype=np.float32)
    for lo, samples in placements:
        mix[lo : lo + len(samples)] += samples
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 1.0:  # summed overlaps can clip; rescale once, globally
        mix /= peak

    return Mixture(mix, sr, turns, transcripts)


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
