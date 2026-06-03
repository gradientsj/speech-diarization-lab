"""WER and DER implemented from scratch.

The scoring math is the part of this project that everything else leans on,
so it is written in plain Python with numpy/scipy only and tested against
hand-computed values. Where a metric is undefined (empty reference), it
returns NaN rather than a silently-wrong 0.0, and NaN must be treated as a
failure by anything that gates on these numbers.

Definitions follow the standard tooling:

- WER: word error rate, (substitutions + deletions + insertions) / reference
  words, computed by Levenshtein alignment.
- DER: diarization error rate as scored by NIST md-eval, (missed speech +
  false alarm speech + speaker confusion) / total reference speech time,
  with an optional no-score collar around reference turn boundaries and an
  optimal one-to-one speaker mapping found by the Hungarian algorithm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Turn

# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation except in-word apostrophes, split on space.

    Whisper emits punctuated, mixed-case text while corpus references are
    often bare uppercase words; both sides must pass through the same
    normalizer before scoring or the WER measures formatting, not recognition.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    # drop apostrophes that do not join two letters ('cause vs. dogs')
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)
    return text.split()


@dataclass(frozen=True)
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def wer(reference: list[str], hypothesis: list[str]) -> WerResult:
    """Word error rate by Levenshtein alignment over already-normalized words."""
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return WerResult(float("nan"), 0, 0, m, 0)

    # dp[i][j] = (cost, subs, dels, ins) for ref[:i] vs hyp[:j]
    cost = np.zeros((n + 1, m + 1), dtype=np.int64)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = reference[i - 1] == hypothesis[j - 1]
            cost[i, j] = min(
                cost[i - 1, j - 1] + (0 if same else 1),  # match / substitution
                cost[i - 1, j] + 1,  # deletion
                cost[i, j - 1] + 1,  # insertion
            )

    # backtrack to split the edit count into S/D/I
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        diag_ok = i > 0 and j > 0
        if diag_ok and cost[i, j] == cost[i - 1, j - 1] and reference[i - 1] == hypothesis[j - 1]:
            i, j = i - 1, j - 1
        elif diag_ok and cost[i, j] == cost[i - 1, j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            dels += 1
            i = i - 1
        else:
            ins += 1
            j = j - 1

    return WerResult((subs + dels + ins) / n, subs, dels, ins, n)


# ---------------------------------------------------------------------------
# DER
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerResult:
    der: float
    missed: float  # seconds of reference speech with too few hypothesis speakers
    false_alarm: float  # seconds of hypothesis speech with too few reference speakers
    confusion: float  # seconds attributed to the wrong (mapped) speaker
    total_reference: float  # scored reference speech in seconds
    speaker_map: dict[str, str]  # hypothesis label -> reference label


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _subtract_intervals(
    span: tuple[float, float], holes: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return the parts of `span` not covered by the merged `holes`."""
    pieces = []
    cursor, end = span
    for h_start, h_end in holes:
        if h_end <= cursor or h_start >= end:
            continue
        if h_start > cursor:
            pieces.append((cursor, h_start))
        cursor = max(cursor, h_end)
        if cursor >= end:
            break
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def _active(turns: list[Turn], start: float, end: float) -> list[str]:
    mid = (start + end) / 2.0
    return [t.speaker for t in turns if t.start < mid < t.end]


def der(reference: list[Turn], hypothesis: list[Turn], collar: float = 0.25) -> DerResult:
    """Diarization error rate with md-eval semantics.

    The timeline is cut at every reference/hypothesis turn boundary into
    elementary intervals on which the active speaker sets are constant, a
    collar around each reference boundary is excluded from scoring, the
    one-to-one speaker mapping that maximizes attributed time is chosen, and
    the three error components are integrated over the scored intervals.
    """
    if not reference:
        return DerResult(float("nan"), 0.0, 0.0, 0.0, 0.0, {})

    holes: list[tuple[float, float]] = []
    if collar > 0:
        for t in reference:
            holes.append((t.start - collar, t.start + collar))
            holes.append((t.end - collar, t.end + collar))
        holes = _merge_intervals(holes)

    boundaries = sorted(
        {t.start for t in reference}
        | {t.end for t in reference}
        | {t.start for t in hypothesis}
        | {t.end for t in hypothesis}
        | {h for hole in holes for h in hole}
    )

    # scored elementary intervals with their active speaker sets
    scored: list[tuple[float, list[str], list[str]]] = []  # (duration, ref_set, hyp_set)
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        for s, e in _subtract_intervals((left, right), holes):
            if e - s <= 0:
                continue
            ref_active = _active(reference, s, e)
            hyp_active = _active(hypothesis, s, e)
            if ref_active or hyp_active:
                scored.append((e - s, ref_active, hyp_active))

    # optimal mapping: maximize the total time on which a hypothesis label
    # overlaps the reference label it is assigned to
    ref_labels = sorted({t.speaker for t in reference})
    hyp_labels = sorted({t.speaker for t in hypothesis})
    overlap = np.zeros((len(ref_labels), len(hyp_labels)))
    for dur, ref_active, hyp_active in scored:
        for r in ref_active:
            for h in hyp_active:
                overlap[ref_labels.index(r), hyp_labels.index(h)] += dur
    mapping: dict[str, str] = {}
    if len(ref_labels) and len(hyp_labels):
        rows, cols = linear_sum_assignment(-overlap)
        mapping = {
            hyp_labels[c]: ref_labels[r]
            for r, c in zip(rows, cols, strict=True)
            if overlap[r, c] > 0
        }

    missed = false_alarm = confusion = total_ref = 0.0
    for dur, ref_active, hyp_active in scored:
        n_ref, n_hyp = len(ref_active), len(hyp_active)
        total_ref += dur * n_ref
        correct = sum(1 for h in hyp_active if mapping.get(h) in ref_active)
        missed += dur * max(0, n_ref - n_hyp)
        false_alarm += dur * max(0, n_hyp - n_ref)
        confusion += dur * (min(n_ref, n_hyp) - correct)

    if total_ref <= 0:
        return DerResult(float("nan"), missed, false_alarm, confusion, 0.0, mapping)
    return DerResult(
        (missed + false_alarm + confusion) / total_ref,
        missed,
        false_alarm,
        confusion,
        total_ref,
        mapping,
    )
