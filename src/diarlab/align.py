"""Assign recognized words to diarized speaker turns.

The ASR and the diarizer run independently and disagree at the edges, so the
join is where the output quality is decided. The rules are deliberately
simple and fully tested:

- a word goes to the turn it overlaps most (ties break to the earlier turn,
  so the result is deterministic);
- a word that overlaps no turn goes to the nearest turn boundary if it is
  within `max_gap` seconds, otherwise its speaker is None rather than a
  guess.
"""

from __future__ import annotations

from .types import Segment, Turn, Word


def _overlap(word: Word, turn: Turn) -> float:
    return max(0.0, min(word.end, turn.end) - max(word.start, turn.start))


def _distance(word: Word, turn: Turn) -> float:
    if word.end < turn.start:
        return turn.start - word.end
    if word.start > turn.end:
        return word.start - turn.end
    return 0.0


def assign_words(
    words: list[Word], turns: list[Turn], max_gap: float = 1.0
) -> list[tuple[Word, str | None]]:
    """Pair every word with a speaker label (or None when unattributable)."""
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    out: list[tuple[Word, str | None]] = []
    for word in words:
        best_turn: Turn | None = None
        best_overlap = 0.0
        for turn in ordered:
            ov = _overlap(word, turn)
            if ov > best_overlap:
                best_overlap, best_turn = ov, turn
        if best_turn is None:
            # no overlap at all: fall back to the nearest turn within max_gap
            candidates = [(d, t) for t in ordered if (d := _distance(word, t)) <= max_gap]
            if candidates:
                candidates.sort(key=lambda dt: (dt[0], dt[1].start))
                best_turn = candidates[0][1]
        out.append((word, best_turn.speaker if best_turn else None))
    return out


def group_segments(
    assigned: list[tuple[Word, str | None]], max_pause: float = 1.0
) -> list[Segment]:
    """Merge consecutive same-speaker words into display segments.

    A segment breaks on a speaker change or on a silence longer than
    `max_pause`, which keeps subtitles readable instead of emitting one
    segment per speaker per file.
    """
    segments: list[Segment] = []
    for word, speaker in assigned:
        current = segments[-1] if segments else None
        if (
            current is not None
            and current.speaker == speaker
            and word.start - current.end <= max_pause
        ):
            current.words.append(word)
            current.end = word.end
            current.text = f"{current.text} {word.text.strip()}".strip()
        else:
            segments.append(
                Segment(
                    start=word.start,
                    end=word.end,
                    speaker=speaker,
                    text=word.text.strip(),
                    words=[word],
                )
            )
    return segments
