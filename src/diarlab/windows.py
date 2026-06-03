"""Pure timeline helpers: VAD post-processing, embedding windows, turn building.

Everything in this module is plain arithmetic over (start, end) spans, kept
free of model dependencies so the logic that actually decides the diarization
output can be unit-tested exhaustively.
"""

from __future__ import annotations

from .types import Region, Turn


def merge_regions(
    regions: list[Region],
    min_gap: float = 0.3,
    min_duration: float = 0.2,
    pad: float = 0.05,
) -> list[Region]:
    """Clean raw VAD output: pad edges, bridge small gaps, drop blips.

    VAD models emit jittery boundaries; merging regions separated by less
    than `min_gap` and dropping regions shorter than `min_duration` keeps one
    region per utterance instead of a stutter of fragments.
    """
    padded = sorted(
        (Region(max(0.0, r.start - pad), r.end + pad) for r in regions),
        key=lambda r: r.start,
    )
    merged: list[Region] = []
    for region in padded:
        if merged and region.start - merged[-1].end < min_gap:
            merged[-1].end = max(merged[-1].end, region.end)
        else:
            merged.append(Region(region.start, region.end))
    return [r for r in merged if r.duration >= min_duration]


def slice_windows(
    regions: list[Region], window: float = 1.5, stride: float = 0.75
) -> list[Region]:
    """Cut speech regions into fixed windows for speaker embedding.

    Windows never cross a region boundary (a window spanning two speakers
    through a silence would embed a mixture). A region shorter than `window`
    yields itself as a single window, and the final window of a region is
    right-aligned so the region's tail is always covered.
    """
    out: list[Region] = []
    for region in regions:
        if region.duration <= window:
            out.append(Region(region.start, region.end))
            continue
        start = region.start
        while start + window < region.end:
            out.append(Region(start, start + window))
            start += stride
        out.append(Region(region.end - window, region.end))
    return out


def windows_to_turns(
    windows: list[Region], labels: list[int], max_merge_gap: float = 0.5
) -> list[Turn]:
    """Turn labeled, possibly overlapping windows into clean speaker turns.

    Consecutive windows with the same label merge into one turn. Where two
    neighboring windows overlap but disagree on the speaker, the boundary is
    placed at the midpoint of the overlap, since neither window has better
    evidence inside the contested span.
    """
    if len(windows) != len(labels):
        raise ValueError("windows and labels must be the same length")
    if not windows:
        return []

    items = sorted(zip(windows, labels, strict=True), key=lambda wl: (wl[0].start, wl[0].end))
    turns: list[Turn] = []
    cur_start, cur_end = items[0][0].start, items[0][0].end
    cur_label = items[0][1]
    for region, label in items[1:]:
        start, end = region.start, region.end
        if label == cur_label and start - cur_end <= max_merge_gap:
            cur_end = max(cur_end, end)
            continue
        if start < cur_end:  # overlapping disagreement: split at the midpoint
            cut = (start + cur_end) / 2.0
            cur_end, start = cut, cut
        turns.append(Turn(cur_start, cur_end, f"SPEAKER_{cur_label:02d}"))
        cur_start, cur_end, cur_label = start, max(start, end), label
    turns.append(Turn(cur_start, cur_end, f"SPEAKER_{cur_label:02d}"))
    return [t for t in turns if t.duration > 0]
