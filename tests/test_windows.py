"""VAD post-processing, window slicing, and turn building."""

import pytest

from diarlab.types import Region
from diarlab.windows import merge_regions, slice_windows, windows_to_turns


def test_merge_bridges_small_gaps():
    regions = [Region(0.0, 1.0), Region(1.1, 2.0)]
    merged = merge_regions(regions, min_gap=0.3, min_duration=0.2, pad=0.05)
    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 2.05


def test_merge_keeps_large_gaps_separate():
    regions = [Region(0.0, 1.0), Region(2.0, 3.0)]
    merged = merge_regions(regions, min_gap=0.3, min_duration=0.2, pad=0.05)
    assert len(merged) == 2


def test_merge_drops_blips():
    merged = merge_regions([Region(0.0, 0.1)], min_gap=0.3, min_duration=0.2, pad=0.05)
    assert merged == []


def test_merge_pad_never_goes_negative():
    merged = merge_regions([Region(0.0, 1.0)], pad=0.5)
    assert merged[0].start == 0.0


def test_slice_short_region_is_one_window():
    windows = slice_windows([Region(0.0, 1.0)], window=1.5, stride=0.75)
    assert len(windows) == 1
    assert (windows[0].start, windows[0].end) == (0.0, 1.0)


def test_slice_covers_region_tail():
    windows = slice_windows([Region(0.0, 4.0)], window=1.5, stride=0.75)
    starts = [w.start for w in windows]
    assert starts == [0.0, 0.75, 1.5, 2.25, 2.5]
    assert all(w.end - w.start == 1.5 for w in windows)
    assert windows[-1].end == 4.0


def test_slice_never_crosses_region_boundary():
    windows = slice_windows([Region(0.0, 2.0), Region(5.0, 6.0)], window=1.5, stride=0.75)
    assert all(not (w.start < 2.0 < w.end) for w in windows)


def test_turns_merge_same_label():
    windows = [Region(0.0, 1.5), Region(0.75, 2.25)]
    turns = windows_to_turns(windows, [0, 0])
    assert len(turns) == 1
    assert (turns[0].start, turns[0].end, turns[0].speaker) == (0.0, 2.25, "SPEAKER_00")


def test_turns_split_overlap_disagreement_at_midpoint():
    windows = [Region(0.0, 1.5), Region(1.0, 2.5)]
    turns = windows_to_turns(windows, [0, 1])
    assert (turns[0].start, turns[0].end) == (0.0, 1.25)
    assert (turns[1].start, turns[1].end) == (1.25, 2.5)
    assert turns[0].speaker != turns[1].speaker


def test_turns_gap_beyond_merge_gap_splits():
    windows = [Region(0.0, 1.0), Region(2.0, 3.0)]
    turns = windows_to_turns(windows, [0, 0], max_merge_gap=0.5)
    assert len(turns) == 2
    assert turns[0].speaker == turns[1].speaker == "SPEAKER_00"


def test_turns_length_mismatch_raises():
    with pytest.raises(ValueError):
        windows_to_turns([Region(0, 1)], [0, 1])


def test_turns_empty():
    assert windows_to_turns([], []) == []
