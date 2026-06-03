"""Word-to-speaker assignment and segment grouping edge cases."""

from diarlab.align import assign_words, group_segments
from diarlab.types import Turn, Word


def test_word_inside_turn():
    words = [Word(1.0, 1.5, "hello")]
    turns = [Turn(0, 2, "A"), Turn(2, 4, "B")]
    assert assign_words(words, turns) == [(words[0], "A")]


def test_word_spanning_boundary_goes_to_larger_overlap():
    # word [1.8, 2.4]: 0.2s in A, 0.4s in B
    words = [Word(1.8, 2.4, "word")]
    turns = [Turn(0, 2, "A"), Turn(2, 4, "B")]
    assert assign_words(words, turns)[0][1] == "B"


def test_overlap_tie_breaks_to_earlier_turn():
    # word [1.5, 2.5]: exactly 0.5s in each turn -> deterministic earlier winner
    words = [Word(1.5, 2.5, "word")]
    turns = [Turn(2, 4, "B"), Turn(0, 2, "A")]  # given out of order on purpose
    assert assign_words(words, turns)[0][1] == "A"


def test_word_in_silence_near_turn_within_gap():
    words = [Word(1.5, 1.8, "word")]
    turns = [Turn(0, 1, "A")]
    assert assign_words(words, turns, max_gap=1.0)[0][1] == "A"


def test_word_far_from_any_turn_is_unattributed():
    words = [Word(5.0, 5.2, "word")]
    turns = [Turn(0, 1, "A")]
    assert assign_words(words, turns, max_gap=1.0)[0][1] is None


def test_no_turns_means_no_speakers():
    words = [Word(0, 1, "word")]
    assert assign_words(words, [])[0][1] is None


def test_group_merges_same_speaker_words():
    assigned = [
        (Word(0.0, 0.4, "hello"), "A"),
        (Word(0.5, 0.9, "there"), "A"),
    ]
    segments = group_segments(assigned)
    assert len(segments) == 1
    assert segments[0].text == "hello there"
    assert segments[0].start == 0.0
    assert segments[0].end == 0.9


def test_group_splits_on_speaker_change():
    assigned = [
        (Word(0.0, 0.4, "hi"), "A"),
        (Word(0.5, 0.9, "yo"), "B"),
    ]
    segments = group_segments(assigned)
    assert [s.speaker for s in segments] == ["A", "B"]


def test_group_splits_on_long_pause():
    assigned = [
        (Word(0.0, 0.4, "one"), "A"),
        (Word(3.0, 3.4, "two"), "A"),
    ]
    segments = group_segments(assigned, max_pause=1.0)
    assert len(segments) == 2


def test_group_empty_input():
    assert group_segments([]) == []
