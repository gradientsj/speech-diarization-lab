"""WER and DER against hand-computed values."""

import math

from diarlab.metrics import der, normalize_text, wer
from diarlab.types import Turn

# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!") == ["hello", "world"]


def test_normalize_keeps_inword_apostrophes():
    assert normalize_text("It's five o'clock") == ["it's", "five", "o'clock"]


def test_normalize_drops_dangling_apostrophes():
    assert normalize_text("' quoted '") == ["quoted"]


def test_normalize_collapses_whitespace_and_digits():
    assert normalize_text("  3   little  pigs ") == ["3", "little", "pigs"]


# ---------------------------------------------------------------------------
# WER: each case hand-checked
# ---------------------------------------------------------------------------


def test_wer_identical_is_zero():
    r = wer(["the", "cat", "sat"], ["the", "cat", "sat"])
    assert r.wer == 0.0
    assert (r.substitutions, r.deletions, r.insertions) == (0, 0, 0)


def test_wer_single_substitution():
    r = wer(["a", "b", "c"], ["a", "x", "c"])
    assert r.wer == 1 / 3
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 0)


def test_wer_single_deletion():
    r = wer(["a", "b", "c"], ["a", "c"])
    assert r.wer == 1 / 3
    assert (r.substitutions, r.deletions, r.insertions) == (0, 1, 0)


def test_wer_single_insertion():
    r = wer(["a", "c"], ["a", "b", "c"])
    assert r.wer == 0.5
    assert (r.substitutions, r.deletions, r.insertions) == (0, 0, 1)


def test_wer_two_deletions():
    # "the cat sat on the mat" -> "the cat sat mat": "on the" dropped
    r = wer("the cat sat on the mat".split(), "the cat sat mat".split())
    assert r.wer == 2 / 6
    assert (r.substitutions, r.deletions, r.insertions) == (0, 2, 0)


def test_wer_empty_reference_is_nan():
    r = wer([], ["anything"])
    assert math.isnan(r.wer)
    assert r.insertions == 1


def test_wer_can_exceed_one():
    r = wer(["a"], ["x", "y", "z"])
    assert r.wer == 3.0  # 1 substitution + 2 insertions over 1 reference word


# ---------------------------------------------------------------------------
# DER: each case hand-checked (collar=0 unless the collar is the test)
# ---------------------------------------------------------------------------


def test_der_perfect_match():
    ref = [Turn(0, 10, "A")]
    hyp = [Turn(0, 10, "X")]
    r = der(ref, hyp, collar=0)
    assert r.der == 0.0
    assert r.speaker_map == {"X": "A"}


def test_der_missed_speech():
    # hyp stops 2s early: miss 2 of 10 reference seconds
    r = der([Turn(0, 10, "A")], [Turn(0, 8, "A")], collar=0)
    assert r.der == 0.2
    assert r.missed == 2.0
    assert r.false_alarm == 0.0
    assert r.confusion == 0.0


def test_der_label_permutation_is_free():
    ref = [Turn(0, 5, "A"), Turn(5, 10, "B")]
    hyp = [Turn(0, 5, "S1"), Turn(5, 10, "S0")]
    assert der(ref, hyp, collar=0).der == 0.0


def test_der_undersegmentation_is_confusion():
    # one hyp speaker covering two 5s ref speakers: half the time is confused
    ref = [Turn(0, 5, "A"), Turn(5, 10, "B")]
    hyp = [Turn(0, 10, "S")]
    r = der(ref, hyp, collar=0)
    assert r.der == 0.5
    assert r.confusion == 5.0


def test_der_false_alarm():
    # hyp invents 3s of speech after the reference ends: 3/5
    r = der([Turn(0, 5, "A")], [Turn(0, 5, "X"), Turn(5, 8, "Y")], collar=0)
    assert r.der == 0.6
    assert r.false_alarm == 3.0


def test_der_overlapped_reference_counts_double():
    # ref has 2s of two-speaker overlap; hyp finds only one speaker there.
    # total reference = 4*1 + 2*2 + 4*1 = 12; missed = 2 -> DER = 1/6
    ref = [Turn(0, 10, "A"), Turn(4, 6, "B")]
    hyp = [Turn(0, 10, "A")]
    r = der(ref, hyp, collar=0)
    assert r.total_reference == 12.0
    assert r.missed == 2.0
    assert abs(r.der - 1 / 6) < 1e-12


def test_der_collar_absorbs_boundary_jitter():
    # hyp starts 0.2s late, inside the 0.25s collar: scored as perfect
    r = der([Turn(0, 10, "A")], [Turn(0.2, 10, "A")], collar=0.25)
    assert r.der == 0.0


def test_der_empty_hypothesis_is_total_miss():
    r = der([Turn(0, 4, "A")], [], collar=0)
    assert r.der == 1.0
    assert r.missed == 4.0


def test_der_empty_reference_is_nan():
    assert math.isnan(der([], [Turn(0, 1, "X")]).der)
