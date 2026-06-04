"""Synthetic mixture construction: exact ground truth by design."""

import numpy as np
import pytest

from diarlab.mixtures import Mixture, Utterance, build_mixture, interleave_speakers


def _utt(freq: float, seconds: float, speaker: str, text: str, sr: int = 16_000) -> Utterance:
    t = np.arange(int(seconds * sr)) / sr
    return Utterance(np.sin(2 * np.pi * freq * t).astype(np.float32), sr, speaker, text)


def test_mixture_turns_and_timing():
    mix = build_mixture(
        [_utt(440, 1.0, "alice", "hello"), _utt(880, 1.0, "bob", "world")],
        gap_range=(0.5, 0.5),
        seed=0,
    )
    assert isinstance(mix, Mixture)
    assert len(mix.samples) == int(2.5 * 16_000)
    assert [(t.start, t.end, t.speaker) for t in mix.turns] == [
        (0.0, 1.0, "alice"),
        (1.5, 2.5, "bob"),
    ]
    assert mix.reference_text() == "hello world"


def test_mixture_gap_is_silent():
    mix = build_mixture(
        [_utt(440, 1.0, "a", "x"), _utt(880, 1.0, "b", "y")],
        gap_range=(0.5, 0.5),
        seed=0,
    )
    gap = mix.samples[int(1.1 * 16_000) : int(1.4 * 16_000)]
    assert np.abs(gap).max() == 0.0


def test_mixture_is_seeded_and_reproducible():
    utts = [_utt(440, 0.5, "a", "x"), _utt(880, 0.5, "b", "y")]
    m1 = build_mixture(utts, seed=7)
    m2 = build_mixture(utts, seed=7)
    assert np.array_equal(m1.samples, m2.samples)
    assert m1.turns == m2.turns


def test_mixture_rejects_mixed_sample_rates():
    with pytest.raises(ValueError):
        build_mixture([_utt(440, 0.5, "a", "x", sr=16_000), _utt(880, 0.5, "b", "y", sr=8_000)])


def test_mixture_rejects_empty():
    with pytest.raises(ValueError):
        build_mixture([])


def test_overlap_prob_one_overlaps_speaker_changes():
    mix = build_mixture(
        [_utt(440, 2.0, "a", "x"), _utt(880, 2.0, "b", "y")],
        seed=0,
        overlap_prob=1.0,
        overlap_range=(0.5, 0.5),
    )
    a, b = mix.turns
    assert b.start == pytest.approx(a.end - 0.5)
    # the contested span carries energy from both signals
    lo, hi = int(b.start * 16_000), int(a.end * 16_000)
    contested = mix.samples[lo:hi]
    assert np.abs(contested).max() > 0


def test_overlap_never_same_speaker():
    mix = build_mixture(
        [_utt(440, 2.0, "a", "x"), _utt(440, 2.0, "a", "y")],
        seed=0,
        overlap_prob=1.0,
    )
    a, b = mix.turns
    assert b.start > a.end  # same speaker: gap, never overlap


def test_overlap_capped_at_half_the_shorter_utterance():
    mix = build_mixture(
        [_utt(440, 4.0, "a", "x"), _utt(880, 1.0, "b", "y")],
        seed=0,
        overlap_prob=1.0,
        overlap_range=(3.0, 3.0),  # asks for more overlap than the cap allows
    )
    a, b = mix.turns
    assert a.end - b.start == pytest.approx(0.5)  # half of the 1.0 s utterance


def test_overlap_zero_matches_plain_concatenation():
    utts = [_utt(440, 1.0, "a", "x"), _utt(880, 1.0, "b", "y")]
    plain = build_mixture(utts, gap_range=(0.5, 0.5), seed=3)
    explicit = build_mixture(utts, gap_range=(0.5, 0.5), seed=3, overlap_prob=0.0)
    assert np.array_equal(plain.samples, explicit.samples)
    assert plain.turns == explicit.turns


def test_overlap_peak_is_normalized():
    # two loud utterances summed over the full overlap would clip without rescale
    sr = 16_000
    loud = np.ones(sr, dtype=np.float32) * 0.9
    utts = [
        Utterance(loud, sr, "a", "x"),
        Utterance(loud.copy(), sr, "b", "y"),
    ]
    mix = build_mixture(utts, seed=0, overlap_prob=1.0, overlap_range=(0.5, 0.5))
    assert np.abs(mix.samples).max() <= 1.0


def test_interleave_preserves_utterances():
    by_speaker = {
        "a": [_utt(440, 0.5, "a", "1"), _utt(440, 0.5, "a", "2")],
        "b": [_utt(880, 0.5, "b", "3")],
    }
    order = interleave_speakers(by_speaker, seed=0)
    assert len(order) == 3
    assert sorted(u.speaker for u in order) == ["a", "a", "b"]
    # within one speaker, original order is kept
    a_texts = [u.transcript for u in order if u.speaker == "a"]
    assert a_texts == ["1", "2"]
