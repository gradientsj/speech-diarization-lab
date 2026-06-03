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
