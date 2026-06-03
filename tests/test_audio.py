"""Audio loading and resampling."""

import numpy as np
import soundfile as sf

from diarlab.audio import load_audio, resample


def test_resample_doubles_length():
    audio = np.sin(2 * np.pi * 440 * np.arange(8_000) / 8_000).astype(np.float32)
    out = resample(audio, 8_000, 16_000)
    assert len(out) == 16_000
    assert out.dtype == np.float32


def test_resample_noop_at_same_rate():
    audio = np.zeros(100, dtype=np.float32)
    assert resample(audio, 16_000, 16_000) is audio


def test_load_audio_mixes_to_mono_and_resamples(tmp_path):
    sr = 8_000
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo.astype(np.float32), sr)

    audio, rate = load_audio(path, target_rate=16_000)
    assert rate == 16_000
    assert audio.ndim == 1
    assert len(audio) == 16_000
    assert audio.dtype == np.float32
