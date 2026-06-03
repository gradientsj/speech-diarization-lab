"""Audio loading: mono float32 at a target sample rate.

soundfile covers wav/flac/ogg, which is everything the benchmark and tests
use. Resampling is polyphase (scipy) rather than naive interpolation so the
embedding and VAD models see properly band-limited input.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16_000


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(source_rate, target_rate)
    out = resample_poly(audio, target_rate // gcd, source_rate // gcd)
    return out.astype(np.float32)


def load_audio(path: str | Path, target_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Read a file as mono float32 at `target_rate`, returning (samples, rate)."""
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    return resample(mono, rate, target_rate), target_rate


def save_audio(path: str | Path, audio: np.ndarray, rate: int) -> None:
    sf.write(str(path), audio, rate)
