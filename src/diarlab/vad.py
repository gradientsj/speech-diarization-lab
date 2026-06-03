"""Voice activity detection via Silero VAD (lazy import).

The model call lives here and nothing else: thresholding, merging, and
padding decisions happen in `windows.merge_regions`, which is pure and
tested. The raw model output is converted from sample indices to seconds
without intermediate rounding.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .types import Region

VAD_SAMPLE_RATE = 16_000


@lru_cache(maxsize=1)
def _model():
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "silero-vad is not installed; install the model backends with "
            "`uv sync --extra models`"
        ) from exc
    return load_silero_vad()


def detect_speech(audio: np.ndarray, sample_rate: int, threshold: float = 0.5) -> list[Region]:
    """Return raw speech regions in seconds (post-process with merge_regions)."""
    if sample_rate != VAD_SAMPLE_RATE:
        raise ValueError(f"Silero VAD expects {VAD_SAMPLE_RATE} Hz audio, got {sample_rate}")
    import torch
    from silero_vad import get_speech_timestamps

    stamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio)),
        _model(),
        sampling_rate=sample_rate,
        threshold=threshold,
    )
    return [Region(s["start"] / sample_rate, s["end"] / sample_rate) for s in stamps]
