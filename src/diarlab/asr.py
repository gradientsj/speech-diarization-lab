"""Transcription via faster-whisper (CTranslate2), with word timestamps.

CTranslate2 is the optimization story of this project: the same Whisper
weights run several times faster than the reference PyTorch implementation,
with int8 quantization making CPU transcription practical. The wrapper
returns plain dataclasses so the rest of the pipeline has no dependency on
faster-whisper types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .types import Word


@dataclass
class TranscriptionResult:
    text: str
    words: list[Word]
    language: str
    audio_duration: float
    transcribe_seconds: float

    @property
    def real_time_factor(self) -> float:
        """Processing time per second of audio; below 1.0 is faster than real time."""
        if self.audio_duration <= 0:
            return float("nan")
        return self.transcribe_seconds / self.audio_duration


@lru_cache(maxsize=2)
def _model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "faster-whisper is not installed; install the model backends with "
            "`uv sync --extra models`"
        ) from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(
    path: str | Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    language: str | None = None,
) -> TranscriptionResult:
    model = _model(model_size, device, compute_type)
    started = time.perf_counter()
    segments, info = model.transcribe(
        str(path),
        beam_size=beam_size,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    words: list[Word] = []
    texts: list[str] = []
    for segment in segments:  # generator: transcription happens during iteration
        texts.append(segment.text.strip())
        for w in segment.words or []:
            words.append(Word(w.start, w.end, w.word.strip(), w.probability))
    elapsed = time.perf_counter() - started

    return TranscriptionResult(
        text=" ".join(t for t in texts if t),
        words=words,
        language=info.language,
        audio_duration=info.duration,
        transcribe_seconds=elapsed,
    )
