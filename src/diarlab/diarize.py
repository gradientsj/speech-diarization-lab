"""Diarization backends.

Two implementations behind one signature, mirroring how the other lab repos
compare a thing built from parts against a pretrained reference:

- `clustered`: silero VAD -> windowed ECAPA embeddings -> agglomerative
  clustering -> turns. Every stage is visible and the stage boundaries are
  pure functions with tests.
- `pyannote`: the pretrained pyannote/speaker-diarization-3.1 pipeline,
  gated on Hugging Face (requires HF_TOKEN with the model terms accepted).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cluster import cluster_embeddings
from .types import Turn
from .windows import merge_regions, slice_windows, windows_to_turns


@dataclass
class ClusteredConfig:
    window: float = 1.5
    stride: float = 0.75
    distance_threshold: float = 0.6
    num_speakers: int | None = None
    vad_threshold: float = 0.5
    min_gap: float = 0.3
    min_duration: float = 0.2
    device: str = "cpu"


def diarize_clustered(
    audio: np.ndarray, sample_rate: int, config: ClusteredConfig | None = None
) -> list[Turn]:
    """The from-parts pipeline: VAD, embed, cluster, build turns."""
    from .embeddings import EcapaEmbedder
    from .vad import detect_speech

    cfg = config or ClusteredConfig()
    raw = detect_speech(audio, sample_rate, threshold=cfg.vad_threshold)
    regions = merge_regions(raw, min_gap=cfg.min_gap, min_duration=cfg.min_duration)
    if not regions:
        return []
    windows = slice_windows(regions, window=cfg.window, stride=cfg.stride)
    embedder = EcapaEmbedder(device=cfg.device)
    embeddings = embedder.embed_windows(audio, sample_rate, windows)
    labels = cluster_embeddings(
        embeddings,
        distance_threshold=cfg.distance_threshold,
        num_speakers=cfg.num_speakers,
    )
    return windows_to_turns(windows, labels)


def diarize_pyannote(
    path: str | Path,
    num_speakers: int | None = None,
    device: str = "cpu",
) -> list[Turn]:
    """The pretrained reference pipeline (gated; needs HF_TOKEN)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. pyannote/speaker-diarization-3.1 is gated: create a "
            "Hugging Face token, accept the terms on the model page (and on "
            "pyannote/segmentation-3.0), then `setx HF_TOKEN <token>`."
        )
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "pyannote.audio is not installed; install the reference backend with "
            "`uv sync --extra reference`"
        ) from exc

    # torch >= 2.6 defaults torch.load to weights_only=True, which rejects the
    # non-tensor objects pickled inside the pyannote 3.1 checkpoint. The
    # checkpoint comes from the gated repo the user explicitly accepted, so
    # allowlist those globals rather than disabling weights_only.
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    torch.serialization.add_safe_globals(
        [torch.torch_version.TorchVersion, Specifications, Problem, Resolution]
    )

    try:  # pyannote.audio >= 4 renamed the kwarg
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
    pipeline.to(torch.device(device))
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    result = pipeline(str(path), **kwargs)
    # pyannote.audio >= 4 wraps the Annotation in a result object
    annotation = getattr(result, "speaker_diarization", result)
    return [
        Turn(segment.start, segment.end, str(label))
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
