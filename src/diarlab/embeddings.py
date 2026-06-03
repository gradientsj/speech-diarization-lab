"""Speaker embeddings from a pretrained ECAPA-TDNN (lazy import).

speechbrain/spkrec-ecapa-voxceleb is ungated, small (~80 MB), and the
standard open speaker-verification embedder. Each diarization window is
embedded independently; batching pads to the longest clip in the batch and
passes relative lengths so padding does not contaminate the embedding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Region

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class EcapaEmbedder:
    def __init__(self, device: str = "cpu", cache_dir: str | Path | None = None) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ImportError(
                "speechbrain is not installed; install the model backends with "
                "`uv sync --extra models`"
            ) from exc
        if device == "cuda":  # speechbrain's parser wants an explicit index
            device = "cuda:0"
        savedir = Path(cache_dir or Path.home() / ".cache" / "diarlab" / "ecapa")
        self._classifier = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE,
            savedir=str(savedir),
            run_opts={"device": device},
        )

    def embed_windows(
        self,
        audio: np.ndarray,
        sample_rate: int,
        windows: list[Region],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Embed each window; returns an array of shape (len(windows), 192)."""
        import torch

        clips = []
        for w in windows:
            lo, hi = int(w.start * sample_rate), int(w.end * sample_rate)
            clips.append(np.ascontiguousarray(audio[lo:hi], dtype=np.float32))

        out: list[np.ndarray] = []
        for i in range(0, len(clips), batch_size):
            batch = clips[i : i + batch_size]
            longest = max(len(c) for c in batch)
            padded = np.zeros((len(batch), longest), dtype=np.float32)
            lengths = np.zeros(len(batch), dtype=np.float32)
            for j, clip in enumerate(batch):
                padded[j, : len(clip)] = clip
                lengths[j] = len(clip) / longest
            with torch.no_grad():
                emb = self._classifier.encode_batch(
                    torch.from_numpy(padded), wav_lens=torch.from_numpy(lengths)
                )
            out.append(emb.squeeze(1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 192), dtype=np.float32)
