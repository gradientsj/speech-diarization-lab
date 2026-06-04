"""Near-real-time speaker-attributed transcription over rolling chunks.

The offline pipeline clusters a whole file at once, which a live stream
cannot do. This module processes fixed-length chunks as they arrive and
keeps speaker identities stable across chunks with an online tracker:
each chunk is clustered locally (same agglomerative recipe), then each
local cluster's centroid is matched against the running global centroids,
reusing a global speaker id when the cosine distance is below the
centroid threshold and minting a new one otherwise.

The centroid threshold is deliberately not the dendrogram cut. Centroids
are denoised means, so cross-speaker centroid distances run far below
window-level linkage distances; measured on the benchmark mixtures,
same-speaker chunk centroids sit at 0.20-0.42 and different-speaker ones
at 0.58+, so the default 0.50 splits the gap with margin on both sides.

Known limits, stated up front: chunk boundaries can split words, a chunk
shorter than a turn can fragment it, and concurrent speech still yields
one speaker at a time (see the overlap section of the README). Speaker
ids can also have gaps: a short sliver of a voice at a chunk boundary may
mint an id whose noisy centroid never matches again. This is the
simplified version of streaming diarization, built to be honest about
exactly those seams.

The model calls are injected, so every decision in this module is testable
on CPU with stub functions; the server wires in the real VAD, embedder,
and ASR.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .align import assign_words, group_segments
from .diarize import ClusteredConfig
from .types import Region, Segment, Turn, Word
from .windows import merge_regions, slice_windows, windows_to_turns

SAMPLE_RATE = 16_000

# Path-free model signatures, all operating on mono float32 at 16 kHz.
VadFn = Callable[[np.ndarray], list[Region]]
EmbedFn = Callable[[np.ndarray, list[Region]], np.ndarray]
TranscribeFn = Callable[[np.ndarray], list[Word]]


class OnlineSpeakerTracker:
    """Stable speaker ids across chunks via running centroid matching."""

    def __init__(self, distance_threshold: float) -> None:
        self.distance_threshold = distance_threshold
        self.centroids: list[np.ndarray] = []  # unit-normalized
        self.counts: list[int] = []

    def assign(self, embeddings: np.ndarray, local_labels: list[int]) -> list[int]:
        """Map per-window local cluster labels to global speaker ids."""
        mapping: dict[int, int] = {}
        for local in sorted(set(local_labels)):
            rows = embeddings[[i for i, lb in enumerate(local_labels) if lb == local]]
            centroid = rows.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) or 1.0)
            mapping[local] = self._match(centroid, len(rows))
        return [mapping[lb] for lb in local_labels]

    def _match(self, centroid: np.ndarray, weight: int) -> int:
        if self.centroids:
            distances = [1.0 - float(np.dot(centroid, c)) for c in self.centroids]
            best = int(np.argmin(distances))
            if distances[best] < self.distance_threshold:
                # running mean, re-normalized, weighted by windows seen
                n = self.counts[best]
                merged = (self.centroids[best] * n + centroid * weight) / (n + weight)
                self.centroids[best] = merged / (np.linalg.norm(merged) or 1.0)
                self.counts[best] += weight
                return best
        self.centroids.append(centroid)
        self.counts.append(weight)
        return len(self.centroids) - 1


class LiveSession:
    """Buffer incoming PCM and emit speaker-attributed segments per chunk."""

    def __init__(
        self,
        vad_fn: VadFn,
        embed_fn: EmbedFn,
        transcribe_fn: TranscribeFn,
        config: ClusteredConfig | None = None,
        chunk_seconds: float = 5.0,
        centroid_threshold: float = 0.5,
    ) -> None:
        self.vad_fn = vad_fn
        self.embed_fn = embed_fn
        self.transcribe_fn = transcribe_fn
        self.cfg = config or ClusteredConfig()
        self.chunk_samples = int(chunk_seconds * SAMPLE_RATE)
        self.tracker = OnlineSpeakerTracker(centroid_threshold)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._consumed = 0  # samples already processed, for absolute times

    def feed(self, samples: np.ndarray) -> list[Segment]:
        """Append audio; process every full chunk that has accumulated."""
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        out: list[Segment] = []
        while len(self._buffer) >= self.chunk_samples:
            chunk = self._buffer[: self.chunk_samples]
            self._buffer = self._buffer[self.chunk_samples :]
            out.extend(self._process(chunk))
            self._consumed += self.chunk_samples
        return out

    def flush(self) -> list[Segment]:
        """Process whatever remains in the buffer (end of stream)."""
        if not len(self._buffer):
            return []
        chunk, self._buffer = self._buffer, np.zeros(0, dtype=np.float32)
        segments = self._process(chunk)
        self._consumed += len(chunk)
        return segments

    def _process(self, chunk: np.ndarray) -> list[Segment]:
        t0 = self._consumed / SAMPLE_RATE
        raw = self.vad_fn(chunk)
        regions = merge_regions(
            raw, min_gap=self.cfg.min_gap, min_duration=self.cfg.min_duration,
            pad=self.cfg.vad_pad,
        )
        if not regions:
            return []
        # transcribe before touching the tracker: a chunk that yields no
        # words must not mint a phantom speaker id
        words = [
            Word(w.start + t0, w.end + t0, w.text, w.probability)
            for w in self.transcribe_fn(chunk)
        ]
        if not words:
            return []
        windows = slice_windows(regions, window=self.cfg.window, stride=self.cfg.stride)
        embeddings = self.embed_fn(chunk, windows)
        from .cluster import cluster_embeddings

        local = cluster_embeddings(embeddings, distance_threshold=self.cfg.distance_threshold)
        global_labels = self.tracker.assign(
            embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True), local
        )
        turns = [
            Turn(t.start + t0, t.end + t0, t.speaker)
            for t in windows_to_turns(windows, global_labels)
        ]
        return group_segments(assign_words(words, turns))
