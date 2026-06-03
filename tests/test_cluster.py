"""Clustering on synthetic embeddings with known structure."""

import numpy as np

from diarlab.cluster import cluster_embeddings


def _two_blobs(n_per: int = 6, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Two tight clusters around orthogonal unit vectors (cosine distance ~1)."""
    rng = np.random.default_rng(seed)
    a = np.zeros(dim)
    a[0] = 1.0
    b = np.zeros(dim)
    b[1] = 1.0
    rows = [a + rng.normal(0, 0.01, dim) for _ in range(n_per)]
    rows += [b + rng.normal(0, 0.01, dim) for _ in range(n_per)]
    return np.asarray(rows)


def test_empty_and_singleton():
    assert cluster_embeddings(np.zeros((0, 8))) == []
    assert cluster_embeddings(np.ones((1, 8))) == [0]


def test_two_separated_blobs_found_without_speaker_count():
    X = _two_blobs()
    labels = cluster_embeddings(X, distance_threshold=0.6)
    assert labels[:6] == [0] * 6
    assert labels[6:] == [1] * 6


def test_forced_speaker_count():
    X = _two_blobs()
    assert set(cluster_embeddings(X, num_speakers=2)) == {0, 1}
    assert set(cluster_embeddings(X, num_speakers=1)) == {0}


def test_labels_appear_in_time_order():
    # first row must always get label 0, whatever fcluster numbers internally
    X = _two_blobs()
    labels = cluster_embeddings(X[::-1], distance_threshold=0.6)
    assert labels[0] == 0
