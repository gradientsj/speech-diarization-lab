"""Agglomerative clustering of speaker embeddings.

Average-linkage clustering over cosine distance, the standard recipe for
offline diarization. The number of speakers is either given or chosen by
cutting the dendrogram at a distance threshold; the threshold is the one
real hyperparameter in the from-parts pipeline and is calibrated on the
synthetic benchmark rather than guessed.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float = 0.6,
    num_speakers: int | None = None,
) -> list[int]:
    """Label each embedding row with a cluster id (0-based, by first appearance).

    With `num_speakers` set, the dendrogram is cut to exactly that many
    clusters; otherwise it is cut at `distance_threshold` cosine distance,
    which lets the speaker count fall out of the data.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    distances = pdist(normed, metric="cosine")
    tree = linkage(distances, method="average")
    if num_speakers is not None:
        raw = fcluster(tree, t=max(1, num_speakers), criterion="maxclust")
    else:
        raw = fcluster(tree, t=distance_threshold, criterion="distance")

    # relabel so cluster ids appear in time order: stable output, easy to test
    remap: dict[int, int] = {}
    labels: list[int] = []
    for r in raw:
        if r not in remap:
            remap[r] = len(remap)
        labels.append(remap[r])
    return labels
