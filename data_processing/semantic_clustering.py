"""W17: cluster classes by WordNet semantic similarity so shards are
visually/semantically distinct from each other -- makes the gating
network's routing decision (which shard does this image belong to) easier
by construction, directly targeting the bottleneck W16 measured (88.08%
routing accuracy, not per-shard classifier quality, was capping combined
accuracy). Zero training, zero image data touched -- purely a one-time,
offline lookup over the real class name strings, done once during data
processing before any shard/slice training exists.
"""
from typing import List, Optional

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from sklearn.cluster import AgglomerativeClustering


def _resolve_synset(class_name: str):
    """First WordNet noun synset found for a class name, trying the literal
    name, then underscore-as-space, then just its last/first word (handles
    compound names like CIFAR-100's "aquarium_fish" -> "fish", which has no
    direct synset of its own). None if nothing resolves."""
    candidates = [class_name, class_name.replace('_', ' ')]
    parts = class_name.replace('-', '_').split('_')
    if len(parts) > 1:
        candidates.append(parts[-1])
        candidates.append(parts[0])

    for candidate in candidates:
        synsets = wn.synsets(candidate, pos=wn.NOUN)
        if synsets:
            return synsets[0]
    return None


def compute_semantic_similarity_matrix(class_names: List[str]) -> np.ndarray:
    """NxN Wu-Palmer similarity matrix over class names. A pair where either
    name's synset didn't resolve gets a neutral 0.5 similarity rather than
    erroring -- degrades gracefully for one odd class name instead of
    blocking the whole run."""
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    synsets = [_resolve_synset(name) for name in class_names]
    n = len(class_names)
    similarity = np.full((n, n), 0.5)
    np.fill_diagonal(similarity, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            if synsets[i] is not None and synsets[j] is not None:
                score = synsets[i].wup_similarity(synsets[j])
                if score is not None:
                    similarity[i, j] = similarity[j, i] = score

    return similarity


def cluster_classes_semantically(class_names: List[str], num_clusters: int) -> List[List[int]]:
    """Group class indices into num_clusters clusters by WordNet semantic
    similarity (agglomerative clustering over a precomputed distance matrix).
    Returns one list of class indices per cluster -- clusters may be uneven
    in size (a natural semantic split, e.g. CIFAR-10's 4 vehicles vs 6
    animals, isn't forced to be even)."""
    similarity = compute_semantic_similarity_matrix(class_names)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)

    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(distance)

    clusters: List[List[int]] = [[] for _ in range(num_clusters)]
    for class_idx, cluster_label in enumerate(labels):
        clusters[cluster_label].append(class_idx)
    return clusters
