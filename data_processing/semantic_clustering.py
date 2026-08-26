"""W17/W26: cluster classes by WordNet semantic similarity so shards are
visually/semantically distinct from each other -- makes the gating network's
routing decision easier by construction, targeting the bottleneck W16 measured
(routing accuracy, not per-shard classifier quality, caps combined accuracy).
Zero training, zero image data touched -- a one-time offline lookup over class
names, done during data processing before any shard/slice training exists.

W26 rewrote name resolution after an audit found the original was badly broken
outside CIFAR-10:

  * ImageNet-family wnids ("n01443537") did not resolve at all, so every
    similarity fell back to the neutral 0.5 and clustering became arbitrary --
    silently. On Tiny-ImageNet all 200 classes failed this way.
  * The `synsets(name)[0]` first-sense heuristic picked the wrong sense for
    ~45% of ambiguous CIFAR-100 names, often landing in the wrong branch
    entirely: "shrew" -> a bad-tempered woman, "skunk" -> a despicable person,
    "seal" -> sealing wax, "ray" -> a beam of light.
  * Space-separated names never reached the compound fallback.
  * Placeholder names resolved to nonsense ("class_0" -> the number zero).
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from sklearn.cluster import AgglomerativeClustering

# A WordNet offset id as used by ImageNet / Tiny-ImageNet directory names.
_WNID_RE = re.compile(r'^n(\d{8})$')

# Senses that are almost never what a vision dataset means. `person` matters most:
# in WordNet a person IS an organism, so preferring "organism" is not enough to
# rule out "shrew = bad-tempered woman" -- people must be demoted explicitly.
_EXCLUDED_ROOTS = frozenset({
    'person.n.01', 'abstraction.n.06', 'communication.n.02',
    'attribute.n.02', 'state.n.02', 'act.n.02', 'measure.n.02',
})

# Preferred concrete branches, most-specific first.
_PREFERRED_ROOTS = ('animal.n.01', 'plant.n.02', 'food.n.01',
                    'artifact.n.01', 'natural_object.n.01')

# Names that resolve to something real but meaningless for a dataset -- treated as
# unresolved so they contribute a neutral similarity rather than active nonsense.
_PLACEHOLDER_RE = re.compile(r'^(class|label|category|cat|c)[\s_-]*\d+$', re.IGNORECASE)


def _hypernym_roots(synset) -> frozenset:
    return frozenset(h.name() for path in synset.hypernym_paths() for h in path)


def _rank_sense(synset) -> Tuple[int, int]:
    """Sort key preferring concrete visual senses. Exclusion is checked FIRST so a
    person-sense can never win merely by also being an organism."""
    roots = _hypernym_roots(synset)
    excluded = 1 if (roots & _EXCLUDED_ROOTS) else 0
    rank = next((i for i, r in enumerate(_PREFERRED_ROOTS) if r in roots), len(_PREFERRED_ROOTS))
    return (excluded, rank)


def _resolve_synset(class_name: str) -> Optional[object]:
    """Best-effort WordNet noun synset for a class name.

    Resolution ladder, most reliable first:
      1. An ImageNet wnid resolves exactly, with no sense ambiguity at all.
      2. The literal name, with spaces normalised to underscores (WordNet's own
         separator -- the original code converted the other way and so missed
         multi-word names entirely).
      3. The compound's head word, then its first word ("aquarium_fish" -> "fish").
    Among candidate senses, concrete visual ones win over person/abstract ones.
    Returns None when nothing usable resolves.
    """
    if not class_name or not class_name.strip():
        return None
    if _PLACEHOLDER_RE.match(class_name.strip()):
        return None

    wnid = _WNID_RE.match(class_name.strip())
    if wnid:
        try:
            return wn.synset_from_pos_and_offset('n', int(wnid.group(1)))
        except Exception:
            return None

    normalized = class_name.strip().replace(' ', '_').replace('-', '_')
    candidates = [normalized, class_name.strip()]
    parts = [p for p in normalized.split('_') if p]
    if len(parts) > 1:
        candidates += [parts[-1], parts[0]]

    for candidate in candidates:
        senses = wn.synsets(candidate, pos=wn.NOUN)
        if senses:
            return sorted(senses, key=_rank_sense)[0]
    return None


def resolve_class_synsets(class_names: Sequence[str]) -> Tuple[List[Optional[object]], Dict[str, str]]:
    """Resolve every class name, returning the synsets plus a name->synset map of
    what resolved (for recording in metadata, so a partition stays auditable)."""
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    synsets = [_resolve_synset(name) for name in class_names]
    resolved = {n: s.name() for n, s in zip(class_names, synsets) if s is not None}
    return synsets, resolved


def compute_semantic_similarity_matrix(class_names: Sequence[str],
                                       synsets: Optional[List] = None) -> np.ndarray:
    """NxN Wu-Palmer similarity over class names. Pairs involving an unresolved
    name get a neutral 0.5 rather than erroring, so one odd name degrades
    gracefully instead of blocking the run."""
    if synsets is None:
        synsets, _ = resolve_class_synsets(class_names)

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


def cluster_classes_semantically(class_names: Sequence[str], num_clusters: int,
                                 min_resolution_rate: float = 0.8) -> List[List[int]]:
    """Group class indices into `num_clusters` clusters by WordNet similarity.

    Clusters may be uneven -- a natural semantic split (CIFAR-10's 4 vehicles vs
    6 animals) is not forced to be even.

    Raises ValueError rather than returning a meaningless partition when the
    inputs cannot support clustering. That matters: before W26, a dataset whose
    names did not resolve produced an all-0.5 similarity matrix and an arbitrary
    split such as [8, 1, 1] with no warning at all. Callers (sharding.py) already
    catch this and fall back to balance-based assignment.
    """
    n = len(class_names)
    if n < 2:
        raise ValueError(f"Semantic clustering needs at least 2 classes, got {n}")
    if num_clusters < 2:
        raise ValueError(f"num_clusters must be at least 2, got {num_clusters}")
    if num_clusters > n:
        raise ValueError(f"Cannot form {num_clusters} clusters from {n} classes")

    synsets, resolved = resolve_class_synsets(class_names)
    rate = len(resolved) / n
    if rate < min_resolution_rate:
        raise ValueError(
            f"Only {len(resolved)}/{n} class names ({rate:.0%}) resolved to WordNet synsets, "
            f"below the {min_resolution_rate:.0%} threshold. Semantic distances would be "
            f"mostly the neutral fallback, making the partition arbitrary."
        )
    if rate < 1.0:
        unresolved = [c for c, s in zip(class_names, synsets) if s is None]
        print(f"   - WARNING: {n - len(resolved)} class name(s) did not resolve in WordNet "
              f"and will use neutral similarity: {unresolved}")

    similarity = compute_semantic_similarity_matrix(class_names, synsets)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)

    clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(distance)

    clusters: List[List[int]] = [[] for _ in range(num_clusters)]
    for class_idx, cluster_label in enumerate(labels):
        clusters[cluster_label].append(class_idx)
    return clusters
