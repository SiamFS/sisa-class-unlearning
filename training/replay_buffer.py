"""Simple, seeded, per-class replay buffer (W4).

A plain dict of {class_id: {'X': ndarray, 'y': ndarray}}, capped per class via
true reservoir sampling (Algorithm R, Vitter 1985) so every sample ever seen
for a class has equal probability min(1, cap/n_seen) of surviving to the
final buffer, independent of when it arrived. This matters here specifically:
a naive "combine everything and uniformly resample down to the cap on every
call" approach (an earlier version of this file) gives later-arriving slices'
samples a systematic edge, since older samples have to survive one resample
per subsequent call while newer ones only have to survive their own -- the
opposite of what a replay buffer protecting older classes from forgetting
should do.
"""
import numpy as np

import config


def compute_replay_ratio(replay_buffer: dict, y_slice: np.ndarray) -> float:
    """Fraction of each training batch that should come from replay (W18).

    A fixed ratio cannot balance a class-incremental sequence. With the old
    `REPLAY_RATIO = 0.3` and `batch_size = 64`, shard 2's slice 5 (frog + horse
    current, five older classes in the buffer) allocated ~36.7 slots to horse and
    exactly 4 to each older class -- a 9:1 imbalance in every batch, which is what
    produced the measured task-recency bias.

    Under `REPLAY_RATIO_MODE = 'balanced'` the ratio instead follows the class split,
    `n_old / (n_old + n_new)`, which equalises each *class*'s expected share of the
    batch rather than each *slice*'s. At that same slice 5 this yields 5/6 ~= 0.83.

    Deterministic by construction: a pure function of how many classes are in the
    buffer and in the current slice, consuming no RNG, so W1 determinism and the
    exactness proof are unaffected. Both the initial training loop and the unlearning
    retrain path must call this -- if they disagree, the W8 scratch-reference
    comparison is no longer apples-to-apples.
    """
    if not replay_buffer:
        return 0.0

    if getattr(config, 'REPLAY_RATIO_MODE', 'static') != 'balanced':
        return float(config.REPLAY_RATIO)

    n_old = len(replay_buffer)
    n_new = int(len(np.unique(y_slice))) if y_slice is not None and len(y_slice) else 0
    if n_new == 0:
        return float(min(config.REPLAY_RATIO_MAX, 1.0))

    ratio = n_old / (n_old + n_new)
    return float(min(ratio, config.REPLAY_RATIO_MAX))


def add_to_replay_buffer(replay_buffer: dict, x_data: np.ndarray, y_data: np.ndarray,
                          rng: np.random.Generator, max_per_class: int,
                          seen_counts: dict) -> dict:
    """Add samples to `replay_buffer` in place, capping each class at `max_per_class`
    via per-class reservoir sampling.

    `seen_counts` tracks, per class, how many samples have ever been offered to
    the buffer (not just how many are currently stored) -- required state for
    correct reservoir probabilities. Callers must create a fresh `{}` for both
    `replay_buffer` and `seen_counts` together at the start of a buffer's
    lifecycle, and thread the same two dicts through every call for that
    lifecycle (mirrors how `replay_buffer` itself already works).
    """
    for label in np.unique(y_data):
        mask = (y_data == label)
        new_x, new_y = x_data[mask], y_data[mask]

        if label not in replay_buffer:
            replay_buffer[label] = {
                'X': np.empty((0,) + x_data.shape[1:], dtype=x_data.dtype),
                'y': np.empty((0,), dtype=y_data.dtype),
            }
            seen_counts[label] = 0

        buf = replay_buffer[label]
        n_seen = seen_counts[label]

        # Phase 1: buffer not yet full -- bulk-append whatever fits in the
        # remaining free space. Nothing is being evicted yet, so no reservoir
        # math is needed for this portion.
        space_left = max_per_class - len(buf['y'])
        if space_left > 0:
            take = min(space_left, len(new_y))
            buf['X'] = np.concatenate([buf['X'], new_x[:take]])
            buf['y'] = np.concatenate([buf['y'], new_y[:take]])
            n_seen += take
            new_x, new_y = new_x[take:], new_y[take:]

        # Phase 2: buffer full -- classic Algorithm R, one item at a time,
        # O(1) each via in-place slot replacement.
        for i in range(len(new_y)):
            n_seen += 1
            j = rng.integers(0, n_seen)  # uniform in [0, n_seen-1]
            if j < max_per_class:
                buf['X'][j] = new_x[i]
                buf['y'][j] = new_y[i]

        seen_counts[label] = n_seen

    return replay_buffer
