"""W8: build a true, independent from-scratch reference for one shard (plan
IMPLEMENTATION_PLAN.md's headline deliverable). Never touches the source
project -- everything happens on a throwaway copy.

Why a full retrain from slice 1, not a checkpoint-reuse shortcut: see
IMPLEMENTATION_PLAN.md decision 3.3-adjacent discussion / the W8 section.
Short version -- a shortcut that reuses the pre-deletion checkpoint and
replays forward is the *same code path* real unlearning already runs, so
comparing it to itself can't catch a bug that lives in that shared code
(replay filtering, head-resizing, ...). An independently-trained witness
that never executes that machinery is what can actually cross-validate it.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import config
from utils.seeding import set_seed
from utils.data_io import load_images
from utils.project_io import clone_project
from training.create_model import save_model_pytorch, DEVICE
from training.train_model import train_model
from training.replay_buffer import add_to_replay_buffer, compute_replay_ratio
from unlearning.sisa_unlearning import SISAUnlearning


# W34: the clone helpers now live in utils/project_io.py, because unlearning test mode
# (config.UNLEARNING_TEST_MODE) needs them too and importing them from here would be a
# circular import (this module imports SISAUnlearning). `_copy_project` is kept as the
# public name because experiments/exactness_eval.py imports it from here.
def _copy_project(source_project: str, dest_project: str) -> str:
    return clone_project(source_project, dest_project)


def snapshot_class_training_samples(source_project: str, model_name: str, class_name: str):
    """The class's real training samples, read from the PRISTINE source project
    before anything is copied or modified. This is the MIA "member" set --
    once real unlearning or the scratch strip runs, these rows are gone from disk."""
    unlearner = SISAUnlearning(source_project, model_name)
    class_idx = unlearner.class_names.index(class_name)
    xs, ys = [], []
    for shard_idx in range(unlearner.num_shards):
        for slice_idx in range(unlearner.num_slices):
            X, y = unlearner._load_slice_data(shard_idx, slice_idx)
            if X is None:
                continue
            mask = (y == class_idx)
            if mask.any():
                xs.append(X[mask])
                ys.append(y[mask])
    if not xs:
        raise ValueError(f"No training samples found for class '{class_name}' in '{source_project}'")
    return np.concatenate(xs), np.concatenate(ys)


def find_class_shard(source_project: str, model_name: str, class_name: str) -> int:
    unlearner = SISAUnlearning(source_project, model_name)
    class_idx = unlearner.class_names.index(class_name)
    for shard_idx in range(unlearner.num_shards):
        with open(os.path.join(unlearner.data_dir, f"shards/shard_{shard_idx+1}/metadata.json")) as f:
            meta = json.load(f)
        if class_idx in meta.get('class_indices_present', []):
            return shard_idx
    raise ValueError(f"Class '{class_name}' not found in any shard of '{source_project}'")


def strip_class_from_scratch_copy(scratch_project: str, model_name: str, class_name: str, shard_idx: int) -> None:
    """Remove every row of `class_name` from `shard_idx`'s slices, on the scratch
    copy only. Reuses SISAUnlearning's own (already-tested) removal logic rather
    than reimplementing row-stripping, so boundary-sharing slices are handled
    identically to real unlearning."""
    unlearner = SISAUnlearning(scratch_project, model_name)
    class_idx = unlearner.class_names.index(class_name)
    for slice_idx in range(unlearner.num_slices):
        unlearner._remove_data_from_slice(shard_idx, slice_idx, class_idx)
    unlearner._update_shard_metadata(shard_idx)


def train_shard_from_scratch(scratch_project: str, model_name: str, shard_idx: int,
                              dataset_mean, dataset_std, num_slices: int):
    """Mirrors training/entry_training.py's per-shard incremental loop, scoped to
    just this one shard, with a head fixed from the start (W6-style) to the
    shard's final (already class-c-stripped) owned-class set, and validation
    filtered to that same set throughout (mirrors get_true_label_validation_data
    -- since shard metadata here already excludes the deleted class, no special
    casing is needed, the existing convention just works).
    """
    set_seed(config.SEED)
    base_dir = os.path.join(config.PROJECTS_DIR, scratch_project)
    sisa_data_dir = os.path.join(base_dir, "sisa_data")
    shard_dir = os.path.join(base_dir, "models", f"shard_{shard_idx+1}")
    os.makedirs(shard_dir, exist_ok=True)

    with open(os.path.join(sisa_data_dir, f"shards/shard_{shard_idx+1}/metadata.json")) as f:
        shard_meta = json.load(f)
    head_classes = sorted(shard_meta['class_indices_present'])
    print(f"   - Scratch shard {shard_idx+1}: training with a fixed {len(head_classes)}-class head from slice 1")

    validation_data = (
        load_images(os.path.join(sisa_data_dir, "validation_data/x_validation.npy")),
        np.load(os.path.join(sisa_data_dir, "validation_data/y_validation.npy")),
    )

    def load_slice(slice_idx):
        x_path = os.path.join(sisa_data_dir, f"shards/shard_{shard_idx+1}/slice_{slice_idx}_x.npy")
        if not os.path.exists(x_path):
            return None, None
        X = load_images(x_path)
        if X.size == 0:
            return None, None
        return X, np.load(x_path.replace('_x.npy', '_y.npy'))

    def cumulative_classes_up_to(slice_idx):
        classes = set()
        for s in range(slice_idx + 1):
            _, y_s = load_slice(s)
            if y_s is not None:
                classes.update(np.unique(y_s).tolist())
        return sorted(classes)

    current_model = None
    replay_buffer, replay_seen_counts = {}, {}
    replay_rng = np.random.default_rng(config.SEED)
    shard_histories = []
    train_start = time.time()

    for slice_idx in range(num_slices):
        x_slice, y_slice = load_slice(slice_idx)
        if x_slice is None:
            print(f"   - Slice {slice_idx+1} is empty, skipping.")
            continue

        known_classes = cumulative_classes_up_to(slice_idx)
        val_mask = np.isin(validation_data[1], list(set(head_classes) & set(known_classes)))
        x_val_f, y_val_f = validation_data[0][val_mask], validation_data[1][val_mask]
        print(f"   - Scratch slice {slice_idx+1}: {len(x_slice)} samples, {len(known_classes)} classes known so far, "
              f"{len(x_val_f)} validation samples")

        current_model, history = train_model(
            x_slice, y_slice, model=current_model,
            epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
            validation_data=(x_val_f, y_val_f), active_classes=known_classes,
            head_classes=head_classes,
            # W31: the scratch reference MUST train under the SAME recipe as the real
            # pipeline, or the exactness comparison measures a recipe difference rather
            # than the effect of unlearning. Two mismatches were present:
            #   * a STATIC config.REPLAY_RATIO (0.3) while training and unlearning both
            #     derive the ratio per slice (W18);
            #   * augmentation_config=None while balanced shards get baseline crop/flip/
            #     cutout (W19/W30).
            replay_buffer=replay_buffer,
            replay_ratio=compute_replay_ratio(replay_buffer, y_slice),
            dataset_mean=dataset_mean, dataset_std=dataset_std,
            training_type='fresh' if current_model is None else 'incremental',
            augmentation_config=config.get_augmentation_config('baseline'), device=DEVICE,
        )
        shard_histories.append(history)

        save_model_pytorch(
            current_model, os.path.join(shard_dir, f"slice_{slice_idx}_model_{model_name}.pth"),
            metadata={'shard_id': shard_idx, 'slice_id': slice_idx},
        )
        add_to_replay_buffer(replay_buffer, x_slice, y_slice, replay_rng,
                              config.MAX_REPLAY_SAMPLES_PER_CLASS, replay_seen_counts)

    pure_train_time = time.time() - train_start
    final_model_path = os.path.join(shard_dir, f"final_model_shard{shard_idx+1}_{model_name}.pth")
    save_model_pytorch(current_model, final_model_path)
    print(f"   - Scratch shard training complete in {pure_train_time:.2f}s (pure), saved to {final_model_path}")

    return current_model, pure_train_time, shard_histories


def build_scratch_reference(source_project: str, class_name: str, model_name: str = None,
                             scratch_project: str = None):
    """Full orchestration: snapshot MIA members from the pristine source, copy it,
    strip the class from the copy, train that shard from scratch. Returns a dict
    with everything exactness_eval.py needs."""
    model_name = model_name or config.MODEL_TYPE
    scratch_project = scratch_project or f"{source_project}_scratch_{class_name}"

    print(f"\n{'='*70}\nBuilding scratch reference for class '{class_name}' (source: {source_project})\n{'='*70}")

    member_x, member_y = snapshot_class_training_samples(source_project, model_name, class_name)
    print(f"   - Snapshotted {len(member_y)} real training samples of '{class_name}' for MIA (before any modification)")

    shard_idx = find_class_shard(source_project, model_name, class_name)
    print(f"   - Class '{class_name}' lives in shard {shard_idx+1}")

    _copy_project(source_project, scratch_project)
    strip_class_from_scratch_copy(scratch_project, model_name, class_name, shard_idx)

    with open(os.path.join(config.PROJECTS_DIR, scratch_project, "sisa_data", "metadata.json")) as f:
        metadata = json.load(f)

    scratch_model, pure_train_time, histories = train_shard_from_scratch(
        scratch_project, model_name, shard_idx,
        metadata['normalization_mean'], metadata['normalization_std'],
        # W31: this shard's own slice count -- shards differ under S_k = |C_k|.
        (metadata.get('slices_per_shard') or [metadata['num_slices']] * metadata['num_shards'])[shard_idx],
    )

    return {
        'scratch_project': scratch_project,
        'shard_idx': shard_idx,
        'scratch_model': scratch_model,
        'pure_train_time': pure_train_time,
        'member_x': member_x,
        'member_y': member_y,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a from-scratch reference shard for W8 exactness proof")
    parser.add_argument('--project-name', type=str, required=True)
    parser.add_argument('--class-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, default=config.MODEL_TYPE)
    args = parser.parse_args()

    result = build_scratch_reference(args.project_name, args.class_name, args.model_name)
    print(f"\nScratch reference ready at project '{result['scratch_project']}' "
          f"(shard {result['shard_idx']+1}, {result['pure_train_time']:.2f}s pure training time)")
