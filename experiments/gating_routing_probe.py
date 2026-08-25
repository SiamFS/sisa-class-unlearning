"""W15 follow-up probe (throwaway, keep for regression use): measures whether
using the ALREADY-EXISTING GatingNetwork (training/create_model.py,
train_gating in training/train_gating_model.py) as the REAL routing decision
-- instead of only as a diagnostic, which is its current role in
entry_training.py -- closes the accuracy gap the energy-routing probe
(experiments/energy_routing_probe.py) failed to close.

Context: the project's own IMPLEMENTATION_PLAN.md (S3.2/W3) already names
this as the documented fallback if self-routing accuracy is insufficient:
"retrain the lightweight gating net on remaining classes each deletion...
still efficient... mirrors RecEraser and is exact, but adds a retrained
component." A standalone run of train_gating() on this project measured
87.74% shard-routing validation accuracy -- much higher than the confidence-
based self-routing mechanism has been achieving end-to-end (45-63% across
30 real Optuna trials). This script measures the actual combined SISA
accuracy if that gating network's prediction is used to pick the shard,
instead of picking via cross-shard confidence/energy comparison.

Requires projects/<name>/models/gating_model.pth to already exist (run
training/train_gating_model.py's train_gating(), or this project's
entry_training.py, first).

Usage:
    py -3.8 experiments/gating_routing_probe.py --project-name cifar10_sisa_pytorch --epochs 100
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchvision.transforms as T

import config
from utils.seeding import set_seed
from training.train_model import train_model
from training.replay_buffer import add_to_replay_buffer
from training.create_model import load_model_pytorch
from plots import load_shard_class_indices, _run_sisa_batch, _scatter_local_to_global
from tuning.tune import (
    _load_project,
    _load_slice,
    _cumulative_classes,
    _shard_validation,
    _augmentation_for_shard,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_all_shards(project_name: str, epochs: int):
    (sisa_data_dir, num_shards, num_slices, class_names, dataset_mean, dataset_std,
     shard_metadatas, validation_data) = _load_project(project_name)

    slice_classes_cache = {}
    augmentation_by_shard = [
        _augmentation_for_shard(sisa_data_dir, i, num_slices) for i in range(num_shards)
    ]

    shard_models = []
    for i in range(num_shards):
        current_model = None
        replay_buffer, replay_seen_counts = {}, {}
        replay_rng = np.random.default_rng(config.SEED)

        for j in range(num_slices):
            x_slice, y_slice = _load_slice(sisa_data_dir, i, j)
            if x_slice is None or len(x_slice) == 0:
                continue

            known_classes = _cumulative_classes(sisa_data_dir, i, j, slice_classes_cache)
            val_data = _shard_validation(shard_metadatas[i], known_classes, validation_data)

            print(f"\n--- Shard {i+1}/{num_shards}, Slice {j+1}/{num_slices} ---")
            current_model, _ = train_model(
                x_slice, y_slice, model=current_model,
                epochs=epochs, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
                validation_data=val_data, active_classes=known_classes,
                replay_buffer=replay_buffer, replay_ratio=config.REPLAY_RATIO,
                dataset_mean=dataset_mean, dataset_std=dataset_std,
                training_type='incremental' if current_model is not None else 'fresh',
                augmentation_config=augmentation_by_shard[i], device=DEVICE,
            )
            add_to_replay_buffer(replay_buffer, x_slice, y_slice, replay_rng,
                                  config.MAX_REPLAY_SAMPLES_PER_CLASS, replay_seen_counts)

        shard_models.append(current_model)

    return shard_models, class_names, dataset_mean, dataset_std, validation_data, \
        load_shard_class_indices(sisa_data_dir, num_shards)


def run_sisa_batch_gating(batch_x_normalized, shard_models, gating_model, class_names, shard_class_indices):
    num_classes = len(class_names)

    with torch.no_grad():
        gating_logits = gating_model(batch_x_normalized)
    predicted_shard = gating_logits.argmax(dim=1)  # (batch,)

    per_shard_probs = []
    for shard_idx, model in enumerate(shard_models):
        owned = shard_class_indices[shard_idx] if shard_idx < len(shard_class_indices) else []
        owned_sorted = sorted(owned)
        with torch.no_grad():
            specialist_logits = model(batch_x_normalized)
        is_reduced_head = specialist_logits.shape[1] != num_classes
        owned_logits = specialist_logits if is_reduced_head else specialist_logits[:, owned_sorted]
        local_probs = torch.softmax(owned_logits / config.PRIMARY_SPECIALIST_TEMPERATURE, dim=1)
        per_shard_probs.append(_scatter_local_to_global(local_probs, owned_sorted, num_classes, DEVICE))

    stacked_probs = torch.stack(per_shard_probs, dim=0)  # (num_shards, batch, num_classes)
    batch_size = batch_x_normalized.size(0)
    combined_probabilities = stacked_probs[predicted_shard, torch.arange(batch_size, device=DEVICE)]
    final_preds = combined_probabilities.argmax(dim=1)
    return final_preds, combined_probabilities, predicted_shard


def evaluate(shard_models, gating_model, class_names, dataset_mean, dataset_std, validation_data,
             shard_class_indices):
    eval_transforms = T.Compose([T.Normalize(dataset_mean, dataset_std)])
    x_val, y_val = validation_data
    for m in shard_models:
        if m is not None:
            m.eval()
    gating_model.eval()

    baseline_preds, gating_preds, gating_winners = [], [], []
    with torch.no_grad():
        for k in range(0, len(x_val), config.BATCH_SIZE):
            batch_x = torch.from_numpy(x_val[k:k + config.BATCH_SIZE]).float()
            batch_x = eval_transforms(batch_x).to(DEVICE)

            b_preds, _ = _run_sisa_batch(batch_x, shard_models, class_names, shard_class_indices)
            baseline_preds.extend(b_preds.cpu().numpy())

            g_preds, _, g_winner = run_sisa_batch_gating(
                batch_x, shard_models, gating_model, class_names, shard_class_indices
            )
            gating_preds.extend(g_preds.cpu().numpy())
            gating_winners.extend(g_winner.cpu().numpy())

    baseline_preds = np.array(baseline_preds)
    gating_preds = np.array(gating_preds)
    gating_winners = np.array(gating_winners)

    baseline_acc = float(np.mean(baseline_preds == y_val))
    gating_acc = float(np.mean(gating_preds == y_val))

    true_owner = np.full(len(y_val), -1)
    for shard_idx, owned in enumerate(shard_class_indices):
        true_owner[np.isin(y_val, owned)] = shard_idx
    gating_routing_acc = float(np.mean(gating_winners == true_owner))

    return baseline_acc, gating_acc, gating_routing_acc


def main():
    parser = argparse.ArgumentParser(description="W15 follow-up probe: gating-network-based routing vs baseline")
    parser.add_argument('--project-name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=config.MAX_EPOCHS)
    args = parser.parse_args()

    base_dir = os.path.join(config.PROJECTS_DIR, args.project_name)
    gating_path = os.path.join(base_dir, "models", "gating_model.pth")
    if not os.path.exists(gating_path):
        raise SystemExit(f"No gating model at {gating_path} -- run train_gating() first.")

    set_seed(config.SEED)
    print(f"Training both shards, all slices, epochs={args.epochs} (config.MAX_EPOCHS={config.MAX_EPOCHS})...")
    shard_models, class_names, dataset_mean, dataset_std, validation_data, shard_class_indices = \
        train_all_shards(args.project_name, args.epochs)

    num_shards = len(shard_models)
    gating_model, _ = load_model_pytorch(gating_path, num_shards=num_shards)
    gating_model.eval()

    print("\n" + "=" * 70)
    print("ROUTING COMPARISON (validation set)")
    print("=" * 70)
    baseline_acc, gating_acc, gating_routing_acc = evaluate(
        shard_models, gating_model, class_names, dataset_mean, dataset_std, validation_data, shard_class_indices,
    )
    print(f"Baseline (softmax-max self-routing) accuracy: {baseline_acc:.4f}")
    print(f"Candidate (gating-network routing) accuracy:  {gating_acc:.4f}")
    print(f"Candidate: % samples routed to correct shard:  {gating_routing_acc:.4f}")


if __name__ == "__main__":
    main()
