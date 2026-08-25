"""W15 probe (throwaway, keep for regression use): measures whether an
energy-based cross-shard routing score closes the accuracy gap between
per-shard validation accuracy (~75-90%, measured per shard on its own
classes) and combined self-routing accuracy (~45-63%, measured across 30
real Optuna trials in tuning/tune.py).

Root cause (see plots.py::_run_sisa_batch): today's routing decision is a
raw softmax-confidence comparison across independently-trained specialists.
Softmax classifiers stay confidently "opinionated" even on inputs from
classes they've never seen, so a specialist often out-shouts the correct
one for inputs that belong to the other shard. Energy-based OOD scores
(Liu et al., NeurIPS 2020, arXiv:2010.03759 -- computed from raw logits via
logsumexp, not softmax) are the literature-established fix for exactly this
discrimination problem (also studied as "task-id inference via OOD
detection" in class-incremental learning, e.g. arXiv:2411.00430).

This script trains real shard models ONCE (reusing tuning/tune.py's pure
per-shard/per-slice loop helpers -- no reimplementation), then evaluates the
SAME trained models with (a) the existing, unmodified plots._run_sisa_batch
(baseline) and (b) a candidate energy-based routing function defined here
(not yet in plots.py, so the idea is proven before any shared code changes).

Usage:
    py -3.8 experiments/energy_routing_probe.py --project-name cifar10_sisa_pytorch --epochs 100
    py -3.8 experiments/energy_routing_probe.py --project-name cifar10_sisa_pytorch --epochs 8   # smoke test
"""
import argparse
import math
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


def _shard_routing_score(owned_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Higher = more in-distribution for this shard. This is the NEGATION of
    Liu et al.'s literal energy E(x) = -T*logsumexp(...), which they define as
    something to minimize -- named "score" here (not "energy") so a future
    reader doesn't flip the sign back into the softmax-overconfidence bug.

    log(k) term de-biases logsumexp's growth with the number of classes
    summed -- required because W6 unlearning shrinks one shard's owned-class
    count relative to the other, which would otherwise hand the untouched
    shard a free, meaningless edge post-unlearning.
    """
    k = owned_logits.shape[1]
    return temperature * (torch.logsumexp(owned_logits / temperature, dim=1) - math.log(k))


def run_sisa_batch_energy(batch_x_normalized, shard_models, class_names, shard_class_indices,
                           temperature=1.0, threshold=None):
    num_classes = len(class_names)
    batch_size = batch_x_normalized.size(0)

    scores = []
    scattered_probs = []
    for shard_idx, model in enumerate(shard_models):
        owned = shard_class_indices[shard_idx] if shard_idx < len(shard_class_indices) else []
        if model is None or not owned:
            scores.append(torch.full((batch_size,), float('-inf'), device=DEVICE))
            scattered_probs.append(torch.zeros(batch_size, num_classes, device=DEVICE))
            continue

        with torch.no_grad():
            specialist_logits = model(batch_x_normalized)

        owned_sorted = sorted(owned)
        is_reduced_head = specialist_logits.shape[1] != num_classes
        owned_logits = specialist_logits if is_reduced_head else specialist_logits[:, owned_sorted]

        scores.append(_shard_routing_score(owned_logits, temperature))
        local_probs = torch.softmax(owned_logits / config.PRIMARY_SPECIALIST_TEMPERATURE, dim=1)
        scattered_probs.append(_scatter_local_to_global(local_probs, owned_sorted, num_classes, DEVICE))

    stacked_scores = torch.stack(scores, dim=0)          # (num_shards, batch)
    winner = stacked_scores.argmax(dim=0)                # (batch,)
    stacked_probs = torch.stack(scattered_probs, dim=0)  # (num_shards, batch, num_classes)
    combined_probabilities = stacked_probs[winner, torch.arange(batch_size, device=DEVICE)]
    top_confidence, final_preds = combined_probabilities.max(dim=1)

    if threshold is not None:
        final_preds = torch.where(top_confidence >= threshold, final_preds, torch.full_like(final_preds, -1))

    return final_preds, combined_probabilities, winner


def evaluate(shard_models, class_names, dataset_mean, dataset_std, validation_data, shard_class_indices,
             temperature=1.0):
    eval_transforms = T.Compose([T.Normalize(dataset_mean, dataset_std)])
    x_val, y_val = validation_data
    for m in shard_models:
        if m is not None:
            m.eval()

    baseline_preds, energy_preds, energy_winners = [], [], []
    with torch.no_grad():
        for k in range(0, len(x_val), config.BATCH_SIZE):
            batch_x = torch.from_numpy(x_val[k:k + config.BATCH_SIZE]).float()
            batch_x = eval_transforms(batch_x).to(DEVICE)

            b_preds, _ = _run_sisa_batch(batch_x, shard_models, class_names, shard_class_indices)
            baseline_preds.extend(b_preds.cpu().numpy())

            e_preds, _, e_winner = run_sisa_batch_energy(
                batch_x, shard_models, class_names, shard_class_indices, temperature=temperature
            )
            energy_preds.extend(e_preds.cpu().numpy())
            energy_winners.extend(e_winner.cpu().numpy())

    baseline_preds = np.array(baseline_preds)
    energy_preds = np.array(energy_preds)
    energy_winners = np.array(energy_winners)

    baseline_acc = float(np.mean(baseline_preds == y_val))
    energy_acc = float(np.mean(energy_preds == y_val))

    # Routing-correctness diagnostic: which shard SHOULD have owned each sample's true label.
    true_owner = np.full(len(y_val), -1)
    for shard_idx, owned in enumerate(shard_class_indices):
        true_owner[np.isin(y_val, owned)] = shard_idx
    energy_routing_acc = float(np.mean(energy_winners == true_owner))

    return baseline_acc, energy_acc, energy_routing_acc


def separability_report(shard_models, dataset_mean, dataset_std, validation_data, shard_class_indices,
                         temperature=1.0):
    eval_transforms = T.Compose([T.Normalize(dataset_mean, dataset_std)])
    x_val, y_val = validation_data

    for shard_idx, model in enumerate(shard_models):
        owned = shard_class_indices[shard_idx]
        owned_sorted = sorted(owned)
        in_scores, out_scores = [], []
        model.eval()
        with torch.no_grad():
            for k in range(0, len(x_val), config.BATCH_SIZE):
                batch_x = torch.from_numpy(x_val[k:k + config.BATCH_SIZE]).float()
                batch_x = eval_transforms(batch_x).to(DEVICE)
                batch_y = y_val[k:k + config.BATCH_SIZE]

                logits = model(batch_x)
                owned_logits = logits if logits.shape[1] == len(owned_sorted) else logits[:, owned_sorted]
                scores = _shard_routing_score(owned_logits, temperature).cpu().numpy()

                is_owned = np.isin(batch_y, owned)
                in_scores.extend(scores[is_owned].tolist())
                out_scores.extend(scores[~is_owned].tolist())

        print(f"  Shard {shard_idx+1} ({len(owned)} owned classes): "
              f"in-distribution score mean/std = {np.mean(in_scores):.3f}/{np.std(in_scores):.3f}  "
              f"out-of-distribution score mean/std = {np.mean(out_scores):.3f}/{np.std(out_scores):.3f}")


def main():
    parser = argparse.ArgumentParser(description="W15 probe: baseline softmax routing vs. energy-based routing")
    parser.add_argument('--project-name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=config.MAX_EPOCHS,
                         help="Per-slice epoch budget (early stopping still applies). Use a small number "
                              "(e.g. 8) only for a quick smoke test of the script itself -- the real "
                              "comparison needs the full budget to be a fair test.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Energy score temperature")
    args = parser.parse_args()

    set_seed(config.SEED)
    print(f"Training both shards, all slices, epochs={args.epochs} (config.MAX_EPOCHS={config.MAX_EPOCHS})...")
    shard_models, class_names, dataset_mean, dataset_std, validation_data, shard_class_indices = \
        train_all_shards(args.project_name, args.epochs)

    print("\n" + "=" * 70)
    print("ROUTING COMPARISON (validation set)")
    print("=" * 70)
    baseline_acc, energy_acc, energy_routing_acc = evaluate(
        shard_models, class_names, dataset_mean, dataset_std, validation_data, shard_class_indices,
        temperature=args.temperature,
    )
    print(f"Baseline (softmax-max routing) accuracy:      {baseline_acc:.4f}")
    print(f"Candidate (energy-based routing) accuracy:    {energy_acc:.4f}")
    print(f"Candidate: % samples routed to correct shard:  {energy_routing_acc:.4f}")

    print("\nPer-shard routing-score separability (higher gap = better OOD discrimination):")
    separability_report(shard_models, dataset_mean, dataset_std, validation_data, shard_class_indices,
                         temperature=args.temperature)


if __name__ == "__main__":
    main()
