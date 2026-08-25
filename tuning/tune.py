"""W13: one-time Optuna hyperparameter search. NOT part of any runtime path --
nothing else in the codebase imports optuna or this file. Run manually, once,
on validation data only; take the winning trial's values and freeze them into
config.py by hand (with a provenance comment referencing best_config.json).

Searches: LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING, REPLAY_RATIO,
BATCH_SIZE, FC_LAYER_DROPOUT (the set config.py's own "Config parameter
guidance" section names as Optuna-tunable). Uses Lane A's simple, seeded
replay (W4's default) -- never SmartReplayBuffer (removed entirely, and was
never deterministic).

Each trial runs the ENTIRE SISA pipeline: every shard, every slice, real
config.MAX_EPOCHS budget with early stopping, balance-based augmentation --
mirroring entry_training.py's per-shard/per-slice loop exactly (this file
does not import entry_training.py, which executes top-level script code on
import; the loop is duplicated here the same way experiments/scratch_reference.py
duplicates a trimmed copy of it, rather than risking a shared-module refactor
of the real training script). The gating network is NOT trained per trial --
it is diagnostic-only and never affects self-routing predictions or the score
being optimized, so training it per trial would only burn time for no signal.
The objective is full-system self-routing accuracy (plots._run_sisa_batch)
evaluated on the VALIDATION set only -- the test set is never touched here.

This is expensive: 2 shards x 5 slices = 10 incremental trainings per trial,
each with early stopping (patience=config.TRAINING_PATIENCE) against up to
config.MAX_EPOCHS. A 30-trial study is effectively 30 full training runs back
to back and can take many hours. Start with a small --n-trials to confirm
the per-trial wall-clock time before committing to a long study.

Usage:
    py -3.8 tuning/tune.py --project-name cifar10_sisa_pytorch --n-trials 30

Requires data processing to have already been run for --project-name.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchvision.transforms as T
import optuna
from optuna.samplers import TPESampler

import config
from utils.seeding import set_seed
from training.train_model import train_model, _run_sisa_batch
from training.replay_buffer import add_to_replay_buffer
from plots import load_shard_class_indices
from utils.run_logging import setup_run_logging


def _load_project(project_name: str):
    sisa_data_dir = os.path.join(config.PROJECTS_DIR, project_name, "sisa_data")
    with open(os.path.join(sisa_data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_shards = metadata['num_shards']
    num_slices = metadata['num_slices']
    class_names = metadata['class_names']
    dataset_mean = metadata['normalization_mean']
    dataset_std = metadata['normalization_std']

    shard_metadatas = []
    for i in range(num_shards):
        with open(os.path.join(sisa_data_dir, f"shards/shard_{i+1}/metadata.json")) as f:
            shard_metadatas.append(json.load(f))

    validation_data = (
        np.load(os.path.join(sisa_data_dir, "validation_data/x_validation.npy")),
        np.load(os.path.join(sisa_data_dir, "validation_data/y_validation.npy")),
    )
    return sisa_data_dir, num_shards, num_slices, class_names, dataset_mean, dataset_std, shard_metadatas, validation_data


def _load_slice(sisa_data_dir: str, shard_idx: int, slice_idx: int):
    x_path = os.path.join(sisa_data_dir, f"shards/shard_{shard_idx+1}/slice_{slice_idx}_x.npy")
    if not os.path.exists(x_path):
        return None, None
    return np.load(x_path), np.load(x_path.replace('_x.npy', '_y.npy'))


def _cumulative_classes(sisa_data_dir: str, shard_idx: int, slice_idx: int, num_slices_seen: dict):
    # num_slices_seen caches each slice's own class set so repeated calls across
    # trials (same shard/slice data every time) don't re-load .npy files.
    key = (shard_idx, slice_idx)
    if key not in num_slices_seen:
        _, y = _load_slice(sisa_data_dir, shard_idx, slice_idx)
        num_slices_seen[key] = sorted(np.unique(y).tolist()) if y is not None else []
    known = set()
    for s in range(slice_idx + 1):
        known.update(num_slices_seen.get((shard_idx, s), []))
    return sorted(known)


def _shard_validation(shard_meta: dict, cumulative_classes, validation_data):
    x_val, y_val = validation_data
    valid_classes = set(shard_meta['class_indices_present']) & set(cumulative_classes)
    mask = np.isin(y_val, list(valid_classes))
    return x_val[mask], y_val[mask]


def _augmentation_for_shard(sisa_data_dir: str, shard_idx: int, num_slices: int):
    """Mirrors entry_training.py's check_class_balance_and_augmentation exactly
    (same thresholds) -- the real pipeline picks augmentation strength from
    each shard's class balance, and that choice measurably affects training."""
    all_labels = []
    for s in range(num_slices):
        _, y = _load_slice(sisa_data_dir, shard_idx, s)
        if y is not None:
            all_labels.extend(y.tolist())
    if not all_labels:
        return None

    counts = Counter(all_labels)
    total = len(all_labels)
    percentages = [c / total * 100 for c in counts.values()]
    min_pct, max_pct = min(percentages), max(percentages)
    balance_ratio = min_pct / max_pct if max_pct > 0 else 1.0
    std_dev = float(np.std(percentages))

    if balance_ratio >= 0.95 and std_dev <= 1.5:
        return None
    elif balance_ratio >= 0.85 and std_dev <= 2.5:
        return config.get_augmentation_config('minimal')
    elif balance_ratio >= 0.75 and std_dev <= 4.0:
        return config.get_augmentation_config('light')
    else:
        return config.get_augmentation_config('moderate')


def make_objective(project_name: str, epochs: int):
    (sisa_data_dir, num_shards, num_slices, class_names, dataset_mean, dataset_std,
     shard_metadatas, validation_data) = _load_project(project_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_transforms = T.Compose([T.Normalize(dataset_mean, dataset_std)])
    shard_class_indices = load_shard_class_indices(sisa_data_dir, num_shards)
    slice_classes_cache = {}
    augmentation_by_shard = [
        _augmentation_for_shard(sisa_data_dir, i, num_slices) for i in range(num_shards)
    ]
    x_val, y_val = validation_data

    def objective(trial: optuna.Trial) -> float:
        set_seed(config.SEED)  # keep every trial's own training deterministic

        lr = trial.suggest_float('LEARNING_RATE', 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float('WEIGHT_DECAY', 1e-5, 1e-2, log=True)
        label_smoothing = trial.suggest_float('LABEL_SMOOTHING', 0.0, 0.3)
        replay_ratio = trial.suggest_float('REPLAY_RATIO', 0.1, 0.5)
        batch_size = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128])
        fc_dropout = trial.suggest_float('FC_LAYER_DROPOUT', 0.2, 0.6)

        # train_model / create_sisa_model read several knobs directly off the
        # config module rather than accepting them as arguments -- set them
        # for the duration of this trial.
        original = {
            'WEIGHT_DECAY': config.WEIGHT_DECAY,
            'LABEL_SMOOTHING': config.LABEL_SMOOTHING,
            'FC_LAYER_DROPOUT': config.FC_LAYER_DROPOUT,
        }
        config.WEIGHT_DECAY = weight_decay
        config.LABEL_SMOOTHING = label_smoothing
        config.FC_LAYER_DROPOUT = fc_dropout

        trial_start = time.time()
        try:
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

                    current_model, _ = train_model(
                        x_slice, y_slice, model=current_model,
                        epochs=epochs, batch_size=batch_size, lr=lr,
                        validation_data=val_data, active_classes=known_classes,
                        replay_buffer=replay_buffer, replay_ratio=replay_ratio,
                        dataset_mean=dataset_mean, dataset_std=dataset_std,
                        training_type='incremental' if current_model is not None else 'fresh',
                        augmentation_config=augmentation_by_shard[i], device=device,
                    )
                    add_to_replay_buffer(replay_buffer, x_slice, y_slice, replay_rng,
                                          config.MAX_REPLAY_SAMPLES_PER_CLASS, replay_seen_counts)

                shard_models.append(current_model)

            # Full-system self-routing accuracy on VALIDATION data only (never test).
            for m in shard_models:
                if m is not None:
                    m.eval()
            all_preds = []
            with torch.no_grad():
                for k in range(0, len(x_val), config.BATCH_SIZE):
                    batch_x = torch.from_numpy(x_val[k:k + config.BATCH_SIZE]).float()
                    batch_x = eval_transforms(batch_x).to(device)
                    preds, _ = _run_sisa_batch(batch_x, shard_models, class_names, shard_class_indices)
                    all_preds.extend(preds.cpu().numpy())
            accuracy = float(np.mean(np.array(all_preds) == y_val))
        finally:
            for k, v in original.items():
                setattr(config, k, v)

        elapsed = time.time() - trial_start
        print(f"[Trial {trial.number}] val_accuracy={accuracy:.4f}  elapsed={elapsed:.1f}s")
        return accuracy

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="W13: one-time Optuna hyperparameter search over the FULL SISA pipeline (validation only)"
    )
    parser.add_argument('--project-name', type=str, required=True)
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--epochs-per-trial', type=int, default=config.MAX_EPOCHS,
                         help="Per-slice epoch budget (early stopping still applies). Defaults to the real "
                              "config.MAX_EPOCHS -- override to a small number only for a quick smoke test.")
    args = parser.parse_args()

    # Project-scoped, timestamped logging (W10) -- same mechanism entry_training.py,
    # entry_data_processing.py, and sisa_unlearning.py already use. Writing here
    # instead of relying on an external shell redirect keeps the log inside the
    # project's own logs/ directory, where it isn't at risk of being swept up by
    # an unrelated cleanup of loose repo-root files.
    log_dir = os.path.join(config.PROJECTS_DIR, args.project_name, "logs")
    _restore_logging, log_path = setup_run_logging(log_dir, "tuning")
    try:
        sampler = TPESampler(seed=config.SEED)  # seeded: the study itself is reproducible
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(make_objective(args.project_name, args.epochs_per_trial), n_trials=args.n_trials)

        print(f"\nBest validation accuracy: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        best_config = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': args.n_trials,
            'epochs_per_trial': args.epochs_per_trial,
            'project_name': args.project_name,
            'seed': config.SEED,
            'timestamp': datetime.now().isoformat(),
        }
        out_path = os.path.join(str(REPO_ROOT), "best_config.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nSaved to {out_path} -- freeze these values into config.py by hand, with a comment "
              f"citing this file, then this script's job is done (keep it here for reproducibility; "
              f"nothing at runtime imports it).")
        print(f"Log saved to: {log_path}")
    finally:
        _restore_logging()


if __name__ == "__main__":
    main()
