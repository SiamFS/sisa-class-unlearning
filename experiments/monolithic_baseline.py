"""Monolithic (non-SISA) baseline -- the denominator for every claim in the paper.

Trains ONE model on all classes at once, i.i.d., with no sharding, no slicing and no
replay, using the same backbone, hyperparameters and augmentation as the SISA
specialists. Two numbers come out of it, and both are needed:

  * ACCURACY -- what the framework costs. "SISA reaches X%" means nothing without
    "a single model on the same data and architecture reaches Y%".
  * FULL-RETRAIN TIME -- the denominator of the unlearning speedup. Deleting a class
    from a monolithic model requires retraining it from scratch on the remaining data;
    that time is what SISA's partial-shard retrain is compared against. Quoting a
    speedup without measuring it is an assumption, not a result.

Training data is reconstructed from the sharded slices rather than re-split from the
raw dataset, so the baseline sees EXACTLY the samples the SISA system saw -- otherwise
the comparison silently includes a data difference.

Usage:
    python experiments/monolithic_baseline.py
    python experiments/monolithic_baseline.py --project-name cifar10_sisa_pytorch
    python experiments/monolithic_baseline.py --exclude-class cat   # the retrain-from-
                                                                    # scratch reference
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import classification_report

import config
from utils.seeding import set_seed
from utils.run_logging import setup_run_logging
from utils.data_io import load_images
from training.train_model import train_model
from training.create_model import save_model_pytorch, DEVICE


def load_all_training_data(sisa_data_dir: str, metadata: dict):
    """Reassemble the full training set from the sharded slices.

    Using the slices (rather than re-splitting the raw dataset) guarantees the baseline
    trains on exactly the samples the SISA system trained on.
    """
    num_shards = metadata['num_shards']
    slices_per_shard = metadata.get('slices_per_shard') or [metadata['num_slices']] * num_shards

    xs, ys = [], []
    for shard_idx in range(num_shards):
        for slice_idx in range(slices_per_shard[shard_idx]):
            x_path = os.path.join(sisa_data_dir, f"shards/shard_{shard_idx+1}/slice_{slice_idx}_x.npy")
            if not os.path.exists(x_path):
                continue
            xs.append(load_images(x_path))
            ys.append(np.load(x_path.replace('_x.npy', '_y.npy')))
    return np.concatenate(xs), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser(description="Monolithic non-SISA baseline")
    parser.add_argument('--project-name', type=str, default=config.PROJECT_NAME)
    parser.add_argument('--exclude-class', type=str, default=None,
                        help="Class NAME to drop before training -- gives the "
                             "retrain-from-scratch reference a naive system would need "
                             "in order to unlearn that class.")
    parser.add_argument('--epochs', type=int, default=config.MAX_EPOCHS)
    args = parser.parse_args()

    set_seed(config.SEED)

    base_dir = os.path.join(config.PROJECTS_DIR, args.project_name)
    sisa_data_dir = os.path.join(base_dir, "sisa_data")
    with open(os.path.join(sisa_data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    class_names = metadata['class_names']
    dataset_mean, dataset_std = metadata['normalization_mean'], metadata['normalization_std']

    _restore_logging, log_path = setup_run_logging(os.path.join(base_dir, "logs"), "monolithic")
    try:
        print("=" * 70)
        print("MONOLITHIC BASELINE (no sharding, no slicing, no replay)")
        print("=" * 70)
        print(f"   - Backbone: {config.MODEL_ARCH}   optimizer: {config.OPTIMIZER}   "
              f"lr: {config.LEARNING_RATE}   label smoothing: {config.LABEL_SMOOTHING}")

        x_train, y_train = load_all_training_data(sisa_data_dir, metadata)
        x_val = load_images(os.path.join(sisa_data_dir, "validation_data/x_validation.npy"))
        y_val = np.load(os.path.join(sisa_data_dir, "validation_data/y_validation.npy"))
        x_test = load_images(os.path.join(sisa_data_dir, "test_data/x_test.npy"))
        y_test = np.load(os.path.join(sisa_data_dir, "test_data/y_test.npy"))

        excluded_idx = None
        if args.exclude_class:
            if args.exclude_class not in class_names:
                raise ValueError(f"Unknown class {args.exclude_class!r}; have {class_names}")
            excluded_idx = class_names.index(args.exclude_class)
            keep = y_train != excluded_idx
            x_train, y_train = x_train[keep], y_train[keep]
            vkeep = y_val != excluded_idx
            x_val, y_val = x_val[vkeep], y_val[vkeep]
            print(f"   - Excluding '{args.exclude_class}' (class {excluded_idx}): "
                  f"retrain-from-scratch reference for a naive system")

        active_classes = sorted(np.unique(y_train).tolist())
        print(f"   - Train {len(x_train):,} samples over {len(active_classes)} classes "
              f"(i.i.d., shuffled -- no slice ordering)")

        # The head keeps GLOBAL width so predictions stay in global class space and are
        # directly comparable with the SISA system's outputs.
        train_start = time.time()
        model, history = train_model(
            x_train, y_train, model=None,
            epochs=args.epochs, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
            validation_data=(x_val, y_val), active_classes=active_classes,
            replay_buffer=None, replay_ratio=0.0,   # nothing to replay: one joint pass
            dataset_mean=dataset_mean, dataset_std=dataset_std,
            training_type='fresh',
            augmentation_config=config.get_augmentation_config('baseline'),
        )
        pure_train_time = time.time() - train_start

        # --- test-set evaluation, restricted to the classes actually trained on ---
        model.eval()
        normalize = T.Compose([T.Normalize(dataset_mean, dataset_std)])
        allowed = torch.tensor(active_classes, device=DEVICE, dtype=torch.long)
        test_mask = np.isin(y_test, active_classes)
        x_eval, y_eval = x_test[test_mask], y_test[test_mask]

        preds = []
        with torch.no_grad():
            for i in range(0, len(x_eval), config.BATCH_SIZE):
                batch = torch.from_numpy(x_eval[i:i + config.BATCH_SIZE].astype(np.float32))
                logits = model(normalize(batch).to(DEVICE))
                if logits.shape[1] != len(allowed):
                    logits = logits[:, allowed]
                preds.append(allowed[logits.argmax(dim=1)].cpu().numpy())
        preds = np.concatenate(preds)
        accuracy = float((preds == y_eval).mean())

        print("\n" + "-" * 60)
        print(classification_report(y_eval, preds,
                                    labels=active_classes,
                                    target_names=[class_names[c] for c in active_classes],
                                    zero_division=0))
        print("-" * 60)
        print(f"Monolithic baseline accuracy : {accuracy:.4f}")
        print(f"Pure training time           : {pure_train_time:.2f} seconds")
        print(f"Epochs run                   : {len(history['loss'])}")

        suffix = f"_without_{args.exclude_class}" if args.exclude_class else ""
        save_model_pytorch(model, os.path.join(base_dir, "models",
                                                f"monolithic_baseline{suffix}.pth"))
        out = {
            'accuracy': accuracy,
            'pure_training_time': pure_train_time,
            'epochs': len(history['loss']),
            'excluded_class': args.exclude_class,
            'train_samples': int(len(x_train)),
            'num_classes': len(active_classes),
            'model_arch': config.MODEL_ARCH,
            'optimizer': config.OPTIMIZER,
            'learning_rate': config.LEARNING_RATE,
            'label_smoothing': config.LABEL_SMOOTHING,
        }
        out_path = os.path.join(sisa_data_dir, f"monolithic_baseline{suffix}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics to {out_path}")
        print(f"Log saved to: {log_path}")
    finally:
        _restore_logging()


if __name__ == "__main__":
    main()
