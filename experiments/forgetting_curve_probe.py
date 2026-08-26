"""Per-slice forgetting analysis (no retraining).

The training pipeline only writes a confusion matrix for each shard's *final* slice,
which cannot show whether catastrophic forgetting was actually fixed -- that question
is about how a class's accuracy evolves across the slices trained *after* it was
learned. Every intermediate checkpoint is on disk (`slice_{j}_model_*.pth`), so the
whole trajectory can be reconstructed after the fact.

For each shard and each slice checkpoint j, this evaluates that checkpoint on the
official test set restricted to the classes known up to slice j, with logits masked
to those classes. That isolates specialist quality from routing entirely -- no gate,
no cross-shard competition -- so any accuracy change between slice j and slice j+1 is
forgetting (or replay-driven recovery), not a routing artefact.

Reports, per shard:
  * the per-class recall matrix across slices (the forgetting curve),
  * retention deltas from each class's first appearance to the final slice,
  * the final-slice confusion matrix.

Usage:
    python experiments/forgetting_curve_probe.py
    python experiments/forgetting_curve_probe.py --project-name cifar10_sisa_pytorch
"""
import argparse
import json
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
from training.create_model import load_model_pytorch, DEVICE


def _evaluate(model, x, y, allowed_classes, normalize, head_classes=None, batch_size=512):
    """Predict over `x`, restricted to `allowed_classes` (global class ids).

    W33: the head may be GLOBAL width (one column per dataset class) or SHARD width
    (one column per owned class, ordered by `sorted(head_classes)`). The columns to
    select differ between the two, and getting it wrong fails silently rather than
    raising -- a shard-width head indexed with global ids returns whichever classes
    happen to sit at those local positions, producing plausible but wrong numbers.
    """
    allowed_sorted = sorted(allowed_classes)
    allowed = torch.tensor(allowed_sorted, device=DEVICE, dtype=torch.long)

    preds = []
    for start in range(0, len(x), batch_size):
        chunk = torch.from_numpy(x[start:start + batch_size].astype(np.float32))
        chunk = normalize(chunk).to(DEVICE)
        with torch.no_grad():
            logits = model(chunk)

        if head_classes is not None and logits.shape[1] == len(head_classes):
            # Shard-width head: translate global ids to this head's local column order.
            head_sorted = sorted(head_classes)
            local = torch.tensor([head_sorted.index(c) for c in allowed_sorted],
                                 device=DEVICE, dtype=torch.long)
            logits = logits[:, local]
        elif logits.shape[1] != len(allowed):
            # Global-width head: the global ids ARE the column indices.
            logits = logits[:, allowed]

        preds.append(allowed[logits.argmax(dim=1)].cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(description="Per-slice catastrophic forgetting analysis")
    parser.add_argument('--project-name', type=str, default=config.PROJECT_NAME)
    args = parser.parse_args()

    set_seed(config.SEED)

    base_dir = os.path.join(config.PROJECTS_DIR, args.project_name)
    sisa_dir = os.path.join(base_dir, "sisa_data")
    models_dir = os.path.join(base_dir, "models")

    with open(os.path.join(sisa_dir, "metadata.json")) as f:
        meta = json.load(f)
    num_shards, num_slices = meta['num_shards'], meta['num_slices']
    class_names = meta['class_names']
    normalize = T.Normalize(meta['normalization_mean'], meta['normalization_std'])

    x_test = np.load(os.path.join(sisa_dir, "test_data", "x_test.npy"))
    y_test = np.load(os.path.join(sisa_dir, "test_data", "y_test.npy"))

    print("=" * 84)
    print("PER-SLICE FORGETTING ANALYSIS  (specialists evaluated in isolation, no routing)")
    print("=" * 84)

    for shard_idx in range(num_shards):
        shard_dir = os.path.join(models_dir, f"shard_{shard_idx + 1}")

        # Which classes each slice introduces, in training order.
        slice_classes, cumulative = [], []
        seen = []
        for sl in range(num_slices):
            yp = os.path.join(sisa_dir, "shards", f"shard_{shard_idx+1}", f"slice_{sl}_y.npy")
            if not os.path.exists(yp):
                slice_classes.append([])
                cumulative.append(list(seen))
                continue
            ys = np.load(yp)
            present = sorted(int(c) for c in np.unique(ys))
            slice_classes.append(present)
            for c in present:
                if c not in seen:
                    seen.append(c)
            cumulative.append(list(seen))

        ordered = list(seen)  # classes in the order they were first trained
        print(f"\n{'='*84}\nSHARD {shard_idx+1} -- {len(ordered)} classes, trained in this order:")
        print("   " + " -> ".join(class_names[c] for c in ordered))

        # recall[class][slice] once that class has been introduced
        recall = {c: {} for c in ordered}
        first_slice = {}
        final_preds = None

        for sl in range(num_slices):
            ckpt = os.path.join(shard_dir, f"slice_{sl}_model_{config.MODEL_TYPE}.pth")
            if not os.path.exists(ckpt) or not cumulative[sl]:
                continue
            model, _ = load_model_pytorch(ckpt)
            model.eval()

            known = cumulative[sl]
            mask = np.isin(y_test, known)
            # W33: `ordered` is this shard's full owned set, i.e. the head's column order
            # when the head is shard-width. Harmless for a global-width head.
            preds = _evaluate(model, x_test[mask], y_test[mask], known, normalize,
                              head_classes=ordered)
            truth = y_test[mask]

            for c in known:
                sel = truth == c
                if sel.any():
                    recall[c][sl] = float((preds[sel] == c).mean())
                    first_slice.setdefault(c, sl)

            if sl == num_slices - 1:
                final_preds, final_truth = preds, truth

        # --- forgetting curve ---
        header = "class".ljust(13) + "".join(f"slice{s+1}".rjust(9) for s in range(num_slices))
        print("\n  Per-class recall after each slice (blank = not yet introduced):")
        print("  " + header + "     first->last")
        print("  " + "-" * (len(header) + 16))
        deltas = []
        for c in ordered:
            row = f"{class_names[c]:<13}"
            for s in range(num_slices):
                row += (f"{recall[c][s]*100:8.1f}%" if s in recall[c] else " " * 9)
            fs = first_slice.get(c)
            ls = max(recall[c]) if recall[c] else None
            if fs is not None and ls is not None and fs != ls:
                d = (recall[c][ls] - recall[c][fs]) * 100
                deltas.append(d)
                row += f"    {d:+7.1f} pts"
            elif fs is not None:
                row += "      (last slice)"
            print("  " + row)

        if deltas:
            print(f"\n  Retention from first appearance to final slice:")
            print(f"    mean {np.mean(deltas):+.1f} pts | worst {min(deltas):+.1f} pts | "
                  f"best {max(deltas):+.1f} pts | classes losing >10 pts: {sum(1 for d in deltas if d < -10)}/{len(deltas)}")

        # --- final-slice confusion matrix ---
        if final_preds is not None:
            print(f"\n  Final-slice confusion matrix (rows = true, cols = predicted):")
            hdr = "true\\pred".ljust(13) + "".join(class_names[c][:8].rjust(9) for c in ordered)
            print("  " + hdr)
            for c in ordered:
                sel = final_truth == c
                row = f"{class_names[c]:<13}"
                for p in ordered:
                    row += f"{int((final_preds[sel] == p).sum()):9d}"
                row += f"   recall {float((final_preds[sel]==c).mean())*100:5.1f}%"
                print("  " + row)
            acc = float((final_preds == final_truth).mean())
            print(f"\n  Shard {shard_idx+1} isolated accuracy (perfect routing): {acc*100:.2f}%")


if __name__ == "__main__":
    main()
