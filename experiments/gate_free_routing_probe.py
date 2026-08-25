"""Gate-free routing comparison (no retraining).

Motivation: the learned gating network (W16) buys ~12 points of combined
accuracy over the parameter-free confidence routing it replaced, but it is a
~544k-parameter model that must itself be retrained on every unlearning
request to keep the exactness claim (see IMPLEMENTATION_PLAN.md W16). This
probe asks whether any *post-hoc* routing score -- computed from the already
trained specialists, with no learned router anywhere -- can close that gap.

Everything here is evaluated on the SAME already-trained shard checkpoints,
so no training happens and the comparison is apples-to-apples by construction.
The gating network is included only as a reference upper line, and an oracle
router (route by the true class's owning shard) gives the ceiling that any
routing strategy could reach with these specialists.

Scores tested (all parameter-free unless noted):
  max_softmax        current gate-free fallback in plots._run_sisa_batch
  max_softmax_temp   ditto, with config.PRIMARY_SPECIALIST_TEMPERATURE
  max_logit          raw max owned logit
  energy             logsumexp over owned logits (Liu et al., NeurIPS 2020)
  energy_debiased    T*(logsumexp(owned/T) - log k), de-biased for owned-class count
  margin             top1 - top2 within owned classes
  neg_entropy        negative entropy of the owned-class softmax
  ncm_euclidean      ARCANE-flavoured: -min distance to a training class mean
  ncm_cosine         max cosine similarity to a training class mean
  mahalanobis        class-conditional Gaussian, tied covariance (Lee et al. 2018)
  zscore_softmax     max_softmax standardised per shard by validation ID stats
  platt              2-parameter logistic per shard, fit on validation (NOT
                     parameter-free -- reported for reference only)

Usage:
    python experiments/gate_free_routing_probe.py
    python experiments/gate_free_routing_probe.py --project-name cifar10_sisa_pytorch
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
from plots import load_shard_class_indices


def _extract(model, x_batch):
    """Return (penultimate features, logits) for one normalized batch.

    SISAConvNet.fc_layer is [Dropout, Linear, BatchNorm1d, ReLU, Dropout, Linear],
    so index 5 is the classifier and everything before it is the feature stack.
    """
    with torch.no_grad():
        feats = model.conv_layer(x_batch)
        feats = feats.reshape(feats.size(0), -1)
        penult = model.fc_layer[:5](feats)
        logits = model.fc_layer[5](penult)
    return penult, logits


def _forward_all(models, x, normalize, batch_size=256):
    """Run every shard model over x, returning per-shard (features, logits) arrays."""
    n = len(x)
    feats_per_shard = [[] for _ in models]
    logits_per_shard = [[] for _ in models]

    for start in range(0, n, batch_size):
        chunk = torch.from_numpy(x[start:start + batch_size].astype(np.float32))
        chunk = normalize(chunk).to(DEVICE)
        for si, model in enumerate(models):
            penult, logits = _extract(model, chunk)
            feats_per_shard[si].append(penult.cpu().numpy())
            logits_per_shard[si].append(logits.cpu().numpy())

    return (
        [np.concatenate(f) for f in feats_per_shard],
        [np.concatenate(l) for l in logits_per_shard],
    )


def _gating_route(gating_model, x, normalize, batch_size=256):
    """Reference only: the learned gate's shard choice for every sample."""
    preds = []
    for start in range(0, len(x), batch_size):
        chunk = torch.from_numpy(x[start:start + batch_size].astype(np.float32))
        chunk = normalize(chunk).to(DEVICE)
        with torch.no_grad():
            preds.append(gating_model(chunk).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _owned(logits, owned_sorted):
    return logits[:, owned_sorted]


# --- routing scores: each returns (n_samples,) "how much this shard claims it" ---

def score_max_softmax(owned_logits, **_):
    return _softmax(owned_logits).max(axis=1)


def score_max_softmax_temp(owned_logits, **_):
    return _softmax(owned_logits / config.PRIMARY_SPECIALIST_TEMPERATURE).max(axis=1)


def score_max_logit(owned_logits, **_):
    return owned_logits.max(axis=1)


def score_energy(owned_logits, **_):
    from scipy.special import logsumexp
    return logsumexp(owned_logits, axis=1)


def score_energy_debiased(owned_logits, temperature=1.0, **_):
    from scipy.special import logsumexp
    k = owned_logits.shape[1]
    return temperature * (logsumexp(owned_logits / temperature, axis=1) - np.log(k))


def score_margin(owned_logits, **_):
    p = np.sort(_softmax(owned_logits), axis=1)
    if p.shape[1] < 2:
        return p[:, -1]
    return p[:, -1] - p[:, -2]


def score_neg_entropy(owned_logits, **_):
    p = _softmax(owned_logits)
    return (p * np.log(p + config.MIN_PROB_EPSILON)).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description="Compare gate-free SISA routing strategies")
    parser.add_argument('--project-name', type=str, default=config.PROJECT_NAME)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    set_seed(config.SEED)

    base_dir = os.path.join(config.PROJECTS_DIR, args.project_name)
    sisa_dir = os.path.join(base_dir, "sisa_data")
    models_dir = os.path.join(base_dir, "models")

    with open(os.path.join(sisa_dir, "metadata.json")) as f:
        meta = json.load(f)
    num_shards = meta['num_shards']
    num_slices = meta['num_slices']
    class_names = meta['class_names']
    normalize = T.Normalize(meta['normalization_mean'], meta['normalization_std'])

    shard_class_indices = load_shard_class_indices(sisa_dir, num_shards)
    owned_sorted = [sorted(c) for c in shard_class_indices]

    print("=" * 78)
    print("GATE-FREE ROUTING PROBE  (no retraining -- existing checkpoints only)")
    print("=" * 78)
    for si, owned in enumerate(owned_sorted):
        print(f"  Shard {si+1} owns {len(owned)} classes: {[class_names[c] for c in owned]}")

    # --- load trained specialists -------------------------------------------------
    models = []
    for si in range(num_shards):
        path = os.path.join(models_dir, f"shard_{si+1}", f"final_model_shard{si+1}_{config.MODEL_TYPE}.pth")
        model, _ = load_model_pytorch(path)
        model.eval()
        models.append(model)

    gating_model = None
    gating_path = os.path.join(models_dir, "gating_model.pth")
    if os.path.exists(gating_path):
        gating_model, _ = load_model_pytorch(gating_path, num_shards=num_shards)
        gating_model.eval()

    # --- data ---------------------------------------------------------------------
    x_test = np.load(os.path.join(sisa_dir, "test_data", "x_test.npy"))
    y_test = np.load(os.path.join(sisa_dir, "test_data", "y_test.npy"))
    x_val = np.load(os.path.join(sisa_dir, "validation_data", "x_validation.npy"))
    y_val = np.load(os.path.join(sisa_dir, "validation_data", "y_validation.npy"))

    # Training data per shard, for class prototypes / covariance (ARCANE-style r_i).
    train_x_per_shard, train_y_per_shard = [], []
    for si in range(num_shards):
        xs, ys = [], []
        for sl in range(num_slices):
            xp = os.path.join(sisa_dir, "shards", f"shard_{si+1}", f"slice_{sl}_x.npy")
            if os.path.exists(xp):
                xs.append(np.load(xp))
                ys.append(np.load(xp.replace('_x.npy', '_y.npy')))
        train_x_per_shard.append(np.concatenate(xs))
        train_y_per_shard.append(np.concatenate(ys))

    class_to_shard = {c: si for si, owned in enumerate(owned_sorted) for c in owned}
    true_shard_test = np.array([class_to_shard[int(c)] for c in y_test])
    true_shard_val = np.array([class_to_shard[int(c)] for c in y_val])

    print(f"\n  Test samples: {len(y_test):,}   Validation samples: {len(y_val):,}")
    print("  Extracting features/logits from trained specialists...")

    test_feats, test_logits = _forward_all(models, x_test, normalize, args.batch_size)
    val_feats, val_logits = _forward_all(models, x_val, normalize, args.batch_size)

    # Each shard's own training data, seen through its own network, for prototypes.
    proto_mu, proto_cov_inv = [], []
    for si in range(num_shards):
        f, _ = _forward_all([models[si]], train_x_per_shard[si], normalize, args.batch_size)
        f = f[0]
        ys = train_y_per_shard[si]
        mus, centered = {}, []
        for c in owned_sorted[si]:
            m = f[ys == c]
            mus[c] = m.mean(axis=0)
            centered.append(m - mus[c])
        proto_mu.append(mus)
        cov = np.cov(np.concatenate(centered).T) + 1e-6 * np.eye(f.shape[1])
        proto_cov_inv.append(np.linalg.pinv(cov))

    # --- evaluation helpers -------------------------------------------------------
    def evaluate(shard_choice):
        """Given a routed shard per sample, compute routing + combined accuracy."""
        routing_acc = float((shard_choice == true_shard_test).mean())
        preds = np.empty(len(y_test), dtype=np.int64)
        for si in range(num_shards):
            sel = shard_choice == si
            if not sel.any():
                continue
            ow = np.array(owned_sorted[si])
            preds[sel] = ow[_owned(test_logits[si], owned_sorted[si])[sel].argmax(axis=1)]
        return routing_acc, float((preds == y_test).mean())

    def route_by_score(fn, **kw):
        s = np.stack([fn(_owned(test_logits[si], owned_sorted[si]), **kw) for si in range(num_shards)])
        return s.argmax(axis=0)

    results = []

    # Ceiling and reference lines.
    results.append(("oracle (perfect routing)", *evaluate(true_shard_test), "ceiling"))
    if gating_model is not None:
        results.append(("gating network (W16)", *evaluate(_gating_route(gating_model, x_test, normalize, args.batch_size)), "learned"))

    for name, fn in [
        ("max_softmax", score_max_softmax),
        ("max_softmax_temp", score_max_softmax_temp),
        ("max_logit", score_max_logit),
        ("energy", score_energy),
        ("energy_debiased", score_energy_debiased),
        ("margin", score_margin),
        ("neg_entropy", score_neg_entropy),
    ]:
        results.append((name, *evaluate(route_by_score(fn)), "free"))

    # Feature-space prototype routing (ARCANE-flavoured, still parameter-free).
    def ncm_scores(feats_per_shard, metric):
        out = []
        for si in range(num_shards):
            f = feats_per_shard[si]
            mus = np.stack([proto_mu[si][c] for c in owned_sorted[si]])
            if metric == 'euclidean':
                d = ((f[:, None, :] - mus[None, :, :]) ** 2).sum(axis=2)
                out.append(-d.min(axis=1))
            elif metric == 'cosine':
                fn_ = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
                mn = mus / (np.linalg.norm(mus, axis=1, keepdims=True) + 1e-12)
                out.append((fn_ @ mn.T).max(axis=1))
            else:  # mahalanobis
                best = None
                for k, c in enumerate(owned_sorted[si]):
                    diff = f - mus[k]
                    m = np.einsum('ij,jk,ik->i', diff, proto_cov_inv[si], diff)
                    best = m if best is None else np.minimum(best, m)
                out.append(-best)
        return np.stack(out)

    for name, metric in [("ncm_euclidean", "euclidean"), ("ncm_cosine", "cosine"), ("mahalanobis", "mahalanobis")]:
        results.append((name, *evaluate(ncm_scores(test_feats, metric).argmax(axis=0)), "free"))

    # W21: the head's own class vectors as prototypes (learned, not post-hoc feature
    # means). With a cosine head these are what the classifier actually trains on; with
    # a linear head this is the same geometry against unnormalized weight rows.
    head_cos = []
    for si in range(num_shards):
        w = models[si].fc_layer[5].weight.detach().cpu().numpy()
        w = w[owned_sorted[si]] if w.shape[0] == len(class_names) else w
        f = test_feats[si]
        fn_ = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
        wn = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)
        head_cos.append((fn_ @ wn.T).max(axis=1))
    results.append(("head_cosine (W21)", *evaluate(np.stack(head_cos).argmax(axis=0)), "free"))

    # Per-shard standardisation of max_softmax using validation in-distribution stats.
    z_stats = []
    for si in range(num_shards):
        s_val = score_max_softmax(_owned(val_logits[si], owned_sorted[si]))
        id_mask = true_shard_val == si
        z_stats.append((s_val[id_mask].mean(), s_val[id_mask].std() + 1e-12))
    z = np.stack([
        (score_max_softmax(_owned(test_logits[si], owned_sorted[si])) - z_stats[si][0]) / z_stats[si][1]
        for si in range(num_shards)
    ])
    results.append(("zscore_softmax", *evaluate(z.argmax(axis=0)), "free"))

    # 2-parameter logistic per shard, fit on validation (has parameters -> reference only).
    platt = []
    for si in range(num_shards):
        s = score_max_softmax(_owned(val_logits[si], owned_sorted[si])).reshape(-1, 1)
        t = (true_shard_val == si).astype(int)
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000).fit(s, t)
            platt.append(lr)
        except Exception:
            platt.append(None)
    if all(p is not None for p in platt):
        p = np.stack([
            platt[si].predict_proba(
                score_max_softmax(_owned(test_logits[si], owned_sorted[si])).reshape(-1, 1)
            )[:, 1]
            for si in range(num_shards)
        ])
        results.append(("platt (2 params/shard)", *evaluate(p.argmax(axis=0)), "fitted"))

    # --- report -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'strategy':<26}{'kind':<10}{'routing acc':>14}{'combined acc':>16}")
    print("-" * 78)
    for name, racc, cacc, kind in results:
        print(f"{name:<26}{kind:<10}{racc*100:>13.2f}%{cacc*100:>15.2f}%")
    print("=" * 78)

    free = [r for r in results if r[3] == 'free']
    if free:
        best = max(free, key=lambda r: r[2])
        gate = next((r for r in results if r[3] == 'learned'), None)
        print(f"\nBest gate-free strategy: {best[0]}  ({best[2]*100:.2f}% combined)")
        if gate:
            print(f"Learned gate:            {gate[0]}  ({gate[2]*100:.2f}% combined)")
            print(f"Gap to close:            {(gate[2]-best[2])*100:+.2f} points")


if __name__ == "__main__":
    main()
