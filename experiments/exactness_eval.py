"""W8: the exactness proof harness's main entry point.

Orchestrates: real unlearning (on its own copy) + a from-scratch reference
(via scratch_reference.py, on its own copy) for the SAME class, then compares
them with the five metrics from IMPLEMENTATION_PLAN.md §4.4 plus an MIA, and
produces the four plots from §4.5. The source project is never modified.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, roc_curve

import config
from training.create_model import load_model_pytorch, DEVICE
from unlearning.sisa_unlearning import SISAUnlearning
from plots import (
    _run_sisa_batch,
    load_shard_class_indices,
    create_exactness_comparison_chart,
    create_mia_roc_chart,
    create_exactness_confusion_matrices,
    create_efficiency_comparison_chart,
)
from experiments.scratch_reference import build_scratch_reference, _copy_project


def _load_system_models(project_name: str, model_name: str, num_shards: int):
    models = [None] * num_shards
    for i in range(num_shards):
        path = os.path.join(config.PROJECTS_DIR, project_name, "models", f"shard_{i+1}",
                             f"final_model_shard{i+1}_{model_name}.pth")
        if os.path.exists(path):
            model, _ = load_model_pytorch(path)
            models[i] = model.eval()
    return models


def _predictions_and_probs(shard_models, shard_class_indices, class_names, X, eval_transforms,
                            batch_size, gating_model=None):
    """W31: evaluate through the router the system actually uses.

    This previously called _run_sisa_batch with no gate, i.e. parameter-free confidence
    routing -- but since W16 the real system routes through the learned gate. Comparing
    an unlearned system and a scratch system under a routing mechanism neither of them
    deploys makes every downstream metric (prediction agreement, output-distribution
    distance, MIA) measure the wrong system.

    The SAME gate is used for both sides. That is deliberate: the gate is retrained
    excluding the deleted class either way, so holding it fixed isolates the specialist
    difference, which is exactly what the exactness metrics are about.
    """
    preds, probs_all = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = torch.from_numpy(X[i:i + batch_size].astype(np.float32))
            batch_x = eval_transforms(batch_x).to(DEVICE)
            p, pr = _run_sisa_batch(batch_x, shard_models, class_names, shard_class_indices,
                                     threshold=None, gating_model=gating_model)
            preds.extend(p.cpu().numpy())
            probs_all.append(pr.cpu().numpy())
    return np.array(preds), np.concatenate(probs_all, axis=0)


def _confidence_scores(shard_models, shard_class_indices, class_names, X, eval_transforms,
                        batch_size, gating_model=None):
    """Max combined-probability on whatever the system predicts -- the MIA signal.
    Deliberately not loss-against-true-label: the affected shard's dynamic head
    (W6) has no output column for the deleted class at all, so that loss would be
    degenerate (-log(0)) for every sample, member or not."""
    if len(X) == 0:
        return np.array([])
    _, probs = _predictions_and_probs(shard_models, shard_class_indices, class_names, X,
                                       eval_transforms, batch_size, gating_model)
    return probs.max(axis=1)


def _compute_mia(shard_models, shard_class_indices, class_names, member_x, nonmember_x,
                  eval_transforms, batch_size, gating_model=None):
    member_conf = _confidence_scores(shard_models, shard_class_indices, class_names, member_x,
                                      eval_transforms, batch_size, gating_model)
    nonmember_conf = _confidence_scores(shard_models, shard_class_indices, class_names, nonmember_x,
                                         eval_transforms, batch_size, gating_model)
    scores = np.concatenate([member_conf, nonmember_conf])
    labels = np.concatenate([np.ones(len(member_conf)), np.zeros(len(nonmember_conf))])
    auc_val = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, auc_val


def _parameter_distance(model_a, model_b):
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    flat_a, flat_b = [], []
    for k in sd_a:
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            flat_a.append(sd_a[k].flatten().float().cpu())
            flat_b.append(sd_b[k].flatten().float().cpu())
    a_cat, b_cat = torch.cat(flat_a), torch.cat(flat_b)
    l2 = float(torch.norm(a_cat - b_cat).item())
    cos_sim = float(torch.nn.functional.cosine_similarity(a_cat.unsqueeze(0), b_cat.unsqueeze(0)).item())
    return {'l2': l2, 'cosine': 1.0 - cos_sim}


def _mean_kl(probs_p: np.ndarray, probs_q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(probs_p, eps, 1.0)
    q = np.clip(probs_q, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def run_exactness_eval(source_project: str, class_name: str, model_name: str = None,
                        unlearned_project: str = None):
    model_name = model_name or config.MODEL_TYPE
    unlearned_project = unlearned_project or f"{source_project}_unlearned_{class_name}"

    print(f"\n{'='*70}\nW8 EXACTNESS EVAL for class '{class_name}' (source: {source_project})\n{'='*70}")

    with open(os.path.join(config.PROJECTS_DIR, source_project, "sisa_data", "metadata.json")) as f:
        src_metadata = json.load(f)
    class_names = src_metadata['class_names']
    num_shards = src_metadata['num_shards']
    eval_transforms = T.Compose([T.Normalize(src_metadata['normalization_mean'], src_metadata['normalization_std'])])
    batch_size = config.BATCH_SIZE
    class_idx = class_names.index(class_name)

    # Original (pre-unlearning) system -- straight from the pristine, untouched source.
    original_models = _load_system_models(source_project, model_name, num_shards)
    original_shard_indices = load_shard_class_indices(os.path.join(config.PROJECTS_DIR, source_project, "sisa_data"), num_shards)

    # Scratch reference FIRST -- this is what snapshots the MIA "member" samples
    # from the pristine source, before anything gets copied or modified.
    scratch_result = build_scratch_reference(source_project, class_name, model_name)

    # Real unlearning, on its own separate copy.
    print(f"\n{'='*70}\nRunning real unlearning on a copy ('{unlearned_project}')\n{'='*70}")
    _copy_project(source_project, unlearned_project)
    unlearner = SISAUnlearning(unlearned_project, model_name)
    unlearn_start = time.time()
    unlearner.unlearn_by_class(class_name)
    real_retrain_time = unlearner.pure_retraining_time

    unlearned_models = _load_system_models(unlearned_project, model_name, num_shards)
    unlearned_shard_indices = load_shard_class_indices(os.path.join(config.PROJECTS_DIR, unlearned_project, "sisa_data"), num_shards)

    # Scratch system = unlearned system's OTHER shards (unaffected, identical by
    # construction -- W6) + the scratch model swapped in for the affected shard.
    shard_idx = scratch_result['shard_idx']
    scratch_models = list(unlearned_models)
    scratch_models[shard_idx] = scratch_result['scratch_model'].eval()
    # Same class set on the affected shard either way (both stripped the same
    # class from the same shard), so the unlearned copy's shard_class_indices
    # apply to the scratch system too.
    scratch_shard_indices = list(unlearned_shard_indices)

    x_test = np.load(os.path.join(config.PROJECTS_DIR, source_project, "sisa_data", "test_data", "x_test.npy"))
    y_test = np.load(os.path.join(config.PROJECTS_DIR, source_project, "sisa_data", "test_data", "y_test.npy"))

    print(f"\n{'='*70}\nComputing exactness metrics\n{'='*70}")

    # Metric 1: parameter distance (affected shard only).
    param_distance = _parameter_distance(unlearned_models[shard_idx], scratch_models[shard_idx])
    print(f"   - Parameter distance: L2={param_distance['l2']:.4f}, cosine dist={param_distance['cosine']:.6f} "
          f"(not expected to be ~0 -- see report notes)")

    preds_original, probs_original = _predictions_and_probs(original_models, original_shard_indices, class_names, x_test, eval_transforms, batch_size, original_gate)
    preds_unlearned, probs_unlearned = _predictions_and_probs(unlearned_models, unlearned_shard_indices, class_names, x_test, eval_transforms, batch_size, unlearned_gate)
    preds_scratch, probs_scratch = _predictions_and_probs(scratch_models, scratch_shard_indices, class_names, x_test, eval_transforms, batch_size, unlearned_gate)

    # Metric 2: prediction agreement.
    pred_agreement = float(np.mean(preds_unlearned == preds_scratch))
    print(f"   - Prediction agreement (unlearned vs scratch): {pred_agreement*100:.2f}%")

    # Metric 3: output-distribution distance.
    output_kl = _mean_kl(probs_unlearned, probs_scratch)
    print(f"   - Mean output KL divergence (unlearned vs scratch): {output_kl:.6f}")

    # Metric 4: MIA. Members = the class's real training samples (snapshotted
    # before deletion); non-members = the class's held-out test samples.
    nonmember_mask = (y_test == class_idx)
    nonmember_x = x_test[nonmember_mask]
    mia_results = {}
    for label, models, shard_idxs, gate in [
        ('original', original_models, original_shard_indices, original_gate),
        ('unlearned', unlearned_models, unlearned_shard_indices, unlearned_gate),
        ('scratch', scratch_models, scratch_shard_indices, unlearned_gate),
    ]:
        fpr, tpr, auc_val = _compute_mia(models, shard_idxs, class_names, scratch_result['member_x'],
                                          nonmember_x, eval_transforms, batch_size, gate)
        mia_results[label] = (fpr, tpr, auc_val)
        print(f"   - MIA AUC ({label}): {auc_val:.4f}")

    # Metric 5: deleted-class behavior -- agreement on what the two systems
    # predict for that class's images (neither can predict the class itself).
    deleted_mask = (y_test == class_idx)
    deleted_agreement = float(np.mean(preds_unlearned[deleted_mask] == preds_scratch[deleted_mask]))
    print(f"   - Deleted-class prediction agreement (unlearned vs scratch): {deleted_agreement*100:.2f}%")

    scratch_time = scratch_result['pure_train_time']
    print(f"\n   - Efficiency: scratch full retrain={scratch_time:.2f}s, real unlearning retrain={real_retrain_time:.2f}s")

    report = {
        'class_name': class_name,
        'source_project': source_project,
        'unlearned_project': unlearned_project,
        'scratch_project': scratch_result['scratch_project'],
        'affected_shard': shard_idx + 1,
        'metrics': {
            'parameter_distance': param_distance,
            'prediction_agreement': pred_agreement,
            'output_distribution_kl': output_kl,
            'mia_auc': {k: v[2] for k, v in mia_results.items()},
            'deleted_class_prediction_agreement': deleted_agreement,
        },
        'efficiency': {
            'scratch_full_retrain_seconds': scratch_time,
            'real_unlearning_retrain_seconds': real_retrain_time,
            'empirical_speedup': (scratch_time / real_retrain_time) if real_retrain_time > 0 else None,
        },
        'notes': [
            "Parameter distance is not expected to be ~0: the scratch script's RNG "
            "consumption sequence differs from the original run's (plan section 7). "
            "Prediction agreement, output-distribution distance, and MIA are the "
            "primary exactness evidence, since they measure behavioral equivalence "
            "rather than weight equality.",
        ],
        'timestamp': datetime.now().isoformat(),
    }

    results_dir = os.path.join(str(REPO_ROOT), "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"{class_name}_exactness_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    create_exactness_comparison_chart(param_distance, pred_agreement, output_kl, class_name, results_dir)
    create_mia_roc_chart(mia_results, class_name, results_dir)
    create_exactness_confusion_matrices(y_test, preds_original, preds_unlearned, preds_scratch, class_names, class_name, results_dir)
    create_efficiency_comparison_chart(scratch_time, real_retrain_time, num_shards, src_metadata['num_slices'], class_name, results_dir)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W8: exactness proof harness")
    parser.add_argument('--project-name', type=str, required=True, help="Trained, untouched source project")
    parser.add_argument('--class-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, default=config.MODEL_TYPE)
    args = parser.parse_args()

    run_exactness_eval(args.project_name, args.class_name, args.model_name)
