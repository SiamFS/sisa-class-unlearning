import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torchvision.transforms as T

# Import global configuration
import config
from utils.seeding import set_seed

set_seed(config.SEED)

from training.train_model import (
    train_model,
    _run_sisa_batch,
)
from plots import (
    create_training_visualizations,
    create_confusion_matrix,
    create_shard_confusion_matrix,
    create_gating_routing_barplots,
    create_overall_sisa_confusion_matrix,
    create_overall_sisa_roc_curve,
    create_overall_sisa_training_curves,
    load_shard_class_indices,
    fit_ensemble_params,
)
from training.create_model import save_model_pytorch, load_model_pytorch, DEVICE
from training.train_gating_model import train_gating
from training.replay_buffer import add_to_replay_buffer, compute_replay_ratio
from utils.run_logging import setup_run_logging

# --- Main Configuration ---
project_name = config.PROJECT_NAME
MODEL_NAME = config.MODEL_TYPE
base_dir = os.path.join(config.PROJECTS_DIR, project_name)
sisa_data_dir = os.path.join(base_dir, "sisa_data")
models_dir = os.path.join(base_dir, "models")
reports_dir = os.path.join(base_dir, "data_info")

# --- Load SISA Metadata & Define Transforms ---
sisa_metadata_path = os.path.join(sisa_data_dir, "metadata.json")
with open(sisa_metadata_path, 'r') as f:
    metadata = json.load(f)
num_shards = metadata['num_shards']
num_slices = metadata['num_slices']
class_names = metadata['class_names']

if 'normalization_mean' in metadata and 'normalization_std' in metadata:
    DATASET_MEAN = metadata['normalization_mean']
    DATASET_STD = metadata['normalization_std']
    print("Loaded dynamic normalization stats from metadata.")
else:
    raise Exception("Normalization stats not found in metadata. Please run data processing first to compute normalization values.")

eval_transforms = T.Compose([T.Normalize(DATASET_MEAN, DATASET_STD)])

def load_slice(shard_idx, slice_idx):
    path = os.path.join(sisa_data_dir, f"shards/shard_{shard_idx+1}/slice_{slice_idx}_x.npy")
    if not os.path.exists(path): return None, None
    X = np.load(path)
    y = np.load(path.replace('_x.npy', '_y.npy'))
    return X, y

def get_actual_classes_in_slice(shard_idx, slice_idx):
    """
    Get the actual classes present in a specific slice by reading the slice data.
    This gives the true classes present in each slice, not assumptions.
    
    Args:
        shard_idx: 0-based shard index
        slice_idx: 0-based slice index  
    
    Returns:
        List of class indices actually present in this specific slice
    """
    try:
        # Load the actual slice labels to see what classes are present
        y_slice_path = os.path.join(config.PROJECTS_DIR, project_name, "sisa_data", "shards", f"shard_{shard_idx+1}", f"slice_{slice_idx}_y.npy")
        y_slice = np.load(y_slice_path)
        
        # Get unique classes in this slice
        slice_classes = sorted(np.unique(y_slice).tolist())
        return slice_classes
        
    except FileNotFoundError:
        print(f"Warning: Could not load slice {slice_idx} for shard {shard_idx}")
        return []

def get_cumulative_classes_up_to_slice(shard_idx, slice_idx):
    """
    Get all classes that should be known up to and including the specified slice.
    This reads actual slice content to determine cumulative classes.
    
    Args:
        shard_idx: 0-based shard index
        slice_idx: 0-based slice index  
    
    Returns:
        List of class indices that should be known by this slice (cumulative)
    """
    cumulative_classes = set()
    
    # Accumulate classes from slice 0 up to and including current slice
    for s_idx in range(slice_idx + 1):
        slice_classes = get_actual_classes_in_slice(shard_idx, s_idx)
        cumulative_classes.update(slice_classes)
    
    return sorted(list(cumulative_classes))

def get_incremental_validation_data(shard_idx, slice_idx, shard_metadatas, validation_data):

    x_val, y_val = validation_data
    
    # Get classes that should be known at this slice (cumulative)
    known_classes = get_cumulative_classes_up_to_slice(shard_idx, slice_idx)
    
    # Create mask for validation samples belonging to known classes
    val_mask = np.isin(y_val, known_classes)
    
    # Filter validation data
    x_val_filtered = x_val[val_mask]
    y_val_filtered = y_val[val_mask]
    
    print(f"   Incremental validation: {len(known_classes)} classes, {len(x_val_filtered)} samples")
    
    return x_val_filtered, y_val_filtered

def get_true_label_validation_data(shard_idx, cumulative_classes, validation_data):

    x_val, y_val = validation_data
    
    # Load shard metadata to get which classes this shard handles
    shard_metadata_path = os.path.join(config.PROJECTS_DIR, project_name, "sisa_data", "shards", f"shard_{shard_idx+1}", "metadata.json")
    try:
        with open(shard_metadata_path, 'r') as f:
            shard_metadata = json.load(f)
        shard_responsible_classes = set(shard_metadata['class_indices_present'])
    except FileNotFoundError:
        shard_responsible_classes = set(cumulative_classes)
    
    # Filter validation samples: must belong to this shard AND be in cumulative classes
    valid_classes_for_shard = shard_responsible_classes.intersection(set(cumulative_classes))
    val_mask = np.isin(y_val, list(valid_classes_for_shard))
    
    x_val_filtered = x_val[val_mask]
    y_val_filtered = y_val[val_mask]
    
    print(f"   Validation: {len(valid_classes_for_shard)} classes, {len(x_val_filtered)} samples")
    
    return x_val_filtered, y_val_filtered

def evaluate_with_self_routing(shard_models, shard_class_indices, class_names, threshold=None, gating_model=None):
    print("\n" + "="*20 + " Final SISA System Evaluation " + "="*20)
    x_test = np.load(os.path.join(sisa_data_dir, "test_data/x_test.npy"))
    y_test = np.load(os.path.join(sisa_data_dir, "test_data/y_test.npy"))

    for model in shard_models:
        if model is not None:
            model.eval()

    all_final_preds = []
    all_final_probs = []
    all_true_labels = []

    batch_size = config.BATCH_SIZE  # From global config
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch_x = torch.from_numpy(x_test[i:i+batch_size]).float()
            batch_x_normalized = eval_transforms(batch_x).to(DEVICE)
            batch_y = y_test[i:i+batch_size]

            batch_final_preds, batch_probs = _run_sisa_batch(
                batch_x_normalized, shard_models, class_names, shard_class_indices, threshold,
                gating_model=gating_model,
            )

            all_final_preds.extend(batch_final_preds.cpu().numpy())
            all_final_probs.append(batch_probs.cpu().numpy())
            all_true_labels.extend(batch_y)

    all_final_preds = np.array(all_final_preds)
    all_final_probs = np.concatenate(all_final_probs, axis=0)
    all_true_labels = np.array(all_true_labels)
    # W9: hand this single inference pass to the confusion-matrix/ROC plots below
    # instead of each of them re-running _run_sisa_batch over the same test set.
    precomputed_eval = (all_final_preds, all_final_probs, all_true_labels)

    # Calculate accuracy using true SISA self-routing method
    self_routing_accuracy = np.mean(all_final_preds == all_true_labels)

    print("\n" + "-"*60)
    print("Final Classification Report:")
    print("-"*60)
    print("Using True SISA Self-Routing Method (masked specialist confidence)")

    report = classification_report(all_true_labels, all_final_preds,
                                 target_names=class_names,
                                 labels=np.arange(len(class_names)),
                                 zero_division=0)
    # Dict form for structured storage (W7) -- same shape _get_training_metrics
    # used to hand-reconstruct by regex from the printed string above.
    report_dict = classification_report(all_true_labels, all_final_preds,
                                 target_names=class_names,
                                 labels=np.arange(len(class_names)),
                                 zero_division=0, output_dict=True)
    final_accuracy = self_routing_accuracy

    print(report)
    print(f"\nFinal SISA System Accuracy: {final_accuracy:.4f}")
    print("-"*60)

    return final_accuracy, final_accuracy, report_dict, precomputed_eval

def check_class_balance_and_augmentation(shard_idx, class_names):

    print(f"\n--- Checking Class Balance for Shard {shard_idx+1} ---")
    
    # Load all slice data for this shard
    all_labels = []
    for slice_idx in range(num_slices):
        _, y_slice = load_slice(shard_idx, slice_idx)
        if y_slice is not None:
            all_labels.extend(y_slice)
    
    if not all_labels:
        print("   No data found for balance analysis")
        return None
    
    # Calculate class distribution
    from collections import Counter
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    print(f"   Total samples: {total_samples}")
    print("   Class distribution:")
    
    # Calculate percentages and find min/max
    class_percentages = {}
    min_percentage = float('inf')
    max_percentage = 0
    
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = (count / total_samples) * 100
        class_percentages[class_idx] = percentage
        min_percentage = min(min_percentage, percentage)
        max_percentage = max(max_percentage, percentage)
        
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        print(f"     {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Calculate balance metrics
    balance_ratio = min_percentage / max_percentage if max_percentage > 0 else 1.0
    std_dev = np.std(list(class_percentages.values()))
    
    print(f"   Balance ratio: {balance_ratio:.3f}")
    print(f"   Standard deviation: {std_dev:.3f}")
    print(f"   Min/Max percentages: {min_percentage:.3f}% / {max_percentage:.3f}%")
    
    # Determine augmentation strategy based on balance
    # Conservative thresholds: only augment if significantly unbalanced
    PERFECT_BALANCE_THRESHOLD = 0.95  # Classes within 95% of each other = perfectly balanced
    BALANCE_THRESHOLD = 0.85  # Classes within 85% of each other = well balanced
    STD_THRESHOLD = 1.5       # Standard deviation threshold for perfect balance
    
    if balance_ratio >= PERFECT_BALANCE_THRESHOLD and std_dev <= STD_THRESHOLD:
        # W19: balanced no longer means "no augmentation". Augmentation is a regularizer
        # first and an imbalance remedy second, and class-isolated CIFAR shards are
        # perfectly balanced by construction -- so this branch always fired and the
        # pipeline never augmented at all. Baseline = standard CIFAR crop + flip, no jitter.
        print("   Classes are perfectly balanced - using baseline geometric augmentation")
        return config.get_augmentation_config('baseline')
    elif balance_ratio >= BALANCE_THRESHOLD and std_dev <= 2.5:
        print("   Classes are well balanced - using minimal augmentation")
        return config.get_augmentation_config('minimal')
    elif balance_ratio >= 0.75 and std_dev <= 4.0:
        print("   Classes moderately unbalanced - using light augmentation")
        return config.get_augmentation_config('light')
    else:
        print("   Classes significantly unbalanced - using moderate augmentation")
        return config.get_augmentation_config('moderate')

def update_shard_metadata_with_balance(shard_idx, balance_info):
    """Update shard metadata with balance analysis results"""
    metadata_path = os.path.join(sisa_data_dir, f"shards/shard_{shard_idx+1}/metadata.json")
    
    try:
        with open(metadata_path, 'r') as f:
            shard_metadata = json.load(f)
    except FileNotFoundError:
        shard_metadata = {}
    
    # Add balance information
    shard_metadata['balance_analysis'] = balance_info
    shard_metadata['last_balance_check'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(metadata_path, 'w') as f:
        json.dump(shard_metadata, f, indent=2)
    
    print(f"   Updated shard {shard_idx+1} metadata with balance information")

if __name__ == "__main__":
    # Project-scoped, timestamped logging (W10).
    _restore_logging, _log_path = setup_run_logging(os.path.join(base_dir, "logs"), "training")

    print("Enhanced SISA Training with Balance-Based Augmentation")
    print("="*70)
    
    overall_start_time = time.time()
    gating_training_start_time = 0
    gating_training_end_time = 0
    base_model_training_start_time = 0
    base_model_training_end_time = 0
    pure_training_time = 0  # NEW: Track pure model training time (excludes evaluations)
    all_shard_final_models = []
    all_shard_histories = []

    shard_metadatas = []
    for i in range(num_shards):
        with open(os.path.join(sisa_data_dir, f"shards/shard_{i+1}/metadata.json"), 'r') as f:
            shard_metadatas.append(json.load(f))

    # GATING-FIRST APPROACH: Train Gating Network Before Shard Training
    print("\n" + "="*60)
    print("TRAINING GATING NETWORK FIRST FOR INTELLIGENT VALIDATION ROUTING")
    print("="*60)
    
    gating_training_start_time = time.time()
    gating_model_path, pure_gating_training_time = train_gating(
        num_shards=num_shards,
        base_dir=base_dir,
        num_slices=num_slices,
        dataset_mean=DATASET_MEAN,
        dataset_std=DATASET_STD
    )
    gating_training_end_time = time.time()
    gating_training_time = gating_training_end_time - gating_training_start_time
    print(f"Gating Network Training Time (with I/O): {gating_training_time:.2f} seconds")
    print(f"Gating Network Training Time (pure): {pure_gating_training_time:.2f} seconds")
    
    if gating_model_path is None:
        print("Error: Gating network training failed. Falling back to standard validation.")
        gating_model = None
    else:
        print("Gating Network trained. Loading for validation routing...")
        gating_model, _ = load_model_pytorch(gating_model_path, num_shards=num_shards)
        gating_model.eval()
    
    print("="*60)

    # Start timing base model training
    base_model_training_start_time = time.time()

    for i in range(num_shards):
        print("\n" + "="*20 + f" Training Shard {i+1}/{num_shards} " + "="*20)
        shard_dir = os.path.join(models_dir, f"shard_{i+1}")
        os.makedirs(shard_dir, exist_ok=True)
        active_classes = shard_metadatas[i]['class_indices_present']
        print(f"This shard is a specialist for {len(active_classes)} classes.")
        
        # Check class balance and determine augmentation strategy
        augmentation_config = check_class_balance_and_augmentation(i, class_names)
        
        # Store balance analysis in shard metadata
        balance_info = {
            'augmentation_config': augmentation_config,
            'active_classes': active_classes,
            'num_classes': len(active_classes)
        }
        update_shard_metadata_with_balance(i, balance_info)

        shard_histories = []
        current_model = None
        replay_buffer = {}
        replay_seen_counts = {}  # per-class "samples ever seen" count, for reservoir sampling
        replay_rng = np.random.default_rng(config.SEED)

        final_model_path = os.path.join(shard_dir, f"final_model_shard{i+1}_{MODEL_NAME}.pth")
        
        if os.path.exists(final_model_path):
            print(f"--- Loading pre-trained model for Shard {i+1} ---")
            current_model, _ = load_model_pytorch(final_model_path)
            # Skip training since model is already trained
            all_shard_final_models.append(current_model)
            all_shard_histories.append([])  # Empty history for pre-trained model
            continue
        else:
            validation_data = (np.load(os.path.join(sisa_data_dir, "validation_data/x_validation.npy")), np.load(os.path.join(sisa_data_dir, "validation_data/y_validation.npy")))
            for j in range(num_slices):
                print(f"\n--- Training Slice {j + 1} of Shard {i+1} ---")
                x_slice, y_slice = load_slice(i, j)
                if x_slice is None or len(x_slice) == 0: 
                    print("   - Slice is empty, skipping.")
                    continue
                print(f"   - Slice contains {len(x_slice)} samples.")
                
                # Get classes known up to this slice for proper active_classes parameter
                known_classes = get_cumulative_classes_up_to_slice(i, j)
                
                # Get validation data for this shard using true labels
                incremental_validation_data = get_true_label_validation_data(
                    shard_idx=i,
                    cumulative_classes=known_classes,
                    validation_data=validation_data
                )
                
                # W18: class-balanced replay ratio, derived from how many classes are in
                # the buffer vs. in this slice. A fixed 0.3 gave the newest class ~9x the
                # per-class batch share of each older one, which is what drove the measured
                # task-recency bias. Deterministic (no RNG), so exactness is unaffected.
                current_replay_ratio = compute_replay_ratio(replay_buffer, y_slice)
                print(f"   - Using replay ratio: {current_replay_ratio:.3f} "
                      f"({len(replay_buffer)} replay class(es), {len(np.unique(y_slice))} in slice)")
                
                # START: Track pure training time (before train_model call)
                slice_train_start = time.time()
                
                # Enhanced training parameters for better accuracy
                current_model, history = train_model(
                    x_slice, y_slice, model=current_model, 
                    epochs=config.MAX_EPOCHS,  # From global config
                    batch_size=config.BATCH_SIZE,  # From global config
                    lr=config.LEARNING_RATE,  # From global config
                    validation_data=incremental_validation_data,  # Use incremental validation
                    active_classes=known_classes,  # Use incremental classes, not all shard classes
                    replay_buffer=replay_buffer,
                    replay_ratio=current_replay_ratio,  # Use dynamic ratio
                    dataset_mean=DATASET_MEAN,
                    dataset_std=DATASET_STD,
                    training_type='incremental' if current_model is not None else 'fresh',
                    augmentation_config=augmentation_config,  # Use balance-based augmentation
                    device=DEVICE
                )
                
                # END: Track pure training time (after train_model call)
                slice_train_end = time.time()
                pure_training_time += (slice_train_end - slice_train_start)
                
                shard_histories.append(history)
                
                slice_checkpoint_path = os.path.join(shard_dir, f"slice_{j}_model_{MODEL_NAME}.pth")
                slice_metadata = {'shard_id': i, 'slice_id': j}
                save_model_pytorch(current_model, slice_checkpoint_path, metadata=slice_metadata)
                print(f"   - Saved checkpoint: {os.path.basename(slice_checkpoint_path)}")
                
                # Incremental test evaluation for this slice (ML standard)
                test_data = (np.load(os.path.join(sisa_data_dir, "test_data/x_test.npy")), 
                           np.load(os.path.join(sisa_data_dir, "test_data/y_test.npy")))
                # Prepare shard models for gating evaluation
                shard_models_for_eval = [None] * config.NUM_SHARDS  # Initialize with None for all shards
                # REMOVED: slice-level evaluation violates SISA architecture
                # Slices are internal data partitions - only final shard model should be evaluated
                
                # Generate visualizations for the last slice of each shard
                if j == num_slices - 1:  # Last slice
                    print(f"   - Generating visualizations for final slice of Shard {i+1}...")
                    
                    # Create training visualizations (loss/accuracy curves)
                    if history is not None:
                        create_training_visualizations(
                            history, i, j, reports_dir, 'training'
                        )
                    
                    if current_model is not None:
                        x_test_full, y_test_full = test_data
                        create_shard_confusion_matrix(
                            current_model,
                            x_test_full,
                            y_test_full,
                            class_names,
                            i,
                            reports_dir,
                            active_classes,
                        )
                
                # Update replay buffer with current slice data (capped per class, W4)
                add_to_replay_buffer(replay_buffer, x_slice, y_slice, replay_rng,
                                      config.MAX_REPLAY_SAMPLES_PER_CLASS, replay_seen_counts)
                        
            save_model_pytorch(current_model, final_model_path)
            all_shard_histories.append(shard_histories)
            print(f"--- Shard {i+1} training complete. Final model saved. ---")
            
        all_shard_final_models.append(current_model)

    print("\n" + "="*20 + " Final Evaluation with Gating-Network Routing " + "="*20)

    # W16: the gating network trained above is the real router now (measured
    # 58%->70% combined accuracy over confidence-based self-routing -- see
    # IMPLEMENTATION_PLAN.md W15/W16). Falls back to confidence-based
    # self-routing only if gating training itself failed above (gating_model
    # is None in that case).
    shard_class_indices = load_shard_class_indices(sisa_data_dir, num_shards)

    # W24: fit the gate/cosine routing blend on validation now that the specialists
    # exist (the gate is trained before them, so this cannot happen any earlier).
    # Attaches two scalars to gating_model; _run_sisa_batch falls back to plain gate
    # routing if this is skipped or if the blend loses to the gate on validation.
    if gating_model is not None and getattr(config, 'ROUTING_MODE', 'gating') == 'ensemble':
        x_val_ens = np.load(os.path.join(sisa_data_dir, "validation_data/x_validation.npy"))
        y_val_ens = np.load(os.path.join(sisa_data_dir, "validation_data/y_validation.npy"))
        fit_ensemble_params(
            all_shard_final_models, gating_model, shard_class_indices, class_names,
            x_val_ens, y_val_ens, T.Normalize(DATASET_MEAN, DATASET_STD),
        )

    classified_accuracy, overall_accuracy, final_report_dict, final_eval_precomputed = evaluate_with_self_routing(
        all_shard_final_models, shard_class_indices, class_names, gating_model=gating_model
    )

    # End timing for base model training (AFTER final evaluation, BEFORE visualizations)
    base_model_training_end_time = time.time()
    base_model_training_time = base_model_training_end_time - base_model_training_start_time

    # Create overall SISA system confusion matrix
    print("\n" + "=" * 50)
    print("CREATING OVERALL SISA SYSTEM CONFUSION MATRIX")
    print("=" * 50)

    # Load test data for final evaluation
    test_data_dir = os.path.join(sisa_data_dir, "test_data")
    x_test = np.load(os.path.join(test_data_dir, "x_test.npy"))
    y_test = np.load(os.path.join(test_data_dir, "y_test.npy"))

    create_overall_sisa_confusion_matrix(
        all_shard_final_models,
        shard_class_indices,
        x_test,
        y_test,
        class_names,
        reports_dir,
        'final_evaluation',
        precomputed=final_eval_precomputed,
    )

    if gating_model is not None:
        print("\n" + "=" * 50)
        print("ANALYZING GATING ROUTING DISTRIBUTIONS (this is the real router used above)")
        print("=" * 50)

        create_gating_routing_barplots(
            gating_model,
            x_test,
            y_test,
            class_names,
            reports_dir,
            'final_evaluation'
        )

    # Create overall SISA system ROC curves
    print("\n" + "=" * 50)
    print("CREATING OVERALL SISA SYSTEM ROC CURVES")
    print("=" * 50)

    create_overall_sisa_roc_curve(
        all_shard_final_models,
        shard_class_indices,
        x_test,
        y_test,
        class_names,
        reports_dir,
        'final_evaluation',
        precomputed=final_eval_precomputed,
    )

    # Create overall SISA system training curves
    print("\n" + "=" * 50)
    print("CREATING OVERALL SISA SYSTEM TRAINING CURVES")
    print("=" * 50)
    
    create_overall_sisa_training_curves(
        all_shard_histories,
        reports_dir,
        'final_evaluation'
    )

    total_time = time.time() - overall_start_time

    # W7: persist every reported number as structured JSON -- unlearning reads
    # this instead of scraping training.txt.
    training_metrics = {
        "final_accuracy": float(classified_accuracy),
        "gating_training_time_with_io": float(gating_training_time),
        "gating_training_time_pure": float(pure_gating_training_time),
        "base_model_training_time_with_eval": float(base_model_training_time),
        "base_model_training_time_pure": float(pure_training_time),
        "total_training_time": float(total_time),
        "classification_report": final_report_dict,
        "timestamp": datetime.now().isoformat(),
    }
    training_metrics_path = os.path.join(sisa_data_dir, "training_metrics.json")
    with open(training_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(training_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("Enhanced SISA Training Completed")
    print("=" * 70)
    print("\nTIMING BREAKDOWN:")
    print(f"Gating Network Training Time (with I/O): {gating_training_time:.2f} seconds")
    print(f"Gating Network Training Time (pure): {pure_gating_training_time:.2f} seconds")
    print(f"Base Model Training Time (with eval): {base_model_training_time:.2f} seconds")
    print(f"Base Model Training Time (pure): {pure_training_time:.2f} seconds")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"\nFinal SISA System Accuracy: {classified_accuracy:.4f}")
    print("Confidence Threshold Used: disabled for final evaluation")
    print(f"Models and reports saved to: {base_dir}")
    print(f"Training metrics saved to: {training_metrics_path}")
    print(f"Log saved to: {_log_path}")
    print("=" * 70)

    _restore_logging()