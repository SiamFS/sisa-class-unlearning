import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torchvision.transforms as T

# Import global configuration
import config

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
)
from training.create_model import save_model_pytorch, load_model_pytorch, DEVICE
from training.train_gating_model import train_gating

# Enhanced logging class to save terminal output to files
class TrainingLogger:
    def __init__(self, log_file="training.txt"):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        # Create log file if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) if os.path.dirname(log_file) else ".", exist_ok=True)
        
        # Open file in write mode to overwrite previous sessions
        self.file = open(log_file, 'w', encoding='utf-8')
        
        # Write session header
        self.file.write(f"{'='*80}\n")
        self.file.write(f"SISA Training Session - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write(f"{'='*80}\n")
        self.file.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        if hasattr(self, 'file'):
            self.file.close()
    
    def __del__(self):
        self.close()

# --- Main Configuration ---
project_name = config.PROJECT_NAME
MODEL_NAME = config.MODEL_TYPE
base_dir = f"../{config.PROJECTS_DIR}/{project_name}"
sisa_data_dir = f"{base_dir}/sisa_data"
models_dir = f"{base_dir}/models"
reports_dir = f"{base_dir}/data_info"

# --- Load SISA Metadata & Define Transforms ---
with open(os.path.join(sisa_data_dir, "metadata.json"), 'r') as f:
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
        y_slice_path = f"../{config.PROJECTS_DIR}/{project_name}/sisa_data/shards/shard_{shard_idx+1}/slice_{slice_idx}_y.npy"
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
    """
    Filter validation data to only include classes seen up to current slice.
    This ensures fair evaluation following incremental learning standards.
    
    Args:
        shard_idx: 0-based shard index
        slice_idx: 0-based slice index
        shard_metadatas: loaded shard metadata (not used anymore)
        validation_data: tuple of (x_val, y_val)
    
    Returns:
        Filtered (x_val_filtered, y_val_filtered) containing only relevant classes
    """
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

def get_gating_routed_validation_data(gating_model, x_val, y_val, target_shard_idx, 
                                      cumulative_classes, dataset_mean, dataset_std, 
                                      confidence_threshold=0.6):
    """
    Use pre-trained gating network to intelligently route validation samples to appropriate shard.
    This provides more targeted validation than simple class filtering.
    
    Args:
        gating_model: Pre-trained gating network (set to None to fall back to standard method)
        x_val, y_val: Full validation dataset
        target_shard_idx: Which shard we want validation samples for
        cumulative_classes: Classes that should be known up to current slice
        dataset_mean, dataset_std: Normalization parameters
        confidence_threshold: Minimum confidence for gating routing decision
    
    Returns:
        Intelligently routed validation samples for the target shard
    """
    if gating_model is None:
        # Use standard class filtering if gating network unavailable
        val_mask = np.isin(y_val, cumulative_classes)
        x_val_filtered = x_val[val_mask]
        y_val_filtered = y_val[val_mask]
        print(f"   Standard validation (gating unavailable): {len(cumulative_classes)} classes, {len(x_val_filtered)} samples")
        return x_val_filtered, y_val_filtered
    
    # Use gating network for intelligent routing
    gating_transform = T.Compose([T.Normalize(dataset_mean, dataset_std)])
    
    with torch.no_grad():
        # Convert to tensor and transform
        x_val_tensor = torch.from_numpy(x_val.astype(np.float32)).to(DEVICE)
        x_val_normalized = gating_transform(x_val_tensor)
        
        # Get gating predictions
        gating_outputs = gating_model(x_val_normalized)
        gating_probs = torch.softmax(gating_outputs, dim=1)
        predicted_shards = torch.argmax(gating_probs, dim=1)
        max_confidences = torch.max(gating_probs, dim=1)[0]
        
        # Create routing mask - route to target shard with minimum confidence
        routing_mask = (predicted_shards == target_shard_idx) & (max_confidences >= confidence_threshold)
        routing_mask_np = routing_mask.cpu().numpy()
        
        # Apply gating-based routing
        x_val_gated = x_val[routing_mask_np]
        y_val_gated = y_val[routing_mask_np]
        
        # Then filter by incremental learning classes (what slice should know)
        if len(x_val_gated) > 0:
            incremental_mask = np.isin(y_val_gated, cumulative_classes)
            x_val_final = x_val_gated[incremental_mask]
            y_val_final = y_val_gated[incremental_mask]
        else:
            x_val_final, y_val_final = np.array([]), np.array([])
    
    print(f"   Gating-routed validation: {len(x_val_final)} samples (from {len(x_val_gated)} gating-routed, {len(cumulative_classes)} classes)")
    
    return x_val_final, y_val_final

def get_true_label_validation_data(shard_idx, cumulative_classes, validation_data):
    """
    Filter validation data using true labels to route to the correct shard.
    """
    x_val, y_val = validation_data
    
    # Load shard metadata to get which classes this shard handles
    shard_metadata_path = f"../{config.PROJECTS_DIR}/{project_name}/sisa_data/shards/shard_{shard_idx+1}/metadata.json"
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

def evaluate_slice_incrementally(model, shard_idx, slice_idx, shard_metadatas, test_data, class_names, gating_model=None, all_shard_models=None):
    """
    Evaluate a specific slice model against only the classes it should know.
    Uses gating network if provided, otherwise direct model evaluation.
    
    Args:
        model: trained model for this slice
        shard_idx: 0-based shard index
        slice_idx: 0-based slice index
        shard_metadatas: loaded shard metadata
        test_data: tuple of (x_test, y_test)
        class_names: list of all class names
        gating_model: optional gating model for SISA evaluation
        all_shard_models: optional list of all shard models
    
    Returns:
        Accuracy on known classes for this slice
    """
    x_test, y_test = test_data
    
    # Get classes that should be known at this slice
    known_classes = get_cumulative_classes_up_to_slice(shard_idx, slice_idx)
    
    # Filter test data to only include known classes
    test_mask = np.isin(y_test, known_classes)
    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]
    
    if len(x_test_filtered) == 0:
        print(f"   No test samples for slice {slice_idx+1} classes")
        return 0.0
    
    # Use gating network evaluation if available
    if gating_model is not None and all_shard_models is not None:
        # Use the provided shard models list
        shard_models_for_eval = all_shard_models
        
        # Set models to eval mode
        for m in shard_models_for_eval:
            if m is not None:
                m.eval()
        gating_model.eval()
        
        all_preds = []
        batch_size = config.BATCH_SIZE
        with torch.no_grad():
            for i in range(0, len(x_test_filtered), batch_size):
                batch_x = torch.from_numpy(x_test_filtered[i:i+batch_size]).float()
                batch_x_normalized = eval_transforms(batch_x).to(DEVICE)
                
                # Use SISA gating evaluation
                batch_preds, _ = _run_sisa_batch(
                    batch_x_normalized, shard_models_for_eval, gating_model, class_names,
                    threshold=None
                )
                all_preds.extend(batch_preds.cpu().numpy())
        
        all_preds = np.array(all_preds)
        correct = np.sum(all_preds == y_test_filtered)
        total = len(y_test_filtered)
    else:
        # Use direct model evaluation
        model.eval()
        correct = 0
        total = len(x_test_filtered)
        
        batch_size = config.BATCH_SIZE
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_x = torch.from_numpy(x_test_filtered[i:i+batch_size]).float()
                batch_x_normalized = eval_transforms(batch_x).to(DEVICE)
                batch_y = y_test_filtered[i:i+batch_size]
                
                outputs = model(batch_x_normalized)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted.cpu().numpy() == batch_y).sum()
    
    accuracy = correct / total
    known_class_names = [class_names[idx] for idx in known_classes]
    
    eval_method = "gating network" if gating_model is not None else "direct model"
    print(f"   Slice {slice_idx + 1} incremental test accuracy ({eval_method}): {accuracy:.4f} "
          f"({correct}/{total}) on classes {known_class_names}")
    
    return accuracy

def evaluate_with_gating_network(shard_models, gating_model, class_names, threshold=None):
    print("\n" + "="*20 + " Final SISA System Evaluation " + "="*20)
    x_test = np.load(os.path.join(sisa_data_dir, "test_data/x_test.npy"))
    y_test = np.load(os.path.join(sisa_data_dir, "test_data/y_test.npy"))

    for model in shard_models: model.eval()
    gating_model.eval()

    all_final_preds = []
    all_true_labels = []

    batch_size = config.BATCH_SIZE  # From global config
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            batch_x = torch.from_numpy(x_test[i:i+batch_size]).float()
            batch_x_normalized = eval_transforms(batch_x).to(DEVICE)
            batch_y = y_test[i:i+batch_size]

            batch_final_preds, _ = _run_sisa_batch(
                batch_x_normalized, shard_models, gating_model, class_names, threshold
            )

            all_final_preds.extend(batch_final_preds.cpu().numpy())
            all_true_labels.extend(batch_y)
            
    all_final_preds = np.array(all_final_preds)
    all_true_labels = np.array(all_true_labels)

    # Calculate accuracy using true SISA gating method
    gating_accuracy = np.mean(all_final_preds == all_true_labels)
    
    print("\n" + "-"*60)
    print("Final Classification Report:")
    print("-"*60)
    print("Using True SISA Gating Method (specialist routing)")
    
    report = classification_report(all_true_labels, all_final_preds, 
                                 target_names=class_names, 
                                 labels=np.arange(len(class_names)), 
                                 zero_division=0)
    final_accuracy = gating_accuracy
    
    print(report)
    print(f"\nFinal SISA System Accuracy: {final_accuracy:.4f}")
    print("-"*60)

    return final_accuracy, final_accuracy

def check_class_balance_and_augmentation(shard_idx, class_names):
    """
    Check class balance in a shard and determine if augmentation is needed.
    Returns augmentation settings based on balance analysis.
    
    Args:
        shard_idx: 0-based shard index
        class_names: List of all class names
        
    Returns:
        dict: Augmentation configuration with conservative settings
    """
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
        print("   Classes are perfectly balanced - NO AUGMENTATION")
        return None
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
    # Initialize logging to save output to training.txt
    logger = TrainingLogger("training.txt")
    sys.stdout = logger
    
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
        
        current_model, replay_buffer, shard_histories = None, None, []
        
        # Initialize replay buffer based on configuration
        if config.USE_SMART_REPLAY:
            from training.smart_replay import create_smart_replay_buffer
            replay_buffer = create_smart_replay_buffer(config)
            print("   - Using smart replay buffer with importance + temporal sampling")
        else:
            replay_buffer = {}
            print("   - Using traditional random replay buffer")
        
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
                
                # Use static replay ratio from config
                current_replay_ratio = config.REPLAY_RATIO
                print(f"   - Using replay ratio: {current_replay_ratio}")
                
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
                    use_smart_replay=config.USE_SMART_REPLAY,
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
                # Copy available final models
                for idx, model in enumerate(all_shard_final_models):
                    if idx < config.NUM_SHARDS:
                        shard_models_for_eval[idx] = model
                # Set current model for this shard
                shard_models_for_eval[i] = current_model
                slice_test_accuracy = evaluate_slice_incrementally(
                    current_model, i, j, shard_metadatas, test_data, class_names,
                    gating_model=gating_model, all_shard_models=shard_models_for_eval
                )
                
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
                
                # Update replay buffer with current slice data
                if config.USE_SMART_REPLAY and hasattr(replay_buffer, 'add_samples'):
                    # Smart replay buffer - add with importance scoring
                    replay_buffer.add_samples(x_slice, y_slice, current_model, DEVICE)
                else:
                    # Traditional replay buffer - simple class-wise storage
                    unique_labels_in_slice = np.unique(y_slice)
                    for label in unique_labels_in_slice:
                        mask = (y_slice == label)
                        class_x_data = x_slice[mask]
                        class_y_data = y_slice[mask]
                        if label in replay_buffer:
                            replay_buffer[label]['X'] = np.vstack([replay_buffer[label]['X'], class_x_data])
                            replay_buffer[label]['y'] = np.concatenate([replay_buffer[label]['y'], class_y_data])
                        else:
                            replay_buffer[label] = {'X': class_x_data, 'y': class_y_data}
                        
            save_model_pytorch(current_model, final_model_path)
            all_shard_histories.append(shard_histories)
            print(f"--- Shard {i+1} training complete. Final model saved. ---")
            
        all_shard_final_models.append(current_model)

    print("\n" + "="*20 + " Final Evaluation with Pre-trained Gating Network " + "="*20)
    
    # Use the gating network we trained at the beginning
    if gating_model is not None:
        classified_accuracy, overall_accuracy = evaluate_with_gating_network(all_shard_final_models, gating_model, class_names)
    else:
        print("Gating network not available for final evaluation")
        classified_accuracy, overall_accuracy = 0.0, 0.0

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
        gating_model, 
        x_test, 
        y_test, 
        class_names, 
        reports_dir, 
        'final_evaluation'
    )

    if gating_model is not None:
        print("\n" + "=" * 50)
        print("ANALYZING GATING ROUTING DISTRIBUTIONS")
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
        gating_model,
        x_test,
        y_test,
        class_names,
        reports_dir,
        'final_evaluation'
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
    print("Training log saved to: training.txt")
    print("=" * 70)
    
    # Cleanup logging
    sys.stdout = logger.terminal
    logger.close()
    print("Training completed! Check training.txt for detailed logs.")