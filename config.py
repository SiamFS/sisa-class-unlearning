import os as _os

# Emoji in progress messages cannot be encoded by legacy consoles (cp1252 on Windows).
def _force_utf8_console():
    import sys

    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError, ValueError):
            pass


_force_utf8_console()

# ================================================================================
# SISA ARCHITECTURE PARAMETERS
# ================================================================================
NUM_SHARDS = 2
NUM_SLICES_PER_SHARD = 3
# ================================================================================
# CONFIDENCE THRESHOLDS (GATING METHOD ONLY)
# ================================================================================
CONFIDENCE_THRESHOLD = 0.60  # Used during unlearning to guard against routing deleted classes
GATING_MARGIN_THRESHOLD = 0.08  # Require this margin between top shards to trust the primary pick
PRIMARY_SPECIALIST_TEMPERATURE = 1.1  # Softens reported probabilities only; monotone, so predictions are unchanged
SPECIALIST_EVAL_TEMPERATURE = 1.15  # Same, for specialist confusion matrices
MIN_PROB_EPSILON = 1e-8  # Numerical stability for probability computations
# ================================================================================
# TRAINING PARAMETERS
# ================================================================================
BATCH_SIZE = 64  # Keep same for consistency
MAX_EPOCHS = 80  # Training epochs - DO NOT OVERRIDE
LEARNING_RATE = 0.0008  # Reduced from 0.001 to reduce oscillations
WEIGHT_DECAY = 0.0005  # L2 regularization to prevent overfitting

# Training Early Stopping
TRAINING_PATIENCE = 7  # Increased from 5 to allow more exploration
TRAINING_MIN_DELTA = 0.0005  # Reduced from 0.001 for more sensitive stopping  

# Replay Buffer Settings
REPLAY_RATIO = 0.3  # Static replay ratio
USE_SMART_REPLAY = True  # Set to True to enable smart replay buffer
REPLAY_DECAY_RATE = 0.95  # Temporal decay for older samples
REPLAY_IMPORTANCE_WEIGHT = 0.7  # Weight for gradient-based importance  
REPLAY_TEMPORAL_WEIGHT = 0.3  # Weight for temporal decay
MAX_REPLAY_SAMPLES_PER_CLASS = 1000  # Maximum samples per class in buffer

LABEL_SMOOTHING = 0.15 # Increased from 0.05 to 0.15 - reduces suppression of unseen neurons for better unlearning AUC

# Regularization Parameters
GATING_DROPOUT_RATE = 0.2  # Gating network dropout  

# CNN Architecture Parameters (flattened conv size is derived in create_model.py)
FC_LAYER_1_HIDDEN = 256  # Hidden layer size for classifier
FC_LAYER_DROPOUT = 0.5   # Dropout rate for FC layers

# Gating Network Architecture - Simplified lightweight routing network
# Note: Architecture is now hardcoded in create_model.py for simplicity
#   2 conv layers: in_channels→32→64 channels
#   2 FC layers: (flattened conv output)→128→num_shards

# Dataset-specific normalization (calculated from actual data)
def get_dataset_normalization(metadata_path=None):
    """
    Get dataset normalization values from metadata.
    
    Args:
        metadata_path: Path to metadata.json file. If None, uses default path from config.
        
    Returns:
        tuple: (mean, std) values for dataset normalization
    """
    import json
    import os
    
    if metadata_path is None:
        metadata_path = _os.path.join(get_project_dir(), "sisa_data", "metadata.json")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if 'normalization_mean' in metadata and 'normalization_std' in metadata:
            return metadata['normalization_mean'], metadata['normalization_std']
        else:
            raise KeyError("Normalization values not found in metadata")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise Exception(f"Could not load dataset normalization from {metadata_path}: {e}") from e

# Gating Network Training - same mechanism and settings as the shard models
GATING_MAX_EPOCHS = MAX_EPOCHS  # Same epoch budget as the shard models
GATING_LEARNING_RATE = LEARNING_RATE  # Same learning rate as the shard models
GATING_EARLY_STOPPING_PATIENCE = TRAINING_PATIENCE  # Same patience as the shard models
GATING_MIN_DELTA = TRAINING_MIN_DELTA  # Same improvement threshold as the shard models
GATING_BATCH_SIZE = 128  # Larger batch size for gating network (more stable gradients)

# ================================================================================
# UNLEARNING PARAMETERS
# ================================================================================
# Unlearning Early Stopping
UNLEARNING_PATIENCE = 7  # Increased from 5 to match training
UNLEARNING_MIN_DELTA = 0.0005  # Reduced to match training sensitivity
# Unlearning Training Settings - GDPR COMPLIANT
UNLEARNING_LEARNING_RATE = 0.0004  # Reduced from 0.0005 for better stability
UNLEARNING_REPLAY_RATIO = 0.3  # Keep at 0.3 to prevent catastrophic forgetting of remaining classes
# Must match LABEL_SMOOTHING, otherwise the before/after comparison is confounded
# by the objective change rather than by unlearning.
UNLEARNING_LABEL_SMOOTHING = LABEL_SMOOTHING

# Unlearning Success Threshold
UNLEARNING_SUCCESS_THRESHOLD = 0.45

# Max accuracy allowed on a deleted class (retrain-from-scratch predicts it ~never)
UNLEARNING_DELETED_CLASS_MAX_ACCURACY = 0.05

# Whether to retrain the gating network after unlearning.
#   False (default): keep the existing router. It emits shard indices only and stores
#                    no class information, and the shard models it routes into have
#                    already forgotten the class. Unlearning stays cheap.
#   True:            refit the router from the surviving slices, so no component of
#                    the ensemble has weights fitted on the deleted images. Stricter,
#                    and its cost is added to the reported unlearning time.
UNLEARNING_RETRAIN_GATING = False

# Membership inference: AUC ~0.5 means members are indistinguishable from non-members
MIA_AUC_TOLERANCE = 0.05  # |AUC - 0.5| must stay within this to pass
MIA_MAX_MEMBER_SAMPLES = 2000  # Deleted-class training samples retained as attack "members"
MIA_MIN_SAMPLES_PER_GROUP = 200  # Below this the AUC estimate is too noisy to quote


# ================================================================================
# SEARCH AND VISUALIZATION PARAMETERS
# ================================================================================
DEFAULT_SEARCH_SAMPLES = 16  

# Plotting Parameters
PLOT_TIGHT_LAYOUT_RECT = [0, 0, 1, 0.96] 

# ================================================================================
# FILE PATHS AND DIRECTORIES
# ================================================================================
PROJECT_NAME = "cifar10_sisa_pytorch"
MODEL_TYPE = "custom_cnn"
DATA_DIR = "data"
PROJECTS_DIR = "projects"
DATASET_NAME = "CIFAR-10"  # Selects the dataset AND is the display name for visualizations

# Everything is anchored to the directory holding this file, so generated data lands
# inside the repository and the scripts work from any working directory.
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))


def get_data_dir():
    """Absolute path to the downloaded-dataset cache."""
    return _os.path.join(REPO_ROOT, DATA_DIR)


def get_projects_root():
    """Absolute path to the directory holding all project run artifacts."""
    return _os.path.join(REPO_ROOT, PROJECTS_DIR)


def get_project_dir(project_name=None):
    """Absolute path to one project's artifacts (data, models, figures, metrics)."""
    return _os.path.join(get_projects_root(), project_name or PROJECT_NAME)

# ================================================================================
# DATASET REGISTRY
# ================================================================================
# Maps DATASET_NAME to its torchvision class. Requires .data, .targets and .classes.
SUPPORTED_DATASETS = {
    "CIFAR-10": "CIFAR10",
    "CIFAR-100": "CIFAR100",
}

# Used only when metadata predates the 'input_shape' field.
DEFAULT_INPUT_SHAPE = (3, 32, 32)

# ================================================================================
# REPRODUCIBILITY
# ================================================================================
RANDOM_SEED = 42  # Master seed for model init, dropout, augmentation and replay sampling
DETERMINISTIC_CUDNN = True  # Force deterministic cuDNN kernels (slower, but repeatable)


def set_seed(seed=None, deterministic=None):
    """Seed every RNG the pipeline draws from, so a run can be reproduced exactly.

    Covers Python's `random`, NumPy's legacy global RNG (used by the replay buffer's
    np.random.choice), and torch on both CPU and GPU (weight init, dropout masks and
    torchvision augmentation).

    Args:
        seed: Seed value; defaults to RANDOM_SEED.
        deterministic: Force deterministic cuDNN kernels; defaults to DETERMINISTIC_CUDNN.

    Returns:
        int: The seed that was applied.
    """
    import os
    import random

    import numpy as np
    import torch

    seed = RANDOM_SEED if seed is None else seed
    deterministic = DETERMINISTIC_CUDNN if deterministic is None else deterministic

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} (deterministic cuDNN: {deterministic})")
    return seed


def _load_metadata(metadata_path=None):
    """Load the SISA metadata.json produced by the data-processing step."""
    import json

    if metadata_path is None:
        metadata_path = _os.path.join(get_project_dir(), "sisa_data", "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_num_classes(metadata_path=None):
    """Return the dataset's class count, derived from metadata (never hard-coded)."""
    metadata = _load_metadata(metadata_path)
    class_names = metadata.get('class_names')
    if not class_names:
        raise KeyError("'class_names' missing from metadata; re-run data processing.")
    return len(class_names)


def get_input_shape(metadata_path=None):
    """Return the (channels, height, width) of a single sample, derived from metadata."""
    try:
        metadata = _load_metadata(metadata_path)
        input_shape = metadata.get('input_shape')
    except (FileNotFoundError, ValueError):
        input_shape = None

    if not input_shape:
        return DEFAULT_INPUT_SHAPE
    return tuple(input_shape)

# ================================================================================
# AUGMENTATION STRATEGIES
# ================================================================================

def get_augmentation_config(level='default', is_unlearning=False):
    """
    Get augmentation configuration based on balance level and training context.
    
    Args:
        level: 'minimal', 'light', or 'moderate' (no 'none' - use None instead)
        is_unlearning: If True, uses more conservative augmentation for unlearning
        
    Returns:
        dict: Augmentation configuration (NO random erasing)
    """
    # Base multiplier for unlearning (more conservative)
    multiplier = 0.6 if is_unlearning else 1.0
    
    configs = {
        'minimal': {
            'random_horizontal_flip': 0.3 * multiplier,
            'color_jitter_brightness': 0.05 * multiplier,
            'color_jitter_contrast': 0.05 * multiplier,
            'color_jitter_saturation': 0.02 * multiplier,
            'color_jitter_hue': 0.01 * multiplier,
            'reason': f'{"unlearning_" if is_unlearning else ""}well_balanced'
        },
        'light': {
            'random_horizontal_flip': 0.4 * multiplier,
            'color_jitter_brightness': 0.07 * multiplier,
            'color_jitter_contrast': 0.07 * multiplier,
            'color_jitter_saturation': 0.03 * multiplier,
            'color_jitter_hue': 0.015 * multiplier,
            'reason': f'{"unlearning_" if is_unlearning else ""}moderately_unbalanced'
        },
        'moderate': {
            'random_horizontal_flip': 0.6 * multiplier,
            'color_jitter_brightness': 0.1 * multiplier,
            'color_jitter_contrast': 0.1 * multiplier,
            'color_jitter_saturation': 0.05 * multiplier,
            'color_jitter_hue': 0.02 * multiplier,
            'reason': f'{"unlearning_" if is_unlearning else ""}significantly_unbalanced'
        }
    }
    
    if level not in configs:
        raise ValueError(f"Unknown augmentation level: {level}. Valid levels: {list(configs.keys())}")
    return configs[level]
