import os

# ================================================================================
# PROJECT ROOT (anchor for every path below — keeps all outputs inside the repo
# regardless of the CWD a script is launched from)
# ================================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ================================================================================
# REPRODUCIBILITY
# ================================================================================
SEED = 42  # Global seed for all RNGs (Python, NumPy, PyTorch). Required for the exactness proof.

# ================================================================================
# SISA ARCHITECTURE PARAMETERS
# ================================================================================
NUM_SHARDS = 2
NUM_SLICES_PER_SHARD = 5
# ================================================================================
# CONFIDENCE THRESHOLDS (self-routing deployment options -- see plan W5: not part
# of the exactness claim; CONFIDENCE_THRESHOLD is opt-in only, e.g. search.py's CLI)
# ================================================================================
CONFIDENCE_THRESHOLD = 0.60  # Opt-in rejection threshold for deployment-time use (search.py)
PRIMARY_SPECIALIST_TEMPERATURE = 1.1  # Mildly flatten specialist logits to avoid overconfident peaks
SPECIALIST_EVAL_TEMPERATURE = 1.15  # Temperature used when producing specialist confusion matrices/ROC
SPECIALIST_ROC_TEMPERATURE = 1.15  # Consistent with evaluation temperature
MIN_PROB_EPSILON = 1e-8  # Numerical stability for probability computations
# ================================================================================
# TRAINING PARAMETERS
# ================================================================================
BATCH_SIZE = 64  # Keep same for consistency
MAX_EPOCHS = 100  # Training epochs - DO NOT OVERRIDE
LEARNING_RATE = 0.0008  # Reduced from 0.001 to reduce oscillations
WEIGHT_DECAY = 0.0005  # L2 regularization to prevent overfitting

# Training Early Stopping
TRAINING_PATIENCE = 7  # Increased from 5 to allow more exploration
TRAINING_MIN_DELTA = 0.0005  # Reduced from 0.001 for more sensitive stopping  

# Replay Buffer Settings (W4: simple, seeded per-class dict replay -- deterministic,
# required for the exactness proof; the old gradient-importance "smart" buffer was
# unseeded and has been removed)
REPLAY_RATIO = 0.3  # Static fallback fraction (used when REPLAY_RATIO_MODE == 'static')
MAX_REPLAY_SAMPLES_PER_CLASS = 1000  # Cap per class; oldest excess is randomly (seeded) evicted
                                       # so buffer memory/build time don't grow unboundedly across
                                       # many slices or repeated unlearning rounds.

# W18: class-balanced replay. A fixed ratio cannot balance a class-incremental
# sequence -- with REPLAY_RATIO=0.3 and batch_size=64, shard 2's slice 5 gave the
# newest class ~36.7 of 64 slots while each of the 5 older classes got 4 (a 9:1
# imbalance), which is what produced the measured task-recency bias (horse 92.4%
# recall / 0.55 precision vs. deer 54.6%). 'balanced' derives the ratio from the
# class split instead: n_old / (n_old + n_new). This is a pure function of data
# composition -- no RNG -- so determinism and the exactness proof are unaffected.
REPLAY_RATIO_MODE = 'balanced'  # 'balanced' | 'static'
REPLAY_RATIO_MAX = 0.8  # Cap so the current slice always keeps a meaningful share of each batch



LABEL_SMOOTHING = 0.15 # Increased from 0.05 to 0.15 - reduces suppression of unseen neurons for better unlearning AUC

# Color Jitter Augmentation
COLOR_JITTER_BRIGHTNESS = 0.2  
COLOR_JITTER_CONTRAST = 0.2    
COLOR_JITTER_SATURATION = 0.2 
COLOR_JITTER_HUE = 0.1  

# Regularization Parameters
DROPOUT_RATE = 0.25  # Reduced dropout for better balance
DROPOUT_2D_RATE = 0.15  # 2D dropout for conv layers
GATING_DROPOUT_RATE = 0.2  # Gating network dropout  

# CNN Architecture Parameters
FC_LAYER_1_INPUT = 2048  # 128 * 4 * 4 (conv output flattened)
FC_LAYER_1_HIDDEN = 256  # Hidden layer size for classifier
FC_LAYER_DROPOUT = 0.5   # Dropout rate for FC layers

# W21: classifier head type.
#   'linear' -- original nn.Linear head.
#   'cosine' -- LUCIR-style scaled cosine classifier: logits = s * cos(f, w_c) with
#               L2-normalized features and class vectors. Two benefits at once:
#               (a) a calibrated routing score for free (max_c cos(f, w_c) over owned
#                   classes -- the 'ncm_cosine' score that led the gate-free probe at
#                   75.06% routing, but with learned class vectors), and
#               (b) removal of the weight-magnitude growth that drives task-recency
#                   bias, which is the same defect W18 attacks from the data side.
#               The class vectors are ordinary final-layer weight rows, so they are
#               already checkpointed and W6's _resize_model_head drops row c unchanged.
CLASSIFIER_TYPE = 'cosine'  # 'cosine' | 'linear'
COSINE_SCALE_INIT = 16.0    # Initial value of the learnable logit scale s
COSINE_SCALE_LEARNABLE = True

# W21: how a test sample is routed to a shard at inference time.
#   'gating'  -- learned gating network (W16). Must be retrained on every deletion.
#   'cosine'  -- parameter-free: score each shard by max_c cos(f, w_c) over its owned
#                classes. Nothing extra to unlearn (the affected shard's retrain
#                rebuilds its own routing signal; unaffected shards never saw the class).
#   'confidence' -- original masked-softmax self-routing fallback.
ROUTING_MODE = 'gating'

# Gating Network Architecture - Simplified lightweight routing network
# Note: Architecture is now hardcoded in create_model.py for simplicity
#   2 conv layers: 3→32→64 channels
#   2 FC layers: (64*8*8)→128→num_shards
# These config params are kept only for dropout rate reference

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

    if metadata_path is None:
        metadata_path = os.path.join(PROJECTS_DIR, PROJECT_NAME, "sisa_data", "metadata.json")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if 'normalization_mean' in metadata and 'normalization_std' in metadata:
            return metadata['normalization_mean'], metadata['normalization_std']
        else:
            raise KeyError("Normalization values not found in metadata")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise Exception(f"Could not load dataset normalization from {metadata_path}: {e}") from e


def get_num_classes(metadata_path=None):
    """
    Derive the number of classes from metadata.json's class_names list.
    Dataset-agnostic replacement for hard-coded `num_classes=10` defaults.
    """
    import json

    if metadata_path is None:
        metadata_path = os.path.join(PROJECTS_DIR, PROJECT_NAME, "sisa_data", "metadata.json")

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if 'class_names' in metadata:
            return len(metadata['class_names'])
        else:
            raise KeyError("class_names not found in metadata")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise Exception(f"Could not determine num_classes from {metadata_path}: {e}") from e

# Gating Network Training - Enhanced for better routing accuracy
GATING_MAX_EPOCHS = 25  # Increased from 10 for better convergence
GATING_LEARNING_RATE = 0.0008  # Slightly reduced for stable training
GATING_EARLY_STOPPING_PATIENCE = 4  # Early stopping patience for gating network training
GATING_BATCH_SIZE = 128  # Larger batch size for gating network (more stable gradients)

# W22: spatial grid kept before the gate's first FC layer. 8 (the original) makes
# fc1 a Linear(4096, 128) -- 524k of the gate's ~544k parameters, for a K-way
# decision. 1 = global average pooling -> Linear(64, 128), shrinking the gate ~19x.
GATING_POOL_SIZE = 1
# W22: shard labels are imbalanced whenever shards own different class counts. W17's
# semantic clustering produced a 4-vehicle / 6-animal split, i.e. 18000 vs 27000
# samples (40/60) -- and an unweighted cross-entropy drifts toward the majority shard.
# Measured effect: shard 1's classes leaked out at ~11% (airplane 20%) vs shard 2's
# ~2%, and routing sent 6321/3679 against a true 6000/4000 split.
GATING_CLASS_WEIGHTS = True

# ================================================================================
# UNLEARNING PARAMETERS
# ================================================================================
# Unlearning Early Stopping
UNLEARNING_PATIENCE = 7  # Increased from 5 to match training
UNLEARNING_MIN_DELTA = 0.0005  # Reduced to match training sensitivity
# Unlearning Training Settings - GDPR COMPLIANT
UNLEARNING_LEARNING_RATE = 0.0004  # Reduced from 0.0005 for better stability
UNLEARNING_REPLAY_RATIO = 0.3  # Keep at 0.3 to prevent catastrophic forgetting of remaining classes
UNLEARNING_LABEL_SMOOTHING = 0.05  # GDPR COMPLIANCE: 0.05 enables exact unlearning (target: ~10% random guessing)
                                    # This matches "trained from scratch without deleted data" requirement
                                    # Result: Model shows NO evidence of training on deleted data (random performance)
                                    # For suppression unlearning (0% accuracy), set to 0.0

# Unlearning Success Threshold
UNLEARNING_SUCCESS_THRESHOLD = 0.45 


# ================================================================================
# SEARCH AND VISUALIZATION PARAMETERS
# ================================================================================
DEFAULT_SEARCH_SAMPLES = 16  
VISUALIZATION_GRID_SIZE = 4 

# Plotting Parameters
PLOT_TIGHT_LAYOUT_RECT = [0, 0, 1, 0.96] 

# ================================================================================
# FILE PATHS AND DIRECTORIES
# ================================================================================
PROJECT_NAME = "cifar10_sisa_pytorch"
MODEL_TYPE = "custom_cnn"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROJECTS_DIR = os.path.join(PROJECT_ROOT, "projects")
# W12: DATASET is the machine-readable key data_processing/datasets.py resolves to
# a torchvision dataset class (see its DATASET_REGISTRY); DATASET_NAME is only for
# display in prints/plots. Switching datasets (e.g. to "cifar100") is a config-only
# change -- num_classes/class_names are always derived from sisa_data/metadata.json,
# never hardcoded.
DATASET = "cifar10"
DATASET_NAME = "CIFAR-10"  # Display name for visualizations
IN_CHANNELS = 3  # Number of input channels (3=RGB, 1=grayscale) for the model factory

# ================================================================================
# AUGMENTATION STRATEGIES
# ================================================================================

# W19: standard CIFAR-style geometric augmentation, applied regardless of class
# balance. Augmentation is a regularizer first and an imbalance remedy second --
# the previous design only ever reached for it when a shard was unbalanced, so on
# class-isolated CIFAR shards (perfectly balanced by construction) it never ran at
# all. RandomCrop with reflect padding + horizontal flip is the standard recipe and
# was entirely absent from every level below.
BASELINE_CROP_PADDING = 4      # RandomCrop(size, padding=4) -- the CIFAR standard
BASELINE_HORIZONTAL_FLIP = 0.5


def get_augmentation_config(level='baseline', is_unlearning=False):
    """
    Get augmentation configuration based on balance level and training context.

    Args:
        level: 'baseline' (always-on geometric augmentation), or 'minimal' / 'light' /
               'moderate' which add colour jitter on top for increasingly unbalanced shards
        is_unlearning: If True, uses more conservative augmentation for unlearning

    Returns:
        dict: Augmentation configuration (NO random erasing)
    """
    # Base multiplier for unlearning (more conservative)
    multiplier = 0.6 if is_unlearning else 1.0
    # Geometric augmentation is shared by every level; crop padding stays an integer.
    crop_padding = max(1, int(round(BASELINE_CROP_PADDING * multiplier)))

    configs = {
        'baseline': {
            'random_crop_padding': crop_padding,
            'random_horizontal_flip': BASELINE_HORIZONTAL_FLIP,
            'color_jitter_brightness': 0.0,
            'color_jitter_contrast': 0.0,
            'color_jitter_saturation': 0.0,
            'color_jitter_hue': 0.0,
            'reason': f'{"unlearning_" if is_unlearning else ""}baseline_geometric'
        },
        'minimal': {
            'random_crop_padding': crop_padding,
            'random_horizontal_flip': 0.5,
            'color_jitter_brightness': 0.05 * multiplier,
            'color_jitter_contrast': 0.05 * multiplier,
            'color_jitter_saturation': 0.02 * multiplier,
            'color_jitter_hue': 0.01 * multiplier,
            'reason': f'{"unlearning_" if is_unlearning else ""}well_balanced'
        },
        'light': {
            'random_crop_padding': crop_padding,
            'random_horizontal_flip': 0.5,
            'color_jitter_brightness': 0.07 * multiplier,
            'color_jitter_contrast': 0.07 * multiplier,
            'color_jitter_saturation': 0.03 * multiplier,
            'color_jitter_hue': 0.015 * multiplier,
            'reason': f'{"unlearning_" if is_unlearning else ""}moderately_unbalanced'
        },
        'moderate': {
            'random_crop_padding': crop_padding,
            'random_horizontal_flip': 0.5,
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
