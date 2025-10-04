# ================================================================================
# SISA ARCHITECTURE PARAMETERS
# ================================================================================
NUM_SHARDS = 2  
NUM_SLICES_PER_SHARD = 5
# ================================================================================
# CONFIDENCE THRESHOLDS (GATING METHOD ONLY)
# ================================================================================
CONFIDENCE_THRESHOLD = 0.60  # Used during unlearning to guard against routing deleted classes
GATING_MARGIN_THRESHOLD = 0.08  # Require this margin between top shards to trust the primary pick
PRIMARY_SPECIALIST_WEIGHT = 0.65  # Portion of confidence retained for the routed shard when confident
PRIMARY_SPECIALIST_TEMPERATURE = 1.1  # Mildly flatten specialist logits to avoid overconfident peaks
SPECIALIST_EVAL_TEMPERATURE = 1.15  # Temperature used when producing specialist confusion matrices/ROC
SPECIALIST_ROC_TEMPERATURE = 1.15  # Consistent with evaluation temperature
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
REPLAY_RATIO = 0.3# Static replay ratio
USE_SMART_REPLAY = True  # Set to True to enable smart replay buffer
REPLAY_DECAY_RATE = 0.95  # Temporal decay for older samples
REPLAY_IMPORTANCE_WEIGHT = 0.7  # Weight for gradient-based importance  
REPLAY_TEMPORAL_WEIGHT = 0.3  # Weight for temporal decay
MAX_REPLAY_SAMPLES_PER_CLASS = 1000  # Maximum samples per class in buffer  



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
    import os
    
    if metadata_path is None:
        metadata_path = f"../{PROJECTS_DIR}/{PROJECT_NAME}/sisa_data/metadata.json"
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if 'normalization_mean' in metadata and 'normalization_std' in metadata:
            return metadata['normalization_mean'], metadata['normalization_std']
        else:
            raise KeyError("Normalization values not found in metadata")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise Exception(f"Could not load dataset normalization from {metadata_path}: {e}") from e

# Confidence Decision Parameters
CONFIDENCE_BOOST_THRESHOLD = 0.15  # Increased to reduce wrong overrides

# Gating Network Training - Enhanced for better routing accuracy
GATING_MAX_EPOCHS = 25  # Increased from 10 for better convergence
GATING_LEARNING_RATE = 0.0008  # Slightly reduced for stable training
GATING_EARLY_STOPPING_PATIENCE = 4  # Early stopping patience for gating network training
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
UNLEARNING_LABEL_SMOOTHING = 0.05  # GDPR COMPLIANCE: 0.05 enables exact unlearning (target: ~10% random guessing)
                                    # This matches "trained from scratch without deleted data" requirement
                                    # Result: Model shows NO evidence of training on deleted data (random performance)
                                    # For suppression unlearning (0% accuracy), set to 0.0

# Replay Buffer
USE_REPLAY_BUFFER = True  

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
DATA_DIR = "data"  
PROJECTS_DIR = "projects"
DATASET_NAME = "CIFAR-10"  # Display name for visualizations

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
