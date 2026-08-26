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
# W26: both accept an int (manual) or 'auto' (derived from the dataset).
#
#   NUM_SLICES_PER_SHARD = 'auto'  ->  S_k = |C_k|, one class per slice, PER SHARD.
#     This is provably optimal for class unlearning: deletion always restarts at a
#     class's FIRST slice, so splitting a class across more slices leaves the retrain
#     point unmoved while adding incremental steps (and forgetting); making slices
#     coarser makes classes share a slice, so deleting one drags the other along.
#     Measured cost of the old global S=5 against 4- and 6-class shards: bird and cat
#     both start in slice 1, so deleting either retrains 100% of shard 2 -- 13% of the
#     available speedup lost to an arbitrary constant.
#
#   NUM_SHARDS = 'auto'  ->  K = ceil(NUM_CLASSES / MAX_SLICES_PER_SHARD).
#     Each constraint on K is a lower bound and smaller K makes routing easier, so the
#     smallest feasible K is chosen. Manual K remains the default because K is the
#     variable the efficiency-vs-accuracy study sweeps -- deriving it would hide the
#     very tradeoff being measured. 'auto' is for applying the method to a new dataset.
NUM_SHARDS = 2
NUM_SLICES_PER_SHARD = 'auto'

# W26: how many classes a shard may hold, i.e. how many sequential slices it trains
# through. NOT a free guess -- it is set by what a training batch can actually keep
# balanced. With one class per slice, the j-th slice must split the batch across j
# classes, so each gets BATCH_SIZE/j samples; below MIN_SAMPLES_PER_CLASS_IN_BATCH the
# gradient and BatchNorm statistics for a class get too thin to be reliable:
#
#     MAX_SLICES_PER_SHARD = BATCH_SIZE // MIN_SAMPLES_PER_CLASS_IN_BATCH
#
# Raising it means fewer, larger shards -- which also raises replay overhead, since
# training cost scales as (S+1)/2 versus a single joint model. Lowering it means more
# shards: cheaper training AND faster unlearning, with harder routing as the only cost.
MIN_SAMPLES_PER_CLASS_IN_BATCH = 8
MAX_SLICES_PER_SHARD = 8  # = BATCH_SIZE // MIN_SAMPLES_PER_CLASS_IN_BATCH (64 // 8)
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
MAX_EPOCHS = 100  # Upper bound only -- training is patience-limited in practice.
                  # Measured: 0 of 10 slices reached it; all stopped between 8 and 30.
LEARNING_RATE = 0.0008  # Reduced from 0.001 to reduce oscillations
WEIGHT_DECAY = 0.0005  # Decoupled weight decay (see OPTIMIZER below)

# W23: AdamW, not Adam. `Adam(..., weight_decay=)` applies L2 *inside* the adaptive
# update, so the effective decay is scaled by each parameter's running gradient
# magnitude and does not act as true weight decay. AdamW decouples it, which is what
# WEIGHT_DECAY above was always intended to mean (Loshchilov & Hutter, ICLR 2019).
OPTIMIZER = 'adamw'  # 'adamw' | 'adam'

# W23: learning-rate schedule. The previous code asserted "No lr_scheduler needed --
# Adam handles adaptive learning rates", which conflates two things: Adam adapts the
# *per-parameter* step scaling, it does not decay the *global* learning rate. The
# measured symptom was val-accuracy oscillation that never settles (shard 2 slice 3
# swung over 0.065 in its final 8 epochs). ReduceLROnPlateau is used rather than a
# cosine schedule because early stopping makes the run length unknown in advance, so
# there is no sensible T_max to anneal over; plateau detection adapts to whatever
# length the run turns out to be. Its patience must stay below TRAINING_PATIENCE so
# the LR actually gets a chance to drop before early stopping fires.
LR_SCHEDULER = 'plateau'  # 'plateau' | 'none'
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 1e-6

# W23: do not let early stopping fire until the LR has actually been annealed.
# Measured on the 74.68% run, every slice stopped at its FIRST plateau while the LR
# was still at its initial 8e-4 -- 8 to 30 epochs, none near the 100-epoch budget.
# A plateau at the starting LR says nothing about whether the model would keep
# improving at a lower one, which is the whole premise of LR annealing. Requiring N
# reductions first means each slice trains through LR 8e-4 -> 4e-4 -> 2e-4 -> 1e-4
# before stopping is even considered. The early-stopping patience still governs when
# it stops after that, so a genuinely converged slice does not run to the cap.
MIN_LR_REDUCTIONS_BEFORE_STOP = 3

# Training Early Stopping
# W23: monitor validation ACCURACY, not validation loss. With LABEL_SMOOTHING > 0,
# smoothed cross-entropy penalises confident-correct predictions, so val_loss bottoms
# out and rises while accuracy is still climbing. Measured on the 74.68% run: in 5 of
# 10 slices the peak val accuracy came *after* the restored best-val_loss epoch --
# shard 2 slice 3 discarded +2.6 points, and shard 2 slice 1 restored epoch 1 of 8.
# Select the checkpoint by the metric actually being reported.
EARLY_STOPPING_MONITOR = 'val_accuracy'  # 'val_accuracy' | 'val_loss'
TRAINING_PATIENCE = 10  # Accuracy is noisier than loss, so allow more exploration
TRAINING_MIN_DELTA = 0.001  # ACCURACY units (0.1%) when monitoring val_accuracy
TRAINING_MIN_DELTA_LOSS = 0.0005  # LOSS units; degenerate-validation fallback only
# W23: floor on epochs before early stopping may fire. Early slices validate on very
# few samples (shard 2 slice 1: 1000 samples, val-accuracy range 0.185 across epochs),
# so an unguarded monitor stops on noise and undertrains the shard's foundation.
TRAINING_MIN_EPOCHS = 10

# Replay Buffer Settings (W4: simple, seeded per-class dict replay -- deterministic,
# required for the exactness proof; the old gradient-importance "smart" buffer was
# unseeded and has been removed)
REPLAY_RATIO = 0.3  # Static fallback fraction (used when REPLAY_RATIO_MODE == 'static')
# W23: raised from 1000. Measured forgetting is a clean dose-response in how many
# slices a class spends replay-only after its last real appearance:
#   bird (4 slices) -17.2 pts | cat (3) -33.5 from peak | deer (2) -14.1 |
#   dog (1) -0.1 | frog (0) +7.8
# W18 equalised each class's *share* of the batch, but at a 1000 cap an old class was
# still represented by only 1000 of its 4500 images -- the same pictures for four
# consecutive slices -- so it overfits them. This is a diversity limit, not a share
# limit. 5000 exceeds the largest per-class count in a shard (4500), so replay is
# effectively complete for CIFAR-10 while the cap still bounds memory for larger
# datasets. Cost: shard 2's buffer grows to roughly 330 MB of float32 in RAM.
MAX_REPLAY_SAMPLES_PER_CLASS = 5000

# W18: class-balanced replay. A fixed ratio cannot balance a class-incremental
# sequence -- with REPLAY_RATIO=0.3 and batch_size=64, shard 2's slice 5 gave the
# newest class ~36.7 of 64 slots while each of the 5 older classes got 4 (a 9:1
# imbalance), which is what produced the measured task-recency bias (horse 92.4%
# recall / 0.55 precision vs. deer 54.6%). 'balanced' derives the ratio from the
# class split instead: n_old / (n_old + n_new). This is a pure function of data
# composition -- no RNG -- so determinism and the exactness proof are unaffected.
REPLAY_RATIO_MODE = 'balanced'  # 'balanced' | 'static'

# W26: raised from 0.8. Equal per-class batch share at a shard's j-th slice needs
# replay_ratio = (j-1)/j, so a 0.8 cap can only balance 5 slices -- at 6 it binds and
# per-class imbalance starts creeping back (1.15x at 6 classes, 1.62x at 8, 4.4x at 20),
# which is the same defect that produced the original task-recency bias. 0.875 supports
# 8 classes per shard, matching MAX_SLICES_PER_SHARD below.
#
# The cap must satisfy:  REPLAY_RATIO_MAX >= (MAX_SLICES_PER_SHARD - 1) / MAX_SLICES_PER_SHARD
REPLAY_RATIO_MAX = 0.875



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

# CNN Architecture Parameters (SISAConvNet path only -- the ResNet path derives its
# widths from RESNET_BASE_WIDTH and must not read these; see W28.)
FC_LAYER_1_INPUT = 2048  # 128 * 4 * 4 (conv output flattened)
FC_LAYER_1_HIDDEN = 256  # Hidden layer size for classifier
FC_LAYER_DROPOUT = 0.5   # Dropout rate for FC layers

# ================================================================================
# W28: BACKBONE SELECTION (architecture ablation)
# ================================================================================
# 'convnet' is the default so the measured 83.26% baseline stays reproducible; the
# ablation is run by flipping this to 'resnet' with everything else held fixed.
#
# Why a residual backbone is worth testing -- parameters per training sample WITHIN a
# shard, which is the quantity that matters, since class-isolated sharding gives each
# specialist only |C_k| * n / N_C samples:
#
#     model                    params    CIFAR-10   CIFAR-100   Tiny-ImageNet
#     SISAConvNet (current)   621,313          31         219             173
#     ResNet-20               272,474          13          96               -
#     ResNet-18 (ImageNet)  11,173,962         497       3,547           3,104
#
# ResNet-20 has FEWER parameters than the current CNN while adding residual
# connections: the current 0.62M sits mostly in Linear(2048, 256), which global
# average pooling removes entirely.
MODEL_ARCH = 'resnet'  # 'convnet' | 'resnet'

# W28: depth follows from input resolution rather than being hardcoded, consistent
# with how NUM_SHARDS, NUM_SLICES_PER_SHARD and MAX_SLICES_PER_SHARD are derived.
# Stages are added until the feature map reaches RESNET_TARGET_SPATIAL before global
# average pooling:
#
#     n_stages = ceil(log2(input_size / RESNET_TARGET_SPATIAL)) + 1
#     widths   = [RESNET_BASE_WIDTH * 2**i for i in range(n_stages)]
#     depth    = 2 * RESNET_BLOCKS_PER_STAGE * n_stages + 2
#
#     32x32 -> 3 stages [16, 32, 64]       -> depth 20  (= ResNet-20 exactly)
#     64x64 -> 4 stages [16, 32, 64, 128]  -> depth 26
#
# So at 32x32 this is not a departure from He et al.'s ResNet-20 -- it is that
# architecture with its resolution assumption made explicit.
RESNET_BLOCKS_PER_STAGE = 3
RESNET_BASE_WIDTH = 16
RESNET_TARGET_SPATIAL = 8

# W28: CIFAR ResNets conventionally use no dropout, and FC_LAYER_DROPOUT=0.5 would be
# actively destructive here: it is applied to a 2048-dim flattened feature in the
# ConvNet, but a residual net's pre-classifier feature is only RESNET_BASE_WIDTH *
# 2**(n_stages-1) wide (64 at 32x32) after global average pooling. Dropping half of 64
# features is the most likely way a first ResNet run underperforms for reasons that
# have nothing to do with the architecture.
RESNET_DROPOUT = 0.0

# W29: the gate uses the same residual family, just shallower -- one architecture with
# a depth parameter rather than two unrelated networks. Measured at 1 block/stage,
# base 16: 77,522 params at 32x32 (28% of a ResNet-20 specialist), versus the current
# GatingNetwork's 544,258, which is 200% of a specialist -- a router twice the size of
# the model it routes to. Note GATING_INPUT_SIZE/GATING_POOL_SIZE are GatingNetwork-only
# knobs; a ResNet gate derives its own stages from resolution.
GATING_BLOCKS_PER_STAGE = 1
GATING_BASE_WIDTH = 16

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
#   'ensemble' -- W24: fitted log-linear blend of the gate and cosine-to-class-vector.
#                Measured 94.63% -> 95.45% routing, 83.26% -> 84.07% combined. The two
#                blend weights are fitted on validation after training (they are not
#                constants: the optimum moves with the gate's strength), and the blend
#                is only adopted if it beats the gate alone on validation.
#   'gating'  -- learned gating network (W16). Must be retrained on every deletion.
#   'cosine'  -- parameter-free: score each shard by max_c cos(f, w_c) over its owned
#                classes. Nothing extra to unlearn (the affected shard's retrain
#                rebuilds its own routing signal; unaffected shards never saw the class).
#   'confidence' -- original masked-softmax self-routing fallback.
ROUTING_MODE = 'ensemble'

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
# W25: the gate trains under the same regime as the specialists. It previously had a
# 25-epoch cap and patience 4 with no LR schedule, so it stopped at its first plateau
# at the initial LR -- measured stopping at epoch 10/25 with its best at epoch 6 while
# training accuracy was still climbing. Same defect W23 fixed for the specialists, and
# it matters most here because routing is now the system bottleneck (94.63% routing
# against an 87.53% ceiling, so each routing point is worth ~0.87 system points).
# The gate reuses LR_SCHEDULER_* and MIN_LR_REDUCTIONS_BEFORE_STOP from above.
GATING_MAX_EPOCHS = 100  # Upper bound only; patience still governs in practice
GATING_LEARNING_RATE = 0.0008  # Slightly reduced for stable training
GATING_EARLY_STOPPING_PATIENCE = 10  # Matches TRAINING_PATIENCE
GATING_MIN_EPOCHS = 10  # Matches TRAINING_MIN_EPOCHS
GATING_BATCH_SIZE = 128  # Larger batch size for gating network (more stable gradients)

# W22: spatial grid kept before the gate's first FC layer. 8 makes fc1 a
# Linear(4096, 128) -- 524k of the gate's ~544k parameters, for a K-way decision.
# 1 = global average pooling -> Linear(64, 128), shrinking the gate ~19x.
#
# MEASURED: the 19x shrink is NOT free. pool_size=1 dropped gate validation accuracy
# 95.31% -> 92.76% and test routing 94.41% -> 90.79%. At the post-W18 oracle ceiling
# of 81.68%, each point of routing accuracy is worth ~0.82 points of system accuracy,
# so the shrink cost ~2.4 points of combined accuracy to save 516k parameters in a
# model that runs once per sample. Not a good trade -- reverted to 8.
GATING_POOL_SIZE = 4

# W24: resolution the gate routes at. 0 = full input (32x32 for CIFAR).
# The gate's compute is conv-bound -- conv2 alone is 4.7M of its 6.1M MACs -- and conv
# cost scales with spatial area, so routing at 16x16 cuts its MACs roughly 4x. That is
# the lever that actually moves the unlearning-round cost, unlike GATING_POOL_SIZE,
# which strips 96% of the parameters but only ~8% of the compute.
#
# DEFAULT OFF, deliberately. This is untested for accuracy, and the last "obviously
# free" gate optimisation (GATING_POOL_SIZE=1) turned out to cost 2.4 points of system
# accuracy. Set to 16 and retrain the gate alone (~20s once specialists exist) to A/B
# it. If enabling, also set GATING_POOL_SIZE = 4: at 16x16 input the conv stack already
# ends at 4x4, so leaving it at 8 makes the adaptive pool upsample and wastes fc1.
GATING_INPUT_SIZE = 16

# W22: shard labels are imbalanced whenever shards own different class counts -- W17's
# semantic clustering produced a 4-vehicle / 6-animal split, i.e. 18000 vs 27000 (40/60).
#
# MEASURED: weighting BACKFIRED. It did not centre the routing; it flipped the bias,
# 3679/6321 -> 4521/5479 against a true 4000/6000. Per-shard in-shard retention went
# shard 1 89% -> 95% and shard 2 98% -> 88%. Because shard 2 holds 6000 of the 10000
# test samples, the unweighted gate's apparent "bias" toward it was in fact
# aggregate-optimal: making routing per-shard *fair* made it overall *worse*
# (94.41% -> 90.79%). Left off; revisit only with a shard-size-aware objective.
GATING_CLASS_WEIGHTS = False

# W22: the gate's own augmentation, kept separate from the specialists'. Routing is a
# coarse whole-image decision, so a random crop can remove the very content that
# distinguishes the shards -- unlike the specialists, where cropping is a strong
# regularizer. Flip is applied per sample here (the original code applied a torchvision
# transform to a batched tensor, giving all 128 samples one shared decision).
GATING_CROP_PADDING = 0
GATING_FLIP_PROB = 0.5

# ================================================================================
# UNLEARNING PARAMETERS
# ================================================================================
# Unlearning Early Stopping -- must mirror the training values (W23). Retraining under
# a different stopping rule than original training would invalidate the W8
# scratch-reference comparison.
UNLEARNING_PATIENCE = 10
UNLEARNING_MIN_DELTA = 0.001  # ACCURACY units, matching TRAINING_MIN_DELTA
UNLEARNING_MIN_EPOCHS = 10
# Unlearning Training Settings
#
# W27: the retrain MUST use the same optimisation hyperparameters as ordinary training.
# Exactness is defined as
#     Unlearn( Train(D u Dc), Dc )  ==  Train(D \ Dc)
# so the retrain on the left has to run the SAME `Train` as the from-scratch reference
# on the right. These previously differed (LR 0.0004 vs 0.0008, label smoothing 0.05 vs
# 0.15) while experiments/scratch_reference.py built its reference with the *training*
# values -- so the two sides ran different algorithms and could never agree on weights,
# which would have shown up as a failed W8 comparison misread as GPU nondeterminism.
# Derived from the training values so they cannot drift apart again.
UNLEARNING_LEARNING_RATE = LEARNING_RATE
UNLEARNING_REPLAY_RATIO = 0.3  # Keep at 0.3 to prevent catastrophic forgetting of remaining classes
# W27: must equal LABEL_SMOOTHING for the same reason as the learning rate above.
# The previous comment claimed 0.05 "enables exact unlearning" -- that reasoning was
# backwards: a smoothing value that DIFFERS from training is precisely what makes the
# retrained model unequal to a from-scratch one. Exactness comes from retraining on
# class-clean data with the same algorithm, not from a particular smoothing value.
UNLEARNING_LABEL_SMOOTHING = LABEL_SMOOTHING

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

# W30: Cutout (DeVries & Taylor 2017) -- the standard companion regularizer to
# crop+flip for CIFAR ResNets, and the one this pipeline was missing. Measured on the
# ResNet-20 run: shard 2 slice 6 ended with a 9.6-point train/val gap (train 0.9766 vs
# val 0.8807), and after the val-loss minimum at epoch 14 train loss kept falling
# (-0.059) while val loss rose (+0.014) -- the textbook memorisation signature.
#
# Expressed as a FRACTION of the image side rather than a pixel count, so it stays
# resolution-agnostic like the partition and backbone: 0.5 gives the standard 16px
# square on 32x32 CIFAR, and the right 32px square on 64x64 Tiny-ImageNet.
BASELINE_CUTOUT_FRACTION = 0.5


def get_input_size(metadata_path=None) -> int:
    """Input image side length, read from metadata.json (W28).

    The residual backbone derives its stage count from resolution, so this has to come
    from the data rather than a constant -- mirrors get_num_classes(). Falls back to 32
    for projects whose metadata predates this field.
    """
    import json

    if metadata_path is None:
        metadata_path = os.path.join(PROJECTS_DIR, PROJECT_NAME, "sisa_data", "metadata.json")

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return int(metadata['input_size'])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 32


def resnet_stage_widths(input_size: int, base_width: int = None, target_spatial: int = None):
    """Channel widths per stage, derived from input resolution (W28).

    Returns e.g. [16, 32, 64] at 32x32 and [16, 32, 64, 128] at 64x64, so the feature
    map entering global average pooling is always about `target_spatial` on a side.
    """
    import math

    base_width = RESNET_BASE_WIDTH if base_width is None else base_width
    target_spatial = RESNET_TARGET_SPATIAL if target_spatial is None else target_spatial
    n_downsamples = max(0, math.ceil(math.log2(max(1, input_size) / target_spatial)))
    return [base_width * (2 ** i) for i in range(n_downsamples + 1)]


def resolve_num_shards(num_classes: int) -> int:
    """Resolve NUM_SHARDS, deriving it from the dataset when set to 'auto' (W26).

    K = ceil(num_classes / MAX_SLICES_PER_SHARD), clamped to [2, num_classes // 2].
    The upper clamp keeps at least 2 classes per shard: below that, slicing does
    nothing and the architecture degenerates into ARCANE's one-model-per-class.
    """
    configured = NUM_SHARDS
    if isinstance(configured, int):
        return configured
    if str(configured).lower() != 'auto':
        raise ValueError(f"NUM_SHARDS must be an int or 'auto', got {configured!r}")

    import math
    k_max = max(2, num_classes // 2)
    k = max(2, math.ceil(num_classes / MAX_SLICES_PER_SHARD))
    if k > k_max:
        raise ValueError(
            f"Derived NUM_SHARDS={k} exceeds the maximum {k_max} for {num_classes} classes. "
            f"Raise MAX_SLICES_PER_SHARD (currently {MAX_SLICES_PER_SHARD})."
        )
    return k


def resolve_slices_for_shard(shard_class_count: int) -> int:
    """Resolve this shard's slice count, deriving it when NUM_SLICES_PER_SHARD is
    'auto' (W26): one class per slice, so S_k = |C_k|."""
    configured = NUM_SLICES_PER_SHARD
    if isinstance(configured, int):
        return configured
    if str(configured).lower() != 'auto':
        raise ValueError(f"NUM_SLICES_PER_SHARD must be an int or 'auto', got {configured!r}")
    return max(1, shard_class_count)


def validate_partition(shard_class_counts) -> list:
    """Check a partition against the replay budget. Returns a list of warnings.

    The two settings are coupled and silently break each other if they drift:
    equal per-class batch share at a shard's j-th slice requires a replay ratio of
    (j-1)/j, so REPLAY_RATIO_MAX bounds how many classes a shard can hold before
    per-class imbalance -- the original cause of task-recency bias -- returns.
    """
    warnings = []
    largest = max(shard_class_counts) if shard_class_counts else 0

    required_ratio = (largest - 1) / largest if largest > 1 else 0.0
    if required_ratio > REPLAY_RATIO_MAX + 1e-9:
        warnings.append(
            f"Largest shard holds {largest} classes, needing replay ratio "
            f"{required_ratio:.3f} for equal per-class batch share, but REPLAY_RATIO_MAX "
            f"is {REPLAY_RATIO_MAX}. The newest class will be over-represented by "
            f"{(BATCH_SIZE * (1 - REPLAY_RATIO_MAX)) / (BATCH_SIZE / largest):.2f}x."
        )
    if largest > MAX_SLICES_PER_SHARD:
        warnings.append(
            f"Largest shard holds {largest} classes, above MAX_SLICES_PER_SHARD="
            f"{MAX_SLICES_PER_SHARD}; each class gets ~{BATCH_SIZE // largest} samples per "
            f"batch, below MIN_SAMPLES_PER_CLASS_IN_BATCH={MIN_SAMPLES_PER_CLASS_IN_BATCH}."
        )
    if any(c < 2 for c in shard_class_counts):
        warnings.append(
            f"A shard holds fewer than 2 classes {list(shard_class_counts)}; slicing cannot "
            f"help there, since deleting its only class retrains the whole shard."
        )
    return warnings


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
    # Milder cutout during unlearning retrains, matching the existing `multiplier`
    # convention for the other augmentation strengths.
    cutout_fraction = BASELINE_CUTOUT_FRACTION * multiplier

    configs = {
        'baseline': {
            'random_crop_padding': crop_padding,
            'random_horizontal_flip': BASELINE_HORIZONTAL_FLIP,
            'cutout_fraction': cutout_fraction,
            'color_jitter_brightness': 0.0,
            'color_jitter_contrast': 0.0,
            'color_jitter_saturation': 0.0,
            'color_jitter_hue': 0.0,
            'reason': f'{"unlearning_" if is_unlearning else ""}baseline_geometric'
        },
        'minimal': {
            'random_crop_padding': crop_padding,
            'random_horizontal_flip': 0.5,
            'cutout_fraction': cutout_fraction,
            'color_jitter_brightness': 0.05 * multiplier,
            'color_jitter_contrast': 0.05 * multiplier,
            'color_jitter_saturation': 0.02 * multiplier,
            'color_jitter_hue': 0.01 * multiplier,
            'reason': f'{"unlearning_" if is_unlearning else ""}well_balanced'
        },
        'light': {
            'random_crop_padding': crop_padding,
            'random_horizontal_flip': 0.5,
            'cutout_fraction': cutout_fraction,
            'color_jitter_brightness': 0.07 * multiplier,
            'color_jitter_contrast': 0.07 * multiplier,
            'color_jitter_saturation': 0.03 * multiplier,
            'color_jitter_hue': 0.015 * multiplier,
            'reason': f'{"unlearning_" if is_unlearning else ""}moderately_unbalanced'
        },
        'moderate': {
            'random_crop_padding': crop_padding,
            'random_horizontal_flip': 0.5,
            'cutout_fraction': cutout_fraction,
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
