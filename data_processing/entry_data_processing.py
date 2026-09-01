import os
import numpy as np
import torch
import torchvision
import argparse
import time
import json
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime

# Setup path for imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add current directory to path for local imports
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from sharding import create_shards_with_indices
from slicing import create_slices
from plots import create_data_processing_visualizations, visualize_sample_images
from datasets import get_dataset_class

sys.path.append('..')
import config
from utils.seeding import set_seed
from utils.run_logging import setup_run_logging
from utils.data_io import save_images, encoded_dtype

set_seed(config.SEED)

# W12: which dataset to load is a config choice, not a hardcoded torchvision call.
DatasetClass = get_dataset_class(config.DATASET)

# Define command-line arguments with config.py defaults
parser = argparse.ArgumentParser(description='SISA Framework Sequential Data Processing')
# W26: both may be 'auto' in config; CLI overrides stay integers.
parser.add_argument('--num-shards', type=int, default=None, help="Number of shards (default: config, may be 'auto')")
parser.add_argument('--num-slices', type=int, default=None, help="Slices per shard (default: config, may be 'auto' => one class per slice)")
parser.add_argument('--force', action='store_true',
                    help="Rewrite the .npy data even if what is on disk already matches this config "
                         "(see the idempotency guard in section 3 -- use this after editing the "
                         "sharding/slicing code itself, which the config fingerprint cannot see).")

args = parser.parse_args()

project_name = config.PROJECT_NAME
# W26: NUM_SHARDS may be 'auto', which needs the class count -- resolved after load_dataset().
num_shards = args.num_shards
num_slices = args.num_slices if args.num_slices is not None else config.NUM_SLICES_PER_SHARD
base_dir = os.path.join(config.PROJECTS_DIR, project_name)
data_info_dir = os.path.join(base_dir, "data_info")
os.makedirs(base_dir, exist_ok=True)
os.makedirs(data_info_dir, exist_ok=True)

# Project-scoped, timestamped logging (W10) -- now that base_dir is known.
_restore_logging, _log_path = setup_run_logging(os.path.join(base_dir, "logs"), "data_processing")

print("=" * 80)
print("SISA FRAMEWORK - DATA PROCESSING")
print("=" * 80)
print(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print(f"   - Number of Shards: {num_shards if num_shards is not None else config.NUM_SHARDS}")
print(f"   - Slices per Shard: {num_slices}")

def load_dataset():
    """Load config.DATASET using the canonical split: the official train set
    (carved into train/validation) and the official test set, untouched."""
    dataset_train = DatasetClass(root=config.DATA_DIR, train=True, download=True)
    dataset_test = DatasetClass(root=config.DATA_DIR, train=False, download=True)
    class_names = dataset_train.classes

    x_train_full = dataset_train.data
    y_train_full = np.array(dataset_train.targets)
    x_test = dataset_test.data
    y_test = np.array(dataset_test.targets)

    # Convert to PyTorch format (C, H, W) and normalize to [0, 1]
    x_train_full = np.transpose(x_train_full, (0, 3, 1, 2)).astype(np.float32) / 255.0
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32) / 255.0

    print(f"Loaded {config.DATASET_NAME} with preprocessing:")
    print(f"   - Official train shape: {x_train_full.shape}")
    print(f"   - Official test shape: {x_test.shape}")
    print(f"   - Data range: [{x_train_full.min():.2f}, {x_train_full.max():.2f}]")

    # Carve validation out of the official 50k train set ONLY (45k/5k stratified).
    # The official 10k test set is never resplit or mixed with train.
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=config.SEED, stratify=y_train_full
    )

    total_samples = len(x_train_full) + len(x_test)
    print(f"Data split completed (canonical {config.DATASET_NAME} split):")
    print(f"   - Train: {len(x_train):,} samples ({len(x_train)/total_samples:.1%})")
    print(f"   - Validation: {len(x_val):,} samples ({len(x_val)/total_samples:.1%})")
    print(f"   - Test: {len(x_test):,} samples ({len(x_test)/total_samples:.1%}) [official {config.DATASET_NAME} test set, untouched]")

    split_info = {
        'total_samples': total_samples,
        'train_samples': len(x_train),
        'val_samples': len(x_val),
        'test_samples': len(x_test),
        'random_state': config.SEED,
        'split_strategy': f'canonical_{config.DATASET}_official_train_test',
    }

    return x_train, y_train, split_info, x_test, y_test, x_val, y_val, class_names

def create_data_visualizations():
    """Generates comprehensive visualizations for SISA data processing."""
    
    # Prepare shards data in the format expected by visualization functions
    shards_data = []
    for shard_idx in range(num_shards):
        slices_data = []
        for slice_idx in range(len(shard_slices_list[shard_idx])):
                slices_data.append({
                    'x': shard_slices_list[shard_idx][slice_idx],
                    'y': y_slices_list[shard_idx][slice_idx],
                    'indices': shard_indices_list[shard_idx][slice_idx]
                })
        
        shards_data.append({
            'slices': slices_data,
            'shard_idx': shard_idx
        })
    
    # Create all visualizations
    create_data_processing_visualizations(
        shards_data, 
        (x_val, y_val), 
        (x_test, y_test), 
        class_names, 
        data_info_dir
    )
    
    # Create sample images visualization for first shard, first slice
    if len(shards_data) > 0 and len(shards_data[0]['slices']) > 0:
        visualize_sample_images(
            shards_data[0]['slices'][0]['x'], 
            shards_data[0]['slices'][0]['y'], 
            class_names, 
            data_info_dir, 
            "Shard_1_Slice_1_"
        )


# --- Main Execution ---

# Load data
x_train, y_train, split_info, x_test, y_test, x_val, y_val, class_names = load_dataset()

# W26: resolve 'auto' shard count now that the class list is known.
if num_shards is None:
    num_shards = config.resolve_num_shards(len(class_names))
    print(f"   - Resolved NUM_SHARDS={num_shards} from {len(class_names)} classes "
          f"(MAX_SLICES_PER_SHARD={config.MAX_SLICES_PER_SHARD})")

# --- NEW: Dynamically calculate normalization statistics from the training data ---
print("\nCalculating normalization statistics from the training set...")
x_train_tensor = torch.from_numpy(x_train)
# Calculate mean and std per channel (axis 0 is samples, axis 1 is channels)
train_mean = x_train_tensor.mean(axis=(0, 2, 3)).tolist()
train_std = x_train_tensor.std(axis=(0, 2, 3)).tolist()
print(f"   - Calculated Mean: { [f'{m:.4f}' for m in train_mean] }")
print(f"   - Calculated Std Dev: { [f'{s:.4f}' for s in train_std] }")
# --- END OF NEW ---

# 1. Create Shards with Class Isolation
shards, y_shards, index_shards, shard_class_dists, sharding_split_info = create_shards_with_indices(
    x_train, y_train, num_shards, class_names
)
split_info.update(sharding_split_info)

# --- Terminal Output for Shards ---
print("\n" + "="*25 + " Shard Analysis " + "="*25)
for i, dist in enumerate(shard_class_dists):
    total_samples = sum(dist.values())
    print(f"\nShard {i+1} (Total Samples: {total_samples:,})")
    print("-" * 20)
    for class_name, count in sorted(dist.items()):
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"   - {class_name:<12}: {count:>5,} samples ({percentage:.1f}%)")
print("="*68)

# 2. Create Slices Sequentially for each shard
shard_slices_list = []
y_slices_list = []
shard_indices_list = []
slice_class_dists_list = [] 

# W26: slice count is resolved PER SHARD. With NUM_SLICES_PER_SHARD='auto' this is
# S_k = |C_k| -- one class per slice -- which is the optimum for class unlearning:
# deletion restarts at a class's first slice, so a class sharing a slice with another
# drags it along (bird and cat both starting in slice 1 meant deleting either retrained
# all of shard 2). A global constant cannot do this when shards differ in class count.
slices_per_shard = [config.resolve_slices_for_shard(len(np.unique(y_shards[i])))
                    for i in range(len(shards))]
shard_class_counts = [len(np.unique(y_shards[i])) for i in range(len(shards))]
print(f"\nSlice allocation ({'derived' if not isinstance(config.NUM_SLICES_PER_SHARD, int) else 'manual'}): "
      f"{[f'shard {i+1}: {c} classes -> {s} slices' for i, (c, s) in enumerate(zip(shard_class_counts, slices_per_shard))]}")
for _w in config.validate_partition(shard_class_counts):
    print(f"   - WARNING: {_w}")

for i, shard in enumerate(shards):
    print(f"\nProcessing Shard {i+1}/{num_shards}...")
    slices, y_slice, slice_indices, slice_dists = create_slices(
        shard, y_shards[i], slices_per_shard[i], index_shards[i], class_names
    )
    shard_slices_list.append(slices)
    y_slices_list.append(y_slice)
    shard_indices_list.append(slice_indices)
    slice_class_dists_list.append(slice_dists) 
    
    # --- Terminal Output for Slices ---
    print(f"Slice Details for Shard {i+1}:")
    for j, s_dist in enumerate(slice_dists):
        total_slice_samples = sum(s_dist.values())
        print(f"  Slice {j+1} (Total Samples: {total_slice_samples:,})")
        for class_name, count in sorted(s_dist.items()):
             if count > 0:
                print(f"    - {class_name:<12}: {count:>5,} samples")

# 3. Save the processed data using NPY format
print("\nSaving SISA data in NPY format...")
save_start_time = time.time()

sisa_data_dir = f'{base_dir}/sisa_data'
os.makedirs(sisa_data_dir, exist_ok=True)
shards_data_dir = f'{sisa_data_dir}/shards'
os.makedirs(shards_data_dir, exist_ok=True)

# Save main metadata
metadata = {
    'num_shards': num_shards,
    # W28: the residual backbone derives its stage count from input resolution, so
    # the resolution has to come from the data rather than a config constant.
    'input_size': int(x_train.shape[-1]),
    'num_slices': max(slices_per_shard),  # legacy scalar = max, for older readers
    'slices_per_shard': slices_per_shard,  # W26: authoritative per-shard slice counts
    'class_names': class_names,
    'split_info': split_info,
    'sharding_strategy': split_info.get('sharding_strategy', 'class_isolation'),
    'slicing_strategy': 'class_sequential',
    'processing_timestamp': datetime.now().isoformat(),
    # --- Normalization stats for consistent preprocessing ---
    'normalization_mean': train_mean,
    'normalization_std': train_std,
    # --- Shard load balancing info ---
    'shard_info': {
        f'shard_{i+1}': {
            'expected_samples': sum(len(slice_data) for slice_data in shard_slices_list[i]) if i < len(shard_slices_list) else 0,
            'classes_assigned': [class_names[cls] for cls in set().union(*[np.unique(slice_y) for slice_y in y_slices_list[i]])] if i < len(y_slices_list) and y_slices_list[i] else []
        } for i in range(num_shards)
    }
}
test_data_dir = f'{sisa_data_dir}/test_data'
validation_data_dir = f'{sisa_data_dir}/validation_data'

# Every array this script writes, in one place, so the idempotency guard below can
# check them all before anything is written.
expected_arrays = [
    (f'{test_data_dir}/x_test.npy', x_test),
    (f'{test_data_dir}/y_test.npy', y_test),
    (f'{validation_data_dir}/x_validation.npy', x_val),
    (f'{validation_data_dir}/y_validation.npy', y_val),
]
for shard_idx in range(num_shards):
    if not y_slices_list[shard_idx]:
        continue
    shard_dir = f'{shards_data_dir}/shard_{shard_idx + 1}'
    for slice_idx in range(len(shard_slices_list[shard_idx])):
        expected_arrays.append((f'{shard_dir}/slice_{slice_idx}_x.npy', shard_slices_list[shard_idx][slice_idx]))
        expected_arrays.append((f'{shard_dir}/slice_{slice_idx}_y.npy', y_slices_list[shard_idx][slice_idx]))
        expected_arrays.append((f'{shard_dir}/slice_{slice_idx}_idx.npy', shard_indices_list[shard_idx][slice_idx]))


def _npy_matches(path, arr):
    """True if `path` already holds an array of exactly this shape and dtype.

    mmap_mode='r' parses the .npy header only -- it never faults in the data pages, so
    this stays cheap over ~705 MB of arrays. The mapping is closed explicitly rather
    than left to refcounting: `all(...)` below short-circuits on the first mismatch, so
    if the guard then decides to rewrite, any path already checked would still be mapped
    when np.save reopens it -- and on Windows an open mapping means a sharing violation.
    """
    if not os.path.exists(path):
        return False
    existing = None
    try:
        existing = np.load(path, mmap_mode='r')
        # Compare against the dtype save_images would WRITE, not the in-memory dtype:
        # image arrays are float32 here and uint8 on disk (W34). This is also what makes
        # the format migration automatic -- legacy float32 files no longer match, so the
        # guard falls through to a rewrite exactly once and they come back as uint8.
        return existing.shape == arr.shape and existing.dtype == encoded_dtype(arr)
    except (ValueError, OSError):
        return False  # missing, truncated, or not a readable .npy -- rewrite it
    finally:
        if existing is not None:
            existing._mmap.close()


def _fingerprint(meta):
    """The parts of metadata.json that determine the array CONTENTS. Excludes
    processing_timestamp, which changes on every run by construction."""
    return {k: v for k, v in meta.items() if k != 'processing_timestamp'}


# Skip the rewrite when what is on disk already matches this config. These arrays are
# ~705 MB for CIFAR-10 and every invocation rewrote all of them unconditionally, even
# when nothing about the partition had changed -- which is pure disk wear when you are
# just re-running training against the same data.
#
# The check is a CONFIG fingerprint plus a shape/dtype check, not a content hash: the
# pipeline is fully deterministic under config.SEED (set_seed at import), so an
# identical dataset, seed, shard/slice partition, per-shard class assignment and
# normalization stats reproduce byte-identical arrays. What it deliberately cannot see
# is an edit to the sharding/slicing code itself -- pass --force after one of those.
data_is_current = False
if not args.force and os.path.exists(f'{sisa_data_dir}/metadata.json'):
    try:
        with open(f'{sisa_data_dir}/metadata.json') as f:
            existing_metadata = json.load(f)
        data_is_current = (_fingerprint(existing_metadata) == _fingerprint(metadata)
                           and all(_npy_matches(p, a) for p, a in expected_arrays))
    except (json.JSONDecodeError, OSError):
        data_is_current = False

total_slices_saved = sum(len(s) for s in shard_slices_list)

# The JSON metadata is ~3 KB and is rewritten UNCONDITIONALLY, even when the arrays are
# skipped. Unlearning mutates these files in place (SISAUnlearning._update_shard_metadata
# rewrites class_indices_present), so re-running data processing has to restore the
# pristine partition description regardless of whether the arrays needed rewriting.
with open(f'{sisa_data_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

os.makedirs(test_data_dir, exist_ok=True)
os.makedirs(validation_data_dir, exist_ok=True)

for shard_idx in range(num_shards):
    shard_dir = f'{shards_data_dir}/shard_{shard_idx + 1}'
    os.makedirs(shard_dir, exist_ok=True)

    if not y_slices_list[shard_idx]: continue

    all_shard_labels = np.concatenate(y_slices_list[shard_idx])
    unique_classes_in_shard = np.unique(all_shard_labels)
    class_names_in_shard = [class_names[c] for c in unique_classes_in_shard]
    
    shard_metadata = {
        'shard_index': shard_idx + 1,
        'total_samples': len(all_shard_labels),
        'class_indices_present': [int(c) for c in unique_classes_in_shard],
        'class_names_present': class_names_in_shard,
        # --- Load balancing metadata ---
        'load_balance_score': len(all_shard_labels) / (len(x_train) / num_shards),  # Relative load vs ideal
        'unlearned_classes': [],  # Track what classes have been unlearned from this shard
        'performance_metadata': {
            'expected_training_time_ratio': len(all_shard_labels) / (len(x_train) / num_shards),
            'memory_usage_ratio': len(all_shard_labels) / (len(x_train) / num_shards),
            'optimal_batch_size': min(64, max(16, len(all_shard_labels) // 100))  # Adaptive batch size
        }
    }
    with open(f'{shard_dir}/metadata.json', 'w') as f:
        json.dump(shard_metadata, f, indent=2)

expected_mb = sum(a.size * encoded_dtype(a).itemsize for _, a in expected_arrays) / 1e6
if data_is_current:
    print(f"   - Arrays on disk already match this config ({total_slices_saved} slices, "
          f"{expected_mb:.0f} MB) -- skipping rewrite. Pass --force to rebuild anyway.")
else:
    # save_images stores pixels as uint8 (4x smaller than the float32 the pipeline holds
    # in memory) and passes labels/indices through unchanged. Lossless -- load_images
    # reconstructs the identical float32 bits. See utils/data_io.py.
    for path, arr in expected_arrays:
        save_images(path, arr)
    print(f"   - Wrote {len(expected_arrays)} arrays ({expected_mb:.0f} MB, {total_slices_saved} slices).")

save_time = time.time() - save_start_time

# 4. Generate Visualizations
create_data_visualizations()

print("\n" + "=" * 60)
print("Data Processing Completed")
print("=" * 60)
print(f"Project directory: {base_dir}")
print(f"Reports and visualizations saved to: {data_info_dir}")
print(f"Configuration: {num_shards} shards, slices per shard: {slices_per_shard}")
print("Strategy: Class Isolation sharding and Class-Sequential slicing")
print(f"NPY data saved to: {sisa_data_dir}")
print(f"Save time: {save_time:.2f} seconds ({total_slices_saved} slices)")
print("=" * 60)
print(f"Log saved to: {_log_path}")

_restore_logging()