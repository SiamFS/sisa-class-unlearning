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
# Setup automatic output logging
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

# Initialize automatic logging
sys.stdout = Logger('data_processing.txt')

# Add header with timestamp
print("=" * 80)
print("SISA FRAMEWORK - DATA PROCESSING")
print("=" * 80)
print(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

sys.path.append('..')
import config

# Get class names from dataset (configurable for different datasets)
class_names = torchvision.datasets.CIFAR10(root=config.DATA_DIR, train=True, download=True).classes

# Define command-line arguments with config.py defaults
parser = argparse.ArgumentParser(description='SISA Framework Sequential Data Processing')
parser.add_argument('--num-shards', type=int, default=config.NUM_SHARDS, help='Number of shards')
parser.add_argument('--num-slices', type=int, default=config.NUM_SLICES_PER_SHARD, help='Number of slices per shard')

args = parser.parse_args()

project_name = config.PROJECT_NAME
num_shards = args.num_shards
num_slices = args.num_slices
base_dir = f"../{config.PROJECTS_DIR}/{project_name}"
data_info_dir = f"{base_dir}/data_info"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(data_info_dir, exist_ok=True)


print(f"   - Number of Shards: {num_shards}")
print(f"   - Number of Slices per Shard: {num_slices}")

def load_cifar10_data():
    """Load CIFAR-10 data and implement a 70-10-20 split (train-validation-test)."""
    cifar10_train = torchvision.datasets.CIFAR10(root=config.DATA_DIR, train=True, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=config.DATA_DIR, train=False, download=True)

    x_train_val = cifar10_train.data
    y_train_val = np.array(cifar10_train.targets)
    x_test = cifar10_test.data
    y_test = np.array(cifar10_test.targets)

    # Combine for unified processing and splitting
    x_combined = np.concatenate([x_train_val, x_test], axis=0)
    y_combined = np.concatenate([y_train_val, y_test], axis=0)

    # Convert to PyTorch format (C, H, W) and normalize to [0, 1]
    x_combined = np.transpose(x_combined, (0, 3, 1, 2)).astype(np.float32) / 255.0
    
    print(f"Loaded {config.DATASET_NAME} with preprocessing:")
    print(f"   - Combined data shape: {x_combined.shape}")
    print(f"   - Data range: [{x_combined.min():.2f}, {x_combined.max():.2f}]")

    # Implement 70-10-20 split (train-validation-test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp
    )

    total_samples = len(x_combined)
    print("Data split completed:")
    print(f"   - Train: {len(x_train):,} samples ({len(x_train)/total_samples:.1%})")
    print(f"   - Validation: {len(x_val):,} samples ({len(x_val)/total_samples:.1%})")
    print(f"   - Test: {len(x_test):,} samples ({len(x_test)/total_samples:.1%})")
    
    split_info = {
        'total_samples': total_samples,
        'train_samples': len(x_train),
        'val_samples': len(x_val),
        'test_samples': len(x_test),
        'random_state': 42
    }
    
    return x_train, y_train, split_info, x_test, y_test, x_val, y_val

def create_data_visualizations():
    """Generates comprehensive visualizations for SISA data processing."""
    
    # Prepare shards data in the format expected by visualization functions
    shards_data = []
    for shard_idx in range(num_shards):
        slices_data = []
        for slice_idx in range(num_slices):
            if slice_idx < len(shard_slices_list[shard_idx]):
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
x_train, y_train, split_info, x_test, y_test, x_val, y_val = load_cifar10_data()

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

for i, shard in enumerate(shards):
    print(f"\nProcessing Shard {i+1}/{num_shards}...")
    slices, y_slice, slice_indices, slice_dists = create_slices(
        shard, y_shards[i], num_slices, index_shards[i], class_names
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
    'num_slices': num_slices,
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
with open(f'{sisa_data_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save test and validation data
test_data_dir = f'{sisa_data_dir}/test_data'
os.makedirs(test_data_dir, exist_ok=True)
np.save(f'{test_data_dir}/x_test.npy', x_test)
np.save(f'{test_data_dir}/y_test.npy', y_test)

validation_data_dir = f'{sisa_data_dir}/validation_data'
os.makedirs(validation_data_dir, exist_ok=True)
np.save(f'{validation_data_dir}/x_validation.npy', x_val)
np.save(f'{validation_data_dir}/y_validation.npy', y_val)

# Save shards and slices
total_slices_saved = 0
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

    for slice_idx in range(num_slices):
        if slice_idx < len(shard_slices_list[shard_idx]):
            np.save(f'{shard_dir}/slice_{slice_idx}_x.npy', shard_slices_list[shard_idx][slice_idx])
            np.save(f'{shard_dir}/slice_{slice_idx}_y.npy', y_slices_list[shard_idx][slice_idx])
            np.save(f'{shard_dir}/slice_{slice_idx}_idx.npy', shard_indices_list[shard_idx][slice_idx])
            total_slices_saved += 1

save_time = time.time() - save_start_time

# 4. Generate Visualizations
create_data_visualizations()

print("\n" + "=" * 60)
print("Data Processing Completed")
print("=" * 60)
print(f"Project directory: {base_dir}")
print(f"Reports and visualizations saved to: {data_info_dir}")
print(f"Configuration: {num_shards} shards × {num_slices} slices")
print("Strategy: Class Isolation sharding and Class-Sequential slicing")
print(f"NPY data saved to: {sisa_data_dir}")
print(f"Save time: {save_time:.2f} seconds ({total_slices_saved} slices)")
print("=" * 60)