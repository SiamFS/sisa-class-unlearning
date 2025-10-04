import numpy as np
from collections import Counter

def create_shards_with_indices(X, y, part_number, class_names=None):

    if part_number <= 0:
        raise ValueError("part_number must be positive")
    
    print(f"Creating {part_number} shards with class-isolation strategy...")
    
    train_indices = np.arange(len(X))

    return _create_class_isolated_shards(X, y, train_indices, part_number, class_names)

def _create_class_isolated_shards(X, y, indices, part_number, class_names):

    # Mathematical parameters from data_process.txt analysis
    ALPHA = 5.0  # Imbalance threshold multiplier
    BETA = 0.6   # Max shard fraction per class
    GAMMA = 0.5  # Minimum split efficiency factor
    MAX_IMBALANCE_RATIO = 3.0  # Maximum acceptable shard imbalance
    
    # 1. Group all data by class
    class_data = {}
    unique_classes = sorted(np.unique(y))
    total_samples = len(X)
    
    for class_idx in unique_classes:
        mask = (y == class_idx)
        class_count = len(X[mask])
        class_data[class_idx] = {
            'X': X[mask],
            'y': y[mask],
            'indices': indices[mask],
            'count': class_count,
            'fraction': class_count / total_samples
        }
    
    print("🎯 Analyzing class distribution for balanced sharding...")
    
    # 2. Check for extreme class imbalances that need splitting
    classes_to_split = []
    for class_idx, data in class_data.items():
        class_fraction = data['fraction']
        class_name = class_names[class_idx]
        
        # Apply adaptive threshold formula
        if class_fraction > BETA:  # Class is too large for single shard
            print(f"     Class '{class_name}' has {class_fraction:.2%} of data (>{BETA:.0%}) - considering for splitting")
            classes_to_split.append(class_idx)
        else:
            print(f"    Class '{class_name}': {class_fraction:.2%} of data (acceptable)")
    
    # 3. Smart class assignment with load balancing
    shards_content = [[] for _ in range(part_number)]
    shard_sizes = np.zeros(part_number)
    
    # Sort classes by size (descending) for better balancing
    sorted_class_indices = sorted(class_data.keys(), key=lambda k: class_data[k]['count'], reverse=True)
    
    print("\n🔄 Assigning classes to shards with balanced load distribution...")
    
    for class_idx in sorted_class_indices:
        class_name = class_names[class_idx]
        class_count = class_data[class_idx]['count']
        
        # Find target shard with advanced balancing
        if class_idx in classes_to_split:
            # For large classes, use asymmetric splitting strategy
            print(f"   🔀 Applying asymmetric splitting for '{class_name}' ({class_count} samples)")
            target_shard_idx = _apply_asymmetric_splitting(
                class_idx, class_data, shards_content, shard_sizes, part_number, GAMMA
            )
        else:
            # Standard balanced assignment
            target_shard_idx = _find_balanced_shard(shard_sizes, class_count, MAX_IMBALANCE_RATIO)
        
        # Add class to selected shard
        shards_content[target_shard_idx].append(class_idx)
        shard_sizes[target_shard_idx] += class_count
        
        print(f"    Assigned '{class_name}' ({class_count} samples) to Shard {target_shard_idx + 1}")
        
        # Check balance after assignment
        if shard_sizes.max() / (shard_sizes[shard_sizes > 0].min() + 1e-6) > MAX_IMBALANCE_RATIO:
            print(f"    Warning: Shard imbalance ratio is {shard_sizes.max() / shard_sizes[shard_sizes > 0].min():.2f}x")
    
    # 4. Display final shard balance analysis
    print(f"\n Final Shard Balance Analysis:")
    for i, size in enumerate(shard_sizes):
        if size > 0:
            print(f"   Shard {i + 1}: {size:,} samples ({size/total_samples:.1%})")
    
    if shard_sizes.max() > 0 and shard_sizes[shard_sizes > 0].min() > 0:
        final_ratio = shard_sizes.max() / shard_sizes[shard_sizes > 0].min()
        if final_ratio <= MAX_IMBALANCE_RATIO:
            print(f"    Balance ratio: {final_ratio:.2f}x (acceptable)")
        else:
            print(f"    Balance ratio: {final_ratio:.2f}x (may cause training inefficiencies)")

    # 5. Build the final shards from the class assignments
    final_shards = []
    final_y_shards = []
    final_index_shards = []
    class_distributions = []

    for shard_idx in range(part_number):
        classes_in_shard = shards_content[shard_idx]
        
        if not classes_in_shard: 
            continue # Skip empty shards
        
        # Collect all data for the classes assigned to this shard
        shard_x_parts = [class_data[c]['X'] for c in classes_in_shard]
        shard_y_parts = [class_data[c]['y'] for c in classes_in_shard]
        shard_indices_parts = [class_data[c]['indices'] for c in classes_in_shard]
        
        # Concatenate parts into a single shard
        final_shards.append(np.vstack(shard_x_parts))
        final_y_shards.append(np.concatenate(shard_y_parts))
        final_index_shards.append(np.concatenate(shard_indices_parts))
        
        # Calculate final class distribution for this shard
        class_counts = Counter(final_y_shards[-1])
        class_dist = {class_names[k]: v for k, v in class_counts.items()}
        class_distributions.append(class_dist)
        
    split_info = {
        'sharding_strategy': 'adaptive_threshold_balanced',
        'assignment_strategy': 'balanced_load_distribution',
        'data_structure': 'Sharded with class isolation and load balancing',
        'balance_parameters': {
            'max_imbalance_ratio': MAX_IMBALANCE_RATIO,
            'max_class_fraction': BETA,
            'split_efficiency_factor': GAMMA
        }
    }
    
    print(" Adaptive threshold-based sharding completed with load balancing.")
    return final_shards, final_y_shards, final_index_shards, class_distributions, split_info


def _find_balanced_shard(shard_sizes, class_count, max_imbalance_ratio):
    """Find the best shard that maintains balance constraints."""
    # Try each shard and calculate resulting imbalance
    best_shard = 0
    best_imbalance = float('inf')
    
    for shard_idx in range(len(shard_sizes)):
        # Calculate imbalance if we add this class to this shard
        temp_sizes = shard_sizes.copy()
        temp_sizes[shard_idx] += class_count
        
        if temp_sizes.min() > 0:
            imbalance_ratio = temp_sizes.max() / temp_sizes.min()
        else:
            imbalance_ratio = temp_sizes.max()
        
        # Prefer shards that maintain better balance
        if imbalance_ratio < best_imbalance:
            best_imbalance = imbalance_ratio
            best_shard = shard_idx
    
    return best_shard


def _apply_asymmetric_splitting(class_idx, class_data, shards_content, shard_sizes, part_number, gamma):
    """Apply asymmetric splitting for large classes to maintain specialization."""
    # For now, assign to the smallest shard (can be enhanced with actual splitting later)
    # This maintains class isolation while improving balance
    target_shard = np.argmin(shard_sizes)
    
    # Future enhancement: Could implement 70-30 asymmetric split here
    # For now, we maintain full class isolation as per SISA principles
    
    return target_shard