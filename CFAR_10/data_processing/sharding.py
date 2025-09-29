import numpy as np
from collections import Counter

def create_shards_with_indices(X, y, part_number, class_names=None):
    """
    Divides a dataset into shards using a class isolation strategy to ensure
    no single class is split across multiple shards.
    """
    if part_number <= 0:
        raise ValueError("part_number must be positive")
    
    print(f"Creating {part_number} shards with class-isolation strategy...")
    
    train_indices = np.arange(len(X))

    return _create_class_isolated_shards(X, y, train_indices, part_number, class_names)

def _create_class_isolated_shards(X, y, indices, part_number, class_names):
    """
    Assigns whole classes to shards, aiming for balanced shard sizes.
    """
    # 1. Group all data by class
    class_data = {}
    unique_classes = sorted(np.unique(y))
    for class_idx in unique_classes:
        mask = (y == class_idx)
        class_data[class_idx] = {
            'X': X[mask],
            'y': y[mask],
            'indices': indices[mask],
            'count': len(X[mask])
        }
    
    # 2. Assign classes to shards using a greedy algorithm
    shards_content = [[] for _ in range(part_number)]
    shard_sizes = np.zeros(part_number)
    
    # Sort classes by size (descending) to improve balancing
    sorted_class_indices = sorted(class_data.keys(), key=lambda k: class_data[k]['count'], reverse=True)
    
    print("Assigning classes to shards to maintain isolation...")
    for class_idx in sorted_class_indices:
        # Find the shard with the minimum current size
        target_shard_idx = np.argmin(shard_sizes)
        
        # Add the entire class to that shard
        shards_content[target_shard_idx].append(class_idx)
        shard_sizes[target_shard_idx] += class_data[class_idx]['count']
        class_name = class_names[class_idx]
        print(f"   - Assigning class '{class_name}' ({class_data[class_idx]['count']} samples) to Shard {target_shard_idx + 1}")

    # 3. Build the final shards from the class assignments
    final_shards = []
    final_y_shards = []
    final_index_shards = []
    class_distributions = []

    for shard_idx in range(part_number):
        classes_in_shard = shards_content[shard_idx]
        
        if not classes_in_shard: continue # Skip empty shards
        
        # Collect all data for the classes assigned to this shard
        shard_X_parts = [class_data[c]['X'] for c in classes_in_shard]
        shard_y_parts = [class_data[c]['y'] for c in classes_in_shard]
        shard_indices_parts = [class_data[c]['indices'] for c in classes_in_shard]
        
        # Concatenate parts into a single shard
        final_shards.append(np.vstack(shard_X_parts))
        final_y_shards.append(np.concatenate(shard_y_parts))
        final_index_shards.append(np.concatenate(shard_indices_parts))
        
        # Calculate final class distribution for this shard
        class_counts = Counter(final_y_shards[-1])
        class_dist = {class_names[k]: v for k, v in class_counts.items()}
        class_distributions.append(class_dist)
        
    split_info = {
        'sharding_strategy': 'class_isolation',
        'assignment_strategy': 'greedy_balancing',
        'data_structure': 'Sharded with class isolation'
    }
    
    print("Class-isolated sharding completed.")
    return final_shards, final_y_shards, final_index_shards, class_distributions, split_info