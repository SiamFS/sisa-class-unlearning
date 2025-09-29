import numpy as np
from collections import Counter

def create_slices(X, y, num_slices, indices, class_names=None):
    """
    Create class-sequential slices from a shard.

    Parameters:
        X (np.ndarray): Shard data.
        y (np.ndarray): Shard labels.
        num_slices (int): Number of slices.
        indices (list): Indices of data points.
        class_names (list): List of class names.
        
    Returns:
        tuple: (slices, y_slices, slice_indices, class_distributions)
    """
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    
    total_samples = len(X)
    print(f"Creating {num_slices} class-sequential slices from {total_samples} shard samples...")

    return _create_class_sequential_slices(X, y, num_slices, indices, class_names)

def _create_class_sequential_slices(X, y, num_slices, indices, class_names):
    """Fills slices sequentially with all data from one class before moving to the next."""
    
    # 1. Group shard data by class
    class_data = {}
    unique_classes = sorted(np.unique(y))
    for class_idx in unique_classes:
        mask = (y == class_idx)
        class_data[class_idx] = {
            'X': X[mask],
            'y': y[mask],
            'indices': indices[mask]
        }
    
    # 2. Define slice sizes
    total_samples = len(X)
    base_slice_size = total_samples // num_slices
    remainder = total_samples % num_slices
    
    # 3. Fill slices class-sequentially
    slices_temp_X = [[] for _ in range(num_slices)]
    slices_temp_y = [[] for _ in range(num_slices)]
    slices_temp_indices = [[] for _ in range(num_slices)]
    
    current_slice_idx = 0
    class_data_pointers = {class_idx: 0 for class_idx in unique_classes}

    for class_idx in unique_classes:
        while class_data_pointers[class_idx] < len(class_data[class_idx]['X']):
            if current_slice_idx >= num_slices:
                break

            # Determine space left in the current slice
            slice_target_size = base_slice_size + (1 if current_slice_idx < remainder else 0)
            current_slice_fill = sum(len(d) for d in slices_temp_X[current_slice_idx])
            space_left_in_slice = slice_target_size - current_slice_fill

            if space_left_in_slice <= 0:
                current_slice_idx += 1
                continue
            
            # Determine available data from the current class
            start_ptr = class_data_pointers[class_idx]
            available_data_count = len(class_data[class_idx]['X']) - start_ptr
            
            # Determine how much data to take
            num_to_take = min(space_left_in_slice, available_data_count)
            end_ptr = start_ptr + num_to_take
            
            # Add data to the current slice
            slices_temp_X[current_slice_idx].append(class_data[class_idx]['X'][start_ptr:end_ptr])
            slices_temp_y[current_slice_idx].append(class_data[class_idx]['y'][start_ptr:end_ptr])
            slices_temp_indices[current_slice_idx].append(class_data[class_idx]['indices'][start_ptr:end_ptr])
            
            # Update the pointer
            class_data_pointers[class_idx] = end_ptr

    # 4. Finalize slices
    final_slices = [np.vstack(parts) if parts else np.array([]) for parts in slices_temp_X]
    final_y_slices = [np.concatenate(parts) if parts else np.array([]) for parts in slices_temp_y]
    final_slice_indices = [np.concatenate(parts) if parts else np.array([]) for parts in slices_temp_indices]
    
    # 5. Calculate final distributions
    class_distributions = []
    for slice_y in final_y_slices:
        class_counts = Counter(slice_y)
        class_dist = {class_names[k]: v for k, v in class_counts.items()}
        class_distributions.append(class_dist)

    return final_slices, final_y_slices, final_slice_indices, class_distributions