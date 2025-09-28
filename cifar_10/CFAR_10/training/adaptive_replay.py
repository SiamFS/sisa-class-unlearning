"""
Adaptive Replay Ratio Management for SISA Framework
Implements dynamic replay ratio adjustment based on multiple factors
"""

import numpy as np
from typing import Dict, Optional, Tuple

class AdaptiveReplayManager:
    """
    Manages dynamic replay ratios based on multiple factors:
    - Buffer size and available replay samples
    - Slice position in training sequence  
    - Catastrophic forgetting risk assessment
    - Class balance considerations
    """
    
    def __init__(self, base_replay_ratio: float = 0.3, 
                 min_ratio: float = 0.1, max_ratio: float = 0.6,
                 dataset_size: Optional[int] = None):
        """
        Initialize adaptive replay manager.
        
        Args:
            base_replay_ratio: Starting replay ratio (config default)
            min_ratio: Minimum allowed replay ratio
            max_ratio: Maximum allowed replay ratio
            dataset_size: Total dataset size (for scaling thresholds)
        """
        self.base_ratio = base_replay_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.dataset_size = dataset_size
        
        # Dynamic thresholds based on dataset size
        self._compute_dynamic_thresholds()
        
        # Track forgetting statistics
        self.slice_forgetting_scores = {}
        self.class_stability_scores = {}
    
    def _compute_dynamic_thresholds(self):
        """Compute dynamic thresholds based on dataset characteristics."""
        if self.dataset_size is None:
            # Fallback to default thresholds for unknown dataset size
            self.small_buffer_threshold = 1000
            self.large_buffer_threshold = 10000
        else:
            # Scale thresholds based on dataset size
            # Small: < 5% of dataset, Large: > 25% of dataset
            self.small_buffer_threshold = max(100, int(0.05 * self.dataset_size))
            self.large_buffer_threshold = max(1000, int(0.25 * self.dataset_size))
        
    def compute_adaptive_replay_ratio(self, 
                                    current_slice: int,
                                    total_slices: int,
                                    replay_buffer_size: int,
                                    new_data_size: int,
                                    training_type: str = 'fresh',
                                    class_distribution: Optional[Dict[int, int]] = None,
                                    current_shard: int = 0,
                                    total_shards: int = 1) -> float:
        """
        Compute optimal replay ratio for current training context.
        
        Args:
            current_slice: Current slice index (0-based)
            total_slices: Total slices in current shard
            replay_buffer_size: Number of samples in replay buffer
            new_data_size: Number of new samples in current slice
            training_type: 'fresh', 'incremental', or 'unlearning'
            class_distribution: Class distribution in replay buffer
            current_shard: Current shard index (0-based)
            total_shards: Total shards in system
            
        Returns:
            Optimal replay ratio between min_ratio and max_ratio
        """
        # Handle edge cases
        if replay_buffer_size == 0:
            return 0.0  # No replay possible
        if new_data_size == 0:
            return self.min_ratio  # Minimal training data
            
        # Base ratio adjustments
        ratio = self.base_ratio
        
        # 1. Slice Position Factor - truly dynamic based on actual slice numbers
        slice_progress = current_slice / max(1, total_slices - 1) if total_slices > 1 else 0.0
        position_factor = self._compute_position_factor(slice_progress, total_slices)
        
        # 2. Buffer Size Factor - dynamic based on actual buffer and data sizes
        buffer_factor = self._compute_buffer_factor(replay_buffer_size, new_data_size)
        
        # 3. Training Type Factor
        type_factor = self._compute_training_type_factor(training_type)
        
        # 4. Class Balance Factor (if available)
        balance_factor = self._compute_balance_factor(class_distribution) if class_distribution else 1.0
        
        # 5. Forgetting Risk Factor (if we have historical data)
        forgetting_factor = self._compute_forgetting_factor(current_slice)
        
        # 6. Shard Progress Factor - later shards have more forgetting risk
        shard_factor = self._compute_shard_factor(current_shard, total_shards)
        
        # Combine all factors
        ratio = ratio * position_factor * buffer_factor * type_factor * balance_factor * forgetting_factor * shard_factor
        
        # Apply constraints
        ratio = max(self.min_ratio, min(self.max_ratio, ratio))
        
        return ratio
    
    def _compute_position_factor(self, slice_progress: float, total_slices: int) -> float:
        """
        Adjust ratio based on slice position - truly dynamic scaling.
        Early slices: lower ratio (less to replay)
        Later slices: higher ratio (more forgetting risk)
        """
        # Scale the curve based on total number of slices
        if total_slices <= 2:
            # For very few slices, use moderate scaling
            return 0.8 + 0.4 * slice_progress
        elif total_slices <= 5:
            # For moderate slices (like current setup), use standard scaling
            return 0.7 + 0.6 * slice_progress
        else:
            # For many slices, use more aggressive scaling
            return 0.6 + 0.8 * slice_progress
    
    def _compute_shard_factor(self, current_shard: int, total_shards: int) -> float:
        """
        Adjust ratio based on shard position - later shards have more forgetting risk.
        """
        if total_shards <= 1:
            return 1.0
            
        shard_progress = current_shard / max(1, total_shards - 1)
        # Later shards should use slightly more replay to combat cross-shard forgetting
        return 0.95 + 0.15 * shard_progress
    
    def _compute_buffer_factor(self, buffer_size: int, new_data_size: int) -> float:
        """
        Adjust ratio based on replay buffer availability - truly dynamic thresholds.
        """
        if buffer_size == 0:
            return 0.0  # No replay possible
        
        # Ratio of available replay to new data
        replay_to_new_ratio = buffer_size / max(1, new_data_size)
        
        # Dynamic thresholds based on dataset size
        small_threshold = self.small_buffer_threshold
        large_threshold = self.large_buffer_threshold
        
        # Categorize buffer size dynamically
        if buffer_size < small_threshold:  # Small buffer
            return 0.4 + 0.3 * min(1.0, replay_to_new_ratio)
        elif buffer_size > large_threshold:  # Large buffer
            return 1.2 + 0.3 * min(1.0, replay_to_new_ratio / 2.0)
        else:  # Medium buffer - scale based on position between thresholds
            position = (buffer_size - small_threshold) / max(1, large_threshold - small_threshold)
            base_factor = 0.7 + 0.5 * position  # Interpolate between small and large
            return base_factor + 0.2 * min(1.0, replay_to_new_ratio)
    
    def _compute_training_type_factor(self, training_type: str) -> float:
        """
        Adjust ratio based on training context.
        """
        factors = {
            'fresh': 0.8,        # Less replay needed for first slice
            'incremental': 1.0,   # Standard ratio
            'unlearning': 1.2     # More replay to maintain knowledge
        }
        return factors.get(training_type, 1.0)
    
    def _compute_balance_factor(self, class_distribution: Dict[int, int]) -> float:
        """
        Adjust ratio based on class balance in replay buffer.
        """
        if not class_distribution or len(class_distribution) <= 1:
            return 1.0
        
        counts = list(class_distribution.values())
        balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 1.0
        
        # If buffer is imbalanced, increase replay to help balance
        if balance_ratio < 0.5:  # Significantly imbalanced
            return 1.3
        elif balance_ratio < 0.8:  # Moderately imbalanced
            return 1.1
        else:  # Well balanced
            return 1.0
    
    def _compute_forgetting_factor(self, current_slice: int) -> float:
        """
        Adjust ratio based on historical forgetting patterns.
        """
        if current_slice not in self.slice_forgetting_scores:
            return 1.0  # No historical data
        
        forgetting_score = self.slice_forgetting_scores[current_slice]
        
        # Higher forgetting → higher replay ratio
        if forgetting_score > 0.3:  # High forgetting risk
            return 1.4
        elif forgetting_score > 0.15:  # Moderate forgetting risk
            return 1.2
        else:  # Low forgetting risk
            return 0.9
    
    def update_forgetting_statistics(self, 
                                   slice_idx: int,
                                   prev_accuracy: float,
                                   current_accuracy: float,
                                   class_accuracies: Dict[int, float]):
        """
        Update forgetting statistics based on observed performance.
        
        Args:
            slice_idx: Slice index
            prev_accuracy: Accuracy before training this slice
            current_accuracy: Accuracy after training this slice
            class_accuracies: Per-class accuracy dictionary
        """
        # Compute forgetting score (accuracy drop)
        forgetting_score = max(0, prev_accuracy - current_accuracy)
        self.slice_forgetting_scores[slice_idx] = forgetting_score
        
        # Update class stability scores
        for class_idx, accuracy in class_accuracies.items():
            if class_idx not in self.class_stability_scores:
                self.class_stability_scores[class_idx] = []
            self.class_stability_scores[class_idx].append(accuracy)
    
    def get_recommended_ratio_range(self, training_context: str) -> Tuple[float, float]:
        """
        Get recommended ratio range for different training contexts.
        
        Returns:
            (min_recommended, max_recommended) tuple
        """
        recommendations = {
            'early_slices': (0.1, 0.25),      # Limited replay buffer
            'middle_slices': (0.25, 0.4),     # Growing forgetting risk
            'late_slices': (0.35, 0.5),       # High forgetting risk
            'unlearning': (0.3, 0.6),         # Need strong memory retention
            'class_imbalanced': (0.4, 0.6)    # Help balance classes
        }
        return recommendations.get(training_context, (0.2, 0.4))


def create_adaptive_replay_manager(config_obj, dataset_size: Optional[int] = None) -> AdaptiveReplayManager:
    """
    Factory function to create adaptive replay manager from config.
    
    Args:
        config_obj: Configuration object with replay parameters
        dataset_size: Total training dataset size for dynamic threshold scaling
    """
    return AdaptiveReplayManager(
        base_replay_ratio=config_obj.REPLAY_RATIO,
        min_ratio=getattr(config_obj, 'MIN_REPLAY_RATIO', 0.1),
        max_ratio=getattr(config_obj, 'MAX_REPLAY_RATIO', 0.6),
        dataset_size=dataset_size
    )


# Example usage and testing functions
if __name__ == "__main__":
    # Test the adaptive replay manager with different dataset sizes
    print("=== Dynamic Adaptive Replay Ratio Testing ===")
    
    # Test with different dataset sizes (CIFAR-10 vs ImageNet scale)
    dataset_sizes = [50000, 1000000]  # CIFAR-10 size vs larger dataset
    
    for dataset_size in dataset_sizes:
        print(f"\n--- Dataset Size: {dataset_size:,} samples ---")
        manager = AdaptiveReplayManager(dataset_size=dataset_size)
        print(f"Dynamic thresholds - Small: {manager.small_buffer_threshold}, Large: {manager.large_buffer_threshold}")
        
        # Test different scenarios with varying slice counts
        scenarios = [
            ("2 shards, 3 slices each", 2, 3, [(0, 0), (1, 2)]),  # Different configs
            ("3 shards, 5 slices each", 3, 5, [(0, 0), (1, 2), (2, 4)]),
            ("4 shards, 2 slices each", 4, 2, [(0, 0), (2, 1), (3, 1)])
        ]
        
        for scenario_name, total_shards, total_slices, test_points in scenarios:
            print(f"\n  {scenario_name}:")
            
            for shard_idx, slice_idx in test_points:
                # Simulate realistic buffer growth
                base_buffer_size = slice_idx * 2000 + shard_idx * 5000
                new_data_size = int(dataset_size / (total_shards * total_slices))
                
                ratio = manager.compute_adaptive_replay_ratio(
                    current_slice=slice_idx,
                    total_slices=total_slices,
                    replay_buffer_size=base_buffer_size,
                    new_data_size=new_data_size,
                    training_type='incremental',
                    current_shard=shard_idx,
                    total_shards=total_shards
                )
                
                print(f"    Shard {shard_idx+1}/{total_shards}, Slice {slice_idx+1}/{total_slices}: "
                      f"Ratio={ratio:.3f} (Buffer: {base_buffer_size:,}, New: {new_data_size:,})")
    
    print(f"\n=== Key Dynamic Features ===")
    print("Buffer thresholds scale with dataset size")
    print("Position factors adapt to slice/shard counts") 
    print("Ratios adjust to actual buffer vs new data ratios")
    print("No hardcoded assumptions about architecture")