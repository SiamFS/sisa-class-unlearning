"""
Smart Replay Buffer for SISA Framework
Implements gradient-based importance sampling with temporal decay
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json


class SmartReplayBuffer:
    """
    Smart replay buffer that maintains importance scores and temporal information
    for more intelligent sample selection during incremental learning.
    """
    
    def __init__(self, decay_rate: float = 0.95, importance_weight: float = 0.7, 
                 temporal_weight: float = 0.3, max_samples_per_class: int = 1000):
        """
        Initialize smart replay buffer.
        
        Args:
            decay_rate: Exponential decay rate for temporal importance
            importance_weight: Weight for gradient-based importance (0-1)
            temporal_weight: Weight for temporal decay (0-1)
            max_samples_per_class: Maximum samples to store per class
        """
        self.buffer: Dict[int, Dict] = {}
        self.decay_rate = decay_rate
        self.importance_weight = importance_weight
        self.temporal_weight = temporal_weight
        self.max_samples_per_class = max_samples_per_class
        self.current_slice = 0
        
        # Ensure weights sum to 1
        total_weight = importance_weight + temporal_weight
        self.importance_weight /= total_weight
        self.temporal_weight /= total_weight
    
    def add_samples(self, x_data: np.ndarray, y_data: np.ndarray, 
                   model: torch.nn.Module, device: torch.device):
        """
        Add new samples to replay buffer with importance scores.
        
        Args:
            x_data: Feature data [N, C, H, W]
            y_data: Labels [N]
            model: Current model for computing importance
            device: Computing device
        """
        self.current_slice += 1
        
        # Compute importance scores for new samples
        importance_scores = self._compute_importance_scores(x_data, y_data, model, device)
        
        unique_labels = np.unique(y_data)
        for label in unique_labels:
            mask = (y_data == label)
            class_x = x_data[mask]
            class_y = y_data[mask]
            class_importance = importance_scores[mask]
            
            if label in self.buffer:
                # Add to existing class buffer
                self._add_to_existing_class(label, class_x, class_y, class_importance)
            else:
                # Create new class buffer
                self.buffer[label] = {
                    'X': class_x,
                    'y': class_y,
                    'importance': class_importance,
                    'slice_added': np.full(len(class_x), self.current_slice),
                    'sample_count': len(class_x)
                }
    
    def _compute_importance_scores(self, x_data: np.ndarray, y_data: np.ndarray,
                                 model: torch.nn.Module, device: torch.device) -> np.ndarray:
        """
        Compute gradient-based importance scores for samples.
        
        Args:
            x_data: Feature data
            y_data: Labels  
            model: Current model
            device: Computing device
            
        Returns:
            Importance scores for each sample
        """
        model.eval()
        importance_scores = []
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Process in small batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(x_data), batch_size):
            batch_x = torch.from_numpy(x_data[i:i+batch_size]).float().to(device)
            batch_y = torch.from_numpy(y_data[i:i+batch_size]).long().to(device)
            
            batch_x.requires_grad_(True)
            
            # Forward pass
            outputs = model(batch_x)
            losses = criterion(outputs, batch_y)
            
            # Compute gradient norms as importance scores
            batch_importance = []
            for j, loss in enumerate(losses):
                # Gradient w.r.t input
                grad = torch.autograd.grad(loss, batch_x[j:j+1], retain_graph=True, allow_unused=True)[0]
                if grad is not None:
                    importance = torch.norm(grad).item()
                else:
                    importance = 0.0  # Default importance for unused tensors
                batch_importance.append(importance)
            
            importance_scores.extend(batch_importance)
        
        return np.array(importance_scores)
    
    def _add_to_existing_class(self, label: int, class_x: np.ndarray, 
                              class_y: np.ndarray, class_importance: np.ndarray):
        """Add samples to existing class buffer with reservoir sampling if needed."""
        current_buffer = self.buffer[label]
        
        # Simple concatenation if under limit
        new_total = current_buffer['sample_count'] + len(class_x)
        if new_total <= self.max_samples_per_class:
            current_buffer['X'] = np.vstack([current_buffer['X'], class_x])
            current_buffer['y'] = np.concatenate([current_buffer['y'], class_y])
            current_buffer['importance'] = np.concatenate([current_buffer['importance'], class_importance])
            current_buffer['slice_added'] = np.concatenate([
                current_buffer['slice_added'], 
                np.full(len(class_x), self.current_slice)
            ])
            current_buffer['sample_count'] = new_total
        else:
            # Reservoir sampling with importance weighting
            self._reservoir_sample_with_importance(label, class_x, class_y, class_importance)
    
    def _reservoir_sample_with_importance(self, label: int, new_x: np.ndarray,
                                        new_y: np.ndarray, new_importance: np.ndarray):
        """Intelligent reservoir sampling based on importance scores."""
        current_buffer = self.buffer[label]
        
        # Combine old and new samples
        all_x = np.vstack([current_buffer['X'], new_x])
        all_y = np.concatenate([current_buffer['y'], new_y])
        all_importance = np.concatenate([current_buffer['importance'], new_importance])
        all_slice_added = np.concatenate([
            current_buffer['slice_added'],
            np.full(len(new_x), self.current_slice)
        ])
        
        # Compute combined scores (importance + temporal decay)
        temporal_scores = self.decay_rate ** (self.current_slice - all_slice_added)
        combined_scores = (self.importance_weight * all_importance + 
                          self.temporal_weight * temporal_scores)
        
        # Sample top samples based on combined scores
        top_indices = np.argsort(combined_scores)[-self.max_samples_per_class:]
        
        # Update buffer
        current_buffer['X'] = all_x[top_indices]
        current_buffer['y'] = all_y[top_indices]
        current_buffer['importance'] = all_importance[top_indices]
        current_buffer['slice_added'] = all_slice_added[top_indices]
        current_buffer['sample_count'] = len(top_indices)
    
    def sample_for_replay(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from replay buffer using smart sampling strategy.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Tuple of (X_replay, y_replay)
        """
        if not self.buffer or n_samples <= 0:
            return np.empty((0, 32, 32, 3)), np.empty(0, dtype=np.int64)
        
        available_classes = list(self.buffer.keys())
        n_classes = len(available_classes)
        
        # Distribute samples across classes
        samples_per_class = n_samples // n_classes
        extra_samples = n_samples % n_classes
        
        replay_x_list = []
        replay_y_list = []
        
        for i, class_label in enumerate(available_classes):
            class_data = self.buffer[class_label]
            
            # Number of samples for this class
            class_n_samples = samples_per_class + (1 if i < extra_samples else 0)
            class_n_samples = min(class_n_samples, class_data['sample_count'])
            
            if class_n_samples > 0:
                # Smart sampling based on combined importance + temporal scores
                temporal_scores = self.decay_rate ** (self.current_slice - class_data['slice_added'])
                combined_scores = (self.importance_weight * class_data['importance'] + 
                                 self.temporal_weight * temporal_scores)
                
                # Softmax for probability distribution
                probabilities = F.softmax(torch.from_numpy(combined_scores), dim=0).numpy()
                
                # Sample indices
                sampled_indices = np.random.choice(
                    len(class_data['X']), 
                    class_n_samples, 
                    replace=False if len(class_data['X']) >= class_n_samples else True,
                    p=probabilities
                )
                
                replay_x_list.append(class_data['X'][sampled_indices])
                replay_y_list.append(class_data['y'][sampled_indices])
        
        if replay_x_list:
            return np.vstack(replay_x_list), np.concatenate(replay_y_list)
        else:
            return np.empty((0, 32, 32, 3)), np.empty(0, dtype=np.int64)
    
    def get_buffer_stats(self) -> Dict:
        """Get statistics about the current buffer state."""
        stats = {
            'total_classes': len(self.buffer),
            'total_samples': sum(data['sample_count'] for data in self.buffer.values()),
            'current_slice': self.current_slice,
            'class_distribution': {
                label: data['sample_count'] 
                for label, data in self.buffer.items()
            }
        }
        return stats
    
    def save_buffer(self, filepath: str):
        """Save buffer state (metadata only, not raw data)."""
        metadata = {
            'current_slice': self.current_slice,
            'decay_rate': self.decay_rate,
            'importance_weight': self.importance_weight,
            'temporal_weight': self.temporal_weight,
            'max_samples_per_class': self.max_samples_per_class,
            'buffer_stats': self.get_buffer_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def __len__(self) -> int:
        """Total number of samples in buffer."""
        return sum(data['sample_count'] for data in self.buffer.values())


def create_smart_replay_buffer(config) -> SmartReplayBuffer:
    """Factory function to create smart replay buffer with config."""
    return SmartReplayBuffer(
        decay_rate=getattr(config, 'REPLAY_DECAY_RATE', 0.95),
        importance_weight=getattr(config, 'REPLAY_IMPORTANCE_WEIGHT', 0.7),
        temporal_weight=getattr(config, 'REPLAY_TEMPORAL_WEIGHT', 0.3),
        max_samples_per_class=getattr(config, 'MAX_REPLAY_SAMPLES_PER_CLASS', 1000)
    )