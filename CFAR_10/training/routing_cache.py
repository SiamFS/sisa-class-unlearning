"""
Gating Network Routing Cache System for SISA Framework

This module implements caching of gating network routing decisions to avoid
redundant computation during multiple evaluations and unlearning operations.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import hashlib

import config
from training.create_model import DEVICE


class RoutingCache:
    """
    Manages caching of gating network routing decisions for SISA framework.
    
    Features:
    - Caches routing decisions to avoid redundant gating network inference
    - Supports sample fingerprinting for cache validity
    - Handles sequential unlearning operations
    - Integrates with metadata system
    """
    
    def __init__(self, project_name: str, cache_dir: Optional[str] = None):
        self.project_name = project_name
        self.base_dir = f"../{config.PROJECTS_DIR}/{project_name}"
        self.cache_dir = cache_dir or os.path.join(self.base_dir, "routing_cache")
        self.routing_cache = {}  # In-memory cache
        self.sample_fingerprints = {}  # For cache validity
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache if available
        self._load_cache()
    
    def _compute_sample_fingerprint(self, x_data: np.ndarray) -> str:
        """Compute a fingerprint for sample data to detect changes."""
        # Use sample of data points for fingerprinting (more efficient)
        sample_indices = np.linspace(0, len(x_data)-1, min(100, len(x_data)), dtype=int)
        sample_data = x_data[sample_indices]
        
        # Create hash from sample statistics
        fingerprint_data = np.concatenate([
            sample_data.flatten()[:1000],  # First 1000 values
            [sample_data.mean(), sample_data.std(), sample_data.min(), sample_data.max()],
            [len(x_data), x_data.shape[1] if len(x_data.shape) > 1 else 0]
        ])
        
        return hashlib.md5(fingerprint_data.tobytes()).hexdigest()
    
    def _load_cache(self):
        """Load existing routing cache from disk."""
        cache_file = os.path.join(self.cache_dir, "routing_decisions.json")
        fingerprint_file = os.path.join(self.cache_dir, "sample_fingerprints.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Convert string keys back to tuples
                    self.routing_cache = {eval(k): v for k, v in cache_data.items()}
                print(f"   Loaded {len(self.routing_cache)} cached routing decisions")
            
            if os.path.exists(fingerprint_file):
                with open(fingerprint_file, 'r') as f:
                    self.sample_fingerprints = json.load(f)
                    
        except Exception as e:
            print(f"   Warning: Could not load routing cache: {e}")
            self.routing_cache = {}
            self.sample_fingerprints = {}
    
    def _save_cache(self):
        """Save routing cache to disk."""
        cache_file = os.path.join(self.cache_dir, "routing_decisions.json")
        fingerprint_file = os.path.join(self.cache_dir, "sample_fingerprints.json")
        
        try:
            # Convert tuple keys to strings for JSON serialization
            cache_data = {str(k): v for k, v in self.routing_cache.items()}
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            with open(fingerprint_file, 'w') as f:
                json.dump(self.sample_fingerprints, f, indent=2)
                
            print(f"   Saved {len(self.routing_cache)} routing decisions to cache")
            
        except Exception as e:
            print(f"   Warning: Could not save routing cache: {e}")
    
    def get_cached_routing(self, dataset_name: str, x_data: np.ndarray) -> Optional[Dict]:
        """
        Get cached routing decisions for a dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'test', 'validation')
            x_data: Input data for fingerprint validation
            
        Returns:
            Dictionary with routing decisions or None if cache invalid/missing
        """
        cache_key = (dataset_name, len(x_data))
        
        if cache_key not in self.routing_cache:
            return None
        
        # Validate cache using fingerprint
        current_fingerprint = self._compute_sample_fingerprint(x_data)
        stored_fingerprint = self.sample_fingerprints.get(dataset_name)
        
        if stored_fingerprint != current_fingerprint:
            print(f"   Cache invalid for {dataset_name}: data fingerprint changed")
            # Remove invalid cache entries
            if cache_key in self.routing_cache:
                del self.routing_cache[cache_key]
            if dataset_name in self.sample_fingerprints:
                del self.sample_fingerprints[dataset_name]
            return None
        
        print(f"   Using cached routing for {dataset_name} ({len(x_data)} samples)")
        return self.routing_cache[cache_key]
    
    def cache_routing_decisions(self, 
                              dataset_name: str, 
                              x_data: np.ndarray,
                              routing_decisions: Dict,
                              gating_model_path: str = None):
        """
        Cache routing decisions for a dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'test', 'validation')
            x_data: Input data for fingerprinting
            routing_decisions: Dictionary containing routing information
            gating_model_path: Path to gating model used (for versioning)
        """
        cache_key = (dataset_name, len(x_data))
        
        # Add metadata to routing decisions
        import time
        routing_decisions['cached_timestamp'] = str(time.time())
        routing_decisions['gating_model_path'] = gating_model_path
        routing_decisions['sample_count'] = len(x_data)
        
        # Store routing decisions and fingerprint
        self.routing_cache[cache_key] = routing_decisions
        self.sample_fingerprints[dataset_name] = self._compute_sample_fingerprint(x_data)
        
        # Save to disk
        self._save_cache()
        
        print(f"   Cached routing decisions for {dataset_name} ({len(x_data)} samples)")
    
    def compute_and_cache_routing(self,
                                dataset_name: str,
                                x_data: np.ndarray,
                                gating_model: torch.nn.Module,
                                eval_transforms,
                                batch_size: int = 64) -> Dict:
        """
        Compute routing decisions using gating network and cache them.
        
        Args:
            dataset_name: Name of the dataset
            x_data: Input data
            gating_model: Trained gating network
            eval_transforms: Normalization transforms
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing routing decisions
        """
        print(f"   Computing gating network routing for {dataset_name}...")
        
        gating_model.eval()
        all_shard_assignments = []
        all_confidences = []
        all_gating_probs = []
        
        with torch.no_grad():
            for i in range(0, len(x_data), batch_size):
                batch_x = torch.from_numpy(x_data[i:i+batch_size]).float()
                batch_x_normalized = eval_transforms(batch_x).to(DEVICE)
                
                # Get gating network predictions
                gating_logits = gating_model(batch_x_normalized)
                gating_probs = torch.softmax(gating_logits, dim=1)
                
                # Extract routing decisions
                shard_assignments = torch.argmax(gating_probs, dim=1).cpu().numpy()
                confidences = torch.max(gating_probs, dim=1)[0].cpu().numpy()
                
                all_shard_assignments.extend(shard_assignments)
                all_confidences.extend(confidences)
                all_gating_probs.extend(gating_probs.cpu().numpy())
        
        # Create routing decisions dictionary (convert numpy arrays to lists for JSON serialization)
        routing_decisions = {
            'shard_assignments': [int(x) for x in all_shard_assignments],
            'confidences': [float(x) for x in all_confidences],
            'gating_probabilities': [x.tolist() if hasattr(x, 'tolist') else x for x in all_gating_probs],
            'num_shards': int(gating_probs.shape[1]),
            'dataset_size': int(len(x_data))
        }
        
        # Cache the results
        self.cache_routing_decisions(dataset_name, x_data, routing_decisions)
        
        return routing_decisions
    
    def get_or_compute_routing(self,
                             dataset_name: str,
                             x_data: np.ndarray,
                             gating_model: torch.nn.Module,
                             eval_transforms,
                             batch_size: int = 64) -> Dict:
        """
        Get cached routing decisions or compute them if not available.
        
        This is the main interface method that should be used.
        """
        # Try to get cached routing first
        cached_routing = self.get_cached_routing(dataset_name, x_data)
        
        if cached_routing is not None:
            return cached_routing
        
        # Compute and cache if not available
        return self.compute_and_cache_routing(
            dataset_name, x_data, gating_model, eval_transforms, batch_size
        )
    
    def invalidate_cache(self, dataset_name: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            dataset_name: Specific dataset to invalidate, or None for all
        """
        if dataset_name is None:
            # Clear all cache
            self.routing_cache.clear()
            self.sample_fingerprints.clear()
            print("   Cleared all routing cache")
        else:
            # Clear specific dataset
            keys_to_remove = [k for k in self.routing_cache.keys() if k[0] == dataset_name]
            for key in keys_to_remove:
                del self.routing_cache[key]
            
            if dataset_name in self.sample_fingerprints:
                del self.sample_fingerprints[dataset_name]
            
            print(f"   Cleared routing cache for {dataset_name}")
        
        self._save_cache()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the current cache."""
        total_samples = sum(entry['dataset_size'] for entry in self.routing_cache.values())
        
        return {
            'cached_datasets': len(set(k[0] for k in self.routing_cache.keys())),
            'cached_entries': len(self.routing_cache),
            'total_cached_samples': total_samples,
            'cache_directory': self.cache_dir
        }


def create_cached_sisa_batch_runner(routing_cache: RoutingCache):
    """
    Create a cached version of _run_sisa_batch that uses pre-computed routing.
    
    Returns:
        Function that can replace _run_sisa_batch for cached inference
    """
    
    def _run_cached_sisa_batch(
        batch_x_normalized: torch.Tensor,
        shard_models: List[torch.nn.Module],
        routing_decisions: Dict,
        batch_start_idx: int,
        class_names: List[str],
        threshold: Optional[float] = None,
    ):
        """Execute SISA inference using pre-cached routing decisions."""
        
        batch_size = batch_x_normalized.size(0)
        num_classes = len(class_names)
        
        final_preds = torch.empty(batch_size, dtype=torch.long, device=DEVICE)
        combined_probabilities = torch.empty((batch_size, num_classes), dtype=torch.float32, device=DEVICE)
        
        # Extract cached routing information
        shard_assignments = routing_decisions['shard_assignments']
        confidences = routing_decisions['confidences']
        
        for idx in range(batch_size):
            sample_idx = batch_start_idx + idx
            
            # Use cached routing decision
            chosen_shard_idx = shard_assignments[sample_idx]
            confidence = confidences[sample_idx]
            
            # Process through chosen shard (same as original)
            single_sample = batch_x_normalized[idx:idx+1]
            if shard_models[chosen_shard_idx] is not None:
                with torch.no_grad():
                    specialist_logits = shard_models[chosen_shard_idx](single_sample)
                    specialist_probs = torch.softmax(specialist_logits, dim=1)[0]
            else:
                specialist_probs = torch.full((num_classes,), 1.0 / num_classes, device=DEVICE)
            
            # Apply temperature scaling (same as original)
            from plots import _apply_temperature_tensor
            chosen_probs = _apply_temperature_tensor(specialist_probs.unsqueeze(0), config.PRIMARY_SPECIALIST_TEMPERATURE)[0]
            final_pred = torch.argmax(chosen_probs).item()
            
            # Apply confidence thresholding if specified
            if threshold is not None and confidence < threshold:
                final_pred = -1
            
            final_preds[idx] = final_pred
            combined_probabilities[idx] = chosen_probs
        
        return final_preds, combined_probabilities
    
    return _run_cached_sisa_batch