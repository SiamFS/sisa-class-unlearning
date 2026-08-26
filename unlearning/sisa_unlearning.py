import sys
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import global configuration
import config
from utils.seeding import set_seed

set_seed(config.SEED)

from training.train_model import (
    train_model,
)
from plots import (
    create_training_visualizations,
    create_confusion_matrix,
    create_shard_confusion_matrix,
    create_overall_sisa_confusion_matrix,
    create_overall_sisa_roc_curve,
    create_overall_sisa_training_curves,
    create_accuracy_comparison_chart,
    create_time_comparison_chart,
    create_pure_training_time_chart,
    create_classification_metrics_comparison_chart,
    _run_sisa_batch,
    _normalize_probabilities_tensor,
    _apply_temperature_tensor,
    load_shard_class_indices,
)
from training.create_model import save_model_pytorch, load_model_pytorch, create_sisa_model, DEVICE
from training.replay_buffer import add_to_replay_buffer, compute_replay_ratio
from training.train_gating_model import train_gating
from utils.run_logging import setup_run_logging

class SISAUnlearning:
    def __init__(self, project_name: str, model_name: str):
        print(f"Initializing SISA Unlearning for project: '{project_name}' with model: '{model_name}'")
        self.project_name = project_name
        self.model_name = model_name
        self.base_dir = os.path.join(config.PROJECTS_DIR, self.project_name)
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "sisa_data")
        self.reports_dir = os.path.join(self.base_dir, "data_info")
        self.test_data_dir = os.path.join(self.data_dir, "test_data")
        
        self.metadata = self._load_metadata()
        self.class_names = self.metadata.get('class_names', [])
        self.num_shards = self.metadata.get('num_shards', 0)
        # W26: shards may hold different slice counts (S_k = |C_k| under
        # NUM_SLICES_PER_SHARD='auto'). `num_slices` is kept as the MAXIMUM so every
        # existing `range(self.num_slices)` loop still covers the longest shard --
        # `_load_slice_data` returns None for a slice a shorter shard doesn't have, and
        # every one of those loops already guards on that. Only genuinely per-shard
        # questions ("is this the last slice?") need `slices_for_shard`.
        self.slices_per_shard = self.metadata.get('slices_per_shard') or []
        if self.slices_per_shard:
            self.num_slices = max(self.slices_per_shard)
        else:
            self.num_slices = self.metadata.get('num_slices', 0)

        self.validation_data = self._load_validation_data()
        self.forgotten_samples_x = None
        self.forgotten_samples_y = None
        self.unlearning_histories = []  # Store training histories during unlearning
        
        # Cache unlearned classes from metadata for O(1) access
        self._unlearned_classes_cache = None
        self._unlearned_classes_set_cache = None

        # Load normalization stats from metadata (required)
        self.dataset_mean = self.metadata['normalization_mean']
        self.dataset_std = self.metadata['normalization_std']
        print("Loaded normalization stats from metadata.")

        self.eval_transforms = T.Compose([
            T.Normalize(self.dataset_mean, self.dataset_std)
        ])

    def slices_for_shard(self, shard_idx: int) -> int:
        """Slice count for one shard (W26); falls back to the global count.

        `self.num_slices` is the MAXIMUM across shards, which keeps every existing
        `range(self.num_slices)` loop correct (a slice a shorter shard doesn't have
        simply loads as None, and those loops already guard on that). Only genuinely
        per-shard questions -- "is this the last slice?" -- need this.
        """
        if self.slices_per_shard and shard_idx < len(self.slices_per_shard):
            return self.slices_per_shard[shard_idx]
        return self.num_slices

    def _load_metadata(self) -> Dict:
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _load_validation_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x_val_path = os.path.join(self.data_dir, "validation_data/x_validation.npy")
        y_val_path = os.path.join(self.data_dir, "validation_data/y_validation.npy")
        if os.path.exists(x_val_path) and os.path.exists(y_val_path):
            print("Global validation data loaded.")
            return (np.load(x_val_path), np.load(y_val_path))
        return None

    def _load_slice_data(self, shard_idx: int, slice_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        shard_path = os.path.join(self.data_dir, f"shards/shard_{shard_idx+1}")
        x_path = os.path.join(shard_path, f"slice_{slice_idx}_x.npy")
        y_path = os.path.join(shard_path, f"slice_{slice_idx}_y.npy")
        if os.path.exists(x_path):
            X, y = np.load(x_path), np.load(y_path)
            if X.size == 0: return None, None
            return X, y
        return None, None

    def _save_slice_data(self, shard_idx: int, slice_idx: int, X: np.ndarray, y: np.ndarray):
        shard_path = os.path.join(self.data_dir, f"shards/shard_{shard_idx+1}")
        os.makedirs(shard_path, exist_ok=True)
        np.save(os.path.join(shard_path, f"slice_{slice_idx}_x.npy"), X)
        np.save(os.path.join(shard_path, f"slice_{slice_idx}_y.npy"), y)

    def _save_forgotten_class_samples(self, class_to_unlearn: int, num_samples=16):
        # Initialize if first call (single unlearn), otherwise append (batch unlearn)
        if not hasattr(self, 'forgotten_samples_x') or self.forgotten_samples_x is None:
            self.forgotten_samples_x, self.forgotten_samples_y = [], []
        
        print(f"\nStep 0: Saving {num_samples} samples of '{self.class_names[class_to_unlearn]}' for final verification...")
        
        samples_x_temp = []
        samples_y_temp = []
        
        for shard_idx in range(self.num_shards):
            for slice_idx in range(self.num_slices):
                if len(samples_y_temp) >= num_samples: break
                X, y = self._load_slice_data(shard_idx, slice_idx)
                if X is None: continue
                
                mask = (y == class_to_unlearn)
                x_class, y_class = X[mask], y[mask]
                
                samples_to_take = min(len(x_class), num_samples - len(samples_y_temp))
                if samples_to_take > 0:
                    samples_x_temp.append(x_class[:samples_to_take])
                    samples_y_temp.extend(y_class[:samples_to_take])
            if len(samples_y_temp) >= num_samples: break
        
        # Append to existing samples (for batch unlearning)
        if samples_x_temp:
            if isinstance(self.forgotten_samples_x, np.ndarray):
                # Convert back to list if already concatenated from previous class
                self.forgotten_samples_x = [self.forgotten_samples_x]
            self.forgotten_samples_x.append(np.concatenate(samples_x_temp))
            self.forgotten_samples_y.extend(samples_y_temp)
            print(f"   - Saved {len(samples_y_temp)} samples for verification. Total samples: {len(self.forgotten_samples_y)}")

    def _update_shard_metadata(self, shard_idx: int):
        all_y = []
        for slice_idx in range(self.num_slices):
            _, y_slice = self._load_slice_data(shard_idx, slice_idx)
            if y_slice is not None:
                all_y.append(y_slice)
        
        if not all_y:
            print(f"   - Warning: Shard {shard_idx+1} is completely empty.")
            return

        y_combined = np.concatenate(all_y)
        unique_classes = np.unique(y_combined)
        
        metadata_path = os.path.join(self.data_dir, f"shards/shard_{shard_idx+1}/metadata.json")
        with open(metadata_path, 'r') as f:
            shard_metadata = json.load(f)
            
        shard_metadata['class_indices_present'] = [int(c) for c in unique_classes]
        shard_metadata['class_names_present'] = [self.class_names[c] for c in unique_classes]
        
        with open(metadata_path, 'w') as f:
            json.dump(shard_metadata, f, indent=2)
        print(f"   - Updated metadata for Shard {shard_idx+1}. New active classes: {shard_metadata['class_names_present']}")

    def get_unlearned_classes(self) -> List[int]:
        """Get the list of unlearned class indices with optimized metadata caching"""
        # Use cached version if available (O(1) access after first load)
        if self._unlearned_classes_cache is not None:
            return self._unlearned_classes_cache
        
        # Load from main metadata
        unlearned_classes = self.metadata.get('unlearned_classes', [])
        self._unlearned_classes_cache = unlearned_classes
        self._unlearned_classes_set_cache = set(unlearned_classes)
        return unlearned_classes

    def get_unlearned_classes_set(self) -> set:
        """Get unlearned classes as a set for O(1) lookup during post-processing"""
        if self._unlearned_classes_set_cache is not None:
            return self._unlearned_classes_set_cache
        
        # This will populate both caches
        self.get_unlearned_classes()
        return self._unlearned_classes_set_cache or set()
    
    def _update_metadata_with_unlearned_classes(self, unlearned_classes: List[int]):
        """Update main metadata with unlearned classes for centralized storage"""
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        
        try:
            # Load current metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Update with unlearned classes
            metadata['unlearned_classes'] = unlearned_classes
            metadata['unlearning_timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update our cached copy
            self.metadata['unlearned_classes'] = unlearned_classes
            self._unlearned_classes_cache = unlearned_classes
            self._unlearned_classes_set_cache = set(unlearned_classes)
            
            print(f"   - Updated metadata with unlearned classes: {[self.class_names[i] for i in unlearned_classes]}")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"   Warning: Could not update metadata: {e}")

    def _invalidate_unlearned_classes_cache(self):
        """Invalidate cache when unlearned classes change"""
        self._unlearned_classes_cache = None
        self._unlearned_classes_set_cache = None
    
    def _filter_replay_buffer_for_unlearning(self, replay_buffer, deleted_classes: set):
        """
        CRITICAL: Remove ALL samples of deleted classes from replay buffer.
        This ensures deleted class data NEVER re-enters training.

        Args:
            replay_buffer: dict of {class_idx: {'X':.., 'y':..}}
            deleted_classes: Set of class indices to remove

        Returns:
            Cleaned replay buffer
        """
        if not replay_buffer:
            return replay_buffer

        print(f"\n   🔍 FILTERING REPLAY BUFFER: Removing deleted classes {deleted_classes}")

        original_classes = set(replay_buffer.keys())
        classes_to_remove = original_classes & deleted_classes

        if classes_to_remove:
            for class_idx in classes_to_remove:
                sample_count = len(replay_buffer[class_idx]['y'])
                del replay_buffer[class_idx]
                print(f"   ✓ Removed {sample_count} samples of deleted class {self.class_names[class_idx]} from replay buffer")

            remaining_classes = set(replay_buffer.keys())
            print(f"   ✓ Replay buffer cleaned: {len(remaining_classes)} classes remain")
        else:
            print(f"   ✓ Replay buffer already clean (no deleted classes found)")

        return replay_buffer

    def _load_shard_metadatas(self) -> List[Dict]:
        """Load metadata for all shards"""
        shard_metadatas = []
        for shard_idx in range(self.num_shards):
            metadata_path = os.path.join(self.data_dir, f"shards/shard_{shard_idx+1}/metadata.json")
            with open(metadata_path, 'r') as f:
                shard_metadata = json.load(f)
            shard_metadatas.append(shard_metadata)
        return shard_metadatas

    def _track_unlearned_class(self, unlearned_class_idx):
        """Track unlearned class and update metadata"""
        print(f"\n   - Tracking unlearned class: {self.class_names[unlearned_class_idx]}")
        
        # Get current unlearned classes and add the new one
        unlearned_classes = self.metadata.get('unlearned_classes', [])
        if unlearned_class_idx not in unlearned_classes:
            unlearned_classes.append(unlearned_class_idx)
        
        # Update metadata with unlearned classes
        self._update_metadata_with_unlearned_classes(unlearned_classes)
        
        print(f"   - Updated unlearned classes list: {[self.class_names[i] for i in unlearned_classes]}")

    def unlearn_by_class(self, class_name: str):
        if class_name not in self.class_names:
            raise ValueError(f"Class '{class_name}' not found.")
        class_idx = self.class_names.index(class_name)
        overall_start_time = time.time()
        
        # Initialize timing tracking
        self.data_removal_time = 0
        self.retraining_time = 0

        # Backup test set before any modifications
        self._backup_test_set_if_needed()
        
        self._save_forgotten_class_samples(class_idx)
        
        # Display training baseline metrics at the start of unlearning
        print("\n" + "=" * 60)
        print("TRAINING BASELINE METRICS (Before Unlearning)")
        print("=" * 60)
        training_accuracy, gating_training_time, base_model_time, pure_training_time, training_report = self._get_training_metrics()
        print(f"Training Accuracy: {training_accuracy:.4f}")
        print(f"Gating Network Training Time: {gating_training_time:.2f} seconds")
        print(f"Base Model Training Time (with eval): {base_model_time:.2f} seconds")
        print(f"Base Model Training Time (pure): {pure_training_time:.2f} seconds")
        print(f"Total Training Time: {(gating_training_time + base_model_time):.2f} seconds")
        print("=" * 60)
        
        print(f"\nStep 1: Removing data for class '{class_name}'...")
        # Start timing data removal
        data_removal_start = time.time()
        affected_shards = {}
        
        # Track slice states and affected slices per shard
        for shard_idx in range(self.num_shards):
            slice_states = {}  # Track state of each slice: 'empty', 'has_data', 'deleted'
            affected_slices = []  # List of all slices that were affected
            first_affected_slice = -1
            total_removed = 0
            completely_deleted_slices = 0
            
            for slice_idx in range(self.num_slices):
                removed_count, remaining_count = self._remove_data_from_slice(shard_idx, slice_idx, class_idx)
                
                if removed_count > 0:
                    total_removed += removed_count
                    affected_slices.append(slice_idx)
                    
                    if first_affected_slice == -1:
                        first_affected_slice = slice_idx
                    
                    # Check if slice is now completely empty
                    if remaining_count == 0:
                        slice_states[slice_idx] = 'deleted'
                        completely_deleted_slices += 1
                        # Delete empty slice data and model files
                        self._delete_empty_slice_files(shard_idx, slice_idx)
                    else:
                        slice_states[slice_idx] = 'has_data'
                else:
                    # Slice was not affected - check if it has data
                    X, y = self._load_slice_data(shard_idx, slice_idx)
                    if X is None or len(X) == 0:
                        slice_states[slice_idx] = 'empty'
                    else:
                        slice_states[slice_idx] = 'has_data'
            
            if first_affected_slice != -1:
                affected_shards[shard_idx] = {
                    'first_affected_slice': first_affected_slice,
                    'affected_slices': affected_slices,
                    'slice_states': slice_states,
                    'total_removed': total_removed,
                    'deleted_slices': completely_deleted_slices
                }
                
                print(f"\n   Summary for Shard {shard_idx+1}:")
                print(f"   - Total samples removed: {total_removed}")
                print(f"   - Slices affected: {[s+1 for s in affected_slices]}")
                print(f"   - Slices completely deleted (empty): {completely_deleted_slices}")
                
                self._update_shard_metadata(shard_idx)

        # Capture data removal time
        self.data_removal_time = time.time() - data_removal_start
        print(f"Data Removal Time: {self.data_removal_time:.2f} seconds")
        
        # CRITICAL CHECK: Ensure at least one shard still has training data
        if affected_shards:
            has_active_shard = False
            for shard_idx, shard_info in affected_shards.items():
                # Check if this shard has at least one slice with data
                has_data_in_shard = any(
                    state == 'has_data' 
                    for state in shard_info['slice_states'].values()
                )
                if has_data_in_shard:
                    has_active_shard = True
                    break
            
            if not has_active_shard:
                raise ValueError(
                    f"\n{'='*60}\n"
                    f"CRITICAL ERROR: Cannot delete class '{class_name}'\n"
                    f"{'='*60}\n"
                    f"This operation would delete ALL training data from the system.\n"
                    f"At least one shard must contain training data for SISA to function.\n\n"
                    f"Current state after deletion:\n"
                    f"  - All {self.num_shards} shard(s) would have 0 training slices\n"
                    f"  - Total slices deleted: {sum(info['deleted_slices'] for info in affected_shards.values())}\n\n"
                    f"Recommendation: This class cannot be unlearned without breaking the system.\n"
                    f"{'='*60}"
                )
        
        if not affected_shards:
            print(f"No data found for class '{class_name}' in training data.")
            print("However, still proceeding to remove the class from test set...")
            total_retrain_time = 0.0
            shard_models = None
            classified_accuracy = 0.0
            overall_accuracy = 0.0
            
            # Initialize timing variables for consistency
            self.retraining_time = 0.0
            
            # Skip model evaluation since no training was done
            total_time = time.time() - overall_start_time
            print("\n" + "=" * 60)
            print("SISA Unlearning Completed")
            print("=" * 60)
            print(f"Unlearned Class: '{class_name}'")
            # Calculate pure unlearning time for consistency
            pure_unlearning_time = self.data_removal_time + self.retraining_time
            
            print("\nTIMING BREAKDOWN:")
            print(f"  - Data Removal Time: {self.data_removal_time:.2f} seconds")
            print(f"  - Retraining Time Only: {self.retraining_time:.2f} seconds")
            print(f"  - Pure Unlearning Time: {pure_unlearning_time:.2f} seconds (removal + retraining)")
            print(f"  - Total Process Time: {total_time:.2f} seconds")
            print("\nNote: No retraining was needed as class was not in training data.")
            print(f"Output saved to: {self.base_dir}")
            print("=" * 60)
            
            # MODIFICATION: Preserving test samples - NOT removing class from test set
            print("   - Test set preserved: Original test samples maintained for evaluation purposes.")
            return

        total_retrain_time = 0.0
        total_pure_retrain_time = 0.0  # NEW: Track pure training time
        print("\nStep 2: Retraining affected shards incrementally...")
        retraining_start = time.time()
        for shard_idx, info in affected_shards.items():
            retrain_time, pure_retrain_time = self._retrain_shard_incrementally(
                shard_idx,
                info['first_affected_slice'],
                info['slice_states']  # Pass slice states for proper gap handling
            )
            total_retrain_time += retrain_time
            total_pure_retrain_time += pure_retrain_time  # NEW: Accumulate pure training time
        
        # Capture both retraining times
        self.retraining_time = time.time() - retraining_start
        self.pure_retraining_time = total_pure_retrain_time  # NEW: Store pure training time
        print(f"Retraining Time (with eval): {self.retraining_time:.2f} seconds")
        print(f"Retraining Time (pure): {self.pure_retraining_time:.2f} seconds")
        
        # Step 3: Track unlearned class. The gating router (W16) IS retrained
        # here to exclude it -- see final_evaluation_with_self_routing(), called
        # next, which retrains the gate on the updated unlearned-classes list
        # before evaluating.
        self._track_unlearned_class(class_idx)
        
        # Calculate PURE unlearning time (data removal + retraining only, no evaluation/plotting overhead)
        pure_unlearning_time = self.data_removal_time + self.retraining_time
        
        # Step 4: Final evaluation with self-routing
        evaluation_start_time = time.time()
        shard_models, classified_accuracy, overall_accuracy, unlearning_report = self.final_evaluation_with_self_routing()
        evaluation_time = time.time() - evaluation_start_time

        total_time = time.time() - overall_start_time
        
        # Create comparison visualizations using PURE unlearning time (no evaluation/plotting overhead)
        self._create_comparison_visualizations(overall_accuracy, pure_unlearning_time, total_retrain_time, class_name, 
                                            training_report=training_report, unlearning_report=unlearning_report)
        
        # COMPREHENSIVE EVALUATION BEFORE DELETING TEST DATA
        if shard_models:
            self._evaluate_on_forgotten_samples(shard_models)
            # NEW: Comprehensive deleted class accuracy evaluation with bar chart (BEFORE deleting test samples)
            self._evaluate_deleted_class_accuracy(shard_models, class_name, class_idx)
        
        print("\n" + "=" * 60)
        print("SISA Unlearning Completed")
        print("=" * 60)
        print(f"Unlearned Class: '{class_name}'")
        print("\nTIMING BREAKDOWN:")
        print(f"  - Data Removal Time: {self.data_removal_time:.2f} seconds")
        print(f"  - Retraining Time Only: {self.retraining_time:.2f} seconds")
        print(f"  - Gating Retrain Time: {getattr(self, 'gating_retrain_time', 0.0):.2f} seconds")
        print(f"  - Pure Unlearning Time: {pure_unlearning_time:.2f} seconds (removal + retraining)")
        print(f"  - Evaluation & Overhead: {total_time - pure_unlearning_time:.2f} seconds")
        print(f"  - Total Process Time: {total_time:.2f} seconds")
        print("\nACCURACY RESULTS:")
        print(f"  - Final Accuracy on Classified Samples: {classified_accuracy:.4f}")
        print(f"  - Overall Model Accuracy on All Test Samples: {overall_accuracy:.4f}")
        print(f"\nOutput saved to: {self.base_dir}")
        print("=" * 60)

        print("   - Test set preserved: Original test samples maintained for evaluation purposes.")

    def batch_unlearn_by_classes(self, class_names: List[str]):
  
        print("\n" + "="*80)
        print(f"BATCH UNLEARNING: {len(class_names)} CLASSES")
        print("="*80)
        print(f"Classes to unlearn: {class_names}")
        print("="*80)
        
        # Validate all class names first
        class_indices = []
        for class_name in class_names:
            if class_name not in self.class_names:
                raise ValueError(f"Class '{class_name}' not found in dataset")
            class_indices.append(self.class_names.index(class_name))
        
        print(f"Class indices to unlearn: {class_indices}")
        
        overall_start_time = time.time()
        
        # Initialize timing tracking
        self.data_removal_time = 0
        self.retraining_time = 0

        # Initialize forgotten samples storage for batch unlearning
        self.forgotten_samples_x = []
        self.forgotten_samples_y = []

        # Backup test set before any modifications
        self._backup_test_set_if_needed()
        
        # Save forgotten samples for all classes
        print("\n" + "="*60)
        print("Step 0: Saving sample images from classes to be forgotten...")
        print("="*60)
        for class_idx, class_name in zip(class_indices, class_names):
            self._save_forgotten_class_samples(class_idx, num_samples=16)  # Save 16 samples per class for verification grid
        
        # Display training baseline metrics
        print("\n" + "=" * 60)
        print("TRAINING BASELINE METRICS (Before Unlearning)")
        print("=" * 60)
        training_accuracy, gating_training_time, base_model_time, pure_training_time, training_report = self._get_training_metrics()
        print(f"Training Accuracy: {training_accuracy:.4f}")
        print(f"Gating Network Training Time: {gating_training_time:.2f} seconds")
        print(f"Base Model Training Time (with eval): {base_model_time:.2f} seconds")
        print(f"Base Model Training Time (pure): {pure_training_time:.2f} seconds")
        print(f"Total Training Time: {(gating_training_time + base_model_time):.2f} seconds")
        print("=" * 60)
        
        # Step 1: Remove data for ALL classes
        print("\n" + "="*60)
        print(f"Step 1: Removing data for {len(class_names)} classes...")
        print("="*60)
        
        data_removal_start = time.time()
        affected_shards = {}  # {shard_idx: {'first_affected_slice': int, 'slice_states': dict, ...}}
        
        # Process each shard
        for shard_idx in range(self.num_shards):
            print(f"\n--- Processing Shard {shard_idx+1} ---")
            slice_states = {}  # {slice_idx: 'empty'|'has_data'|'deleted'}
            affected_slices = set()  # Track which slices were affected
            first_affected_slice = None
            total_removed = 0
            completely_deleted_slices = 0
            
            # Initialize slice states by checking each slice
            for slice_idx in range(self.num_slices):
                X, y = self._load_slice_data(shard_idx, slice_idx)
                if X is None or len(X) == 0:
                    slice_states[slice_idx] = 'empty'
                else:
                    slice_states[slice_idx] = 'has_data'
            
            # Remove ALL target classes from ALL slices
            for slice_idx in range(self.num_slices):
                slice_removed_count = 0
                remaining_count = 0
                
                # Remove all target classes from this slice
                for class_idx in class_indices:
                    removed, remaining = self._remove_data_from_slice(shard_idx, slice_idx, class_idx)
                    slice_removed_count += removed
                    remaining_count = remaining  # Last remaining count
                
                if slice_removed_count > 0:
                    total_removed += slice_removed_count
                    affected_slices.add(slice_idx)
                    
                    # Track first affected slice
                    if first_affected_slice is None:
                        first_affected_slice = slice_idx
                    
                    # Check if slice is now completely empty after removing all classes
                    if remaining_count == 0:
                        slice_states[slice_idx] = 'deleted'
                        completely_deleted_slices += 1
                        # Delete empty slice files
                        self._delete_empty_slice_files(shard_idx, slice_idx)
                        print(f"   Slice {slice_idx+1}: Completely deleted (empty)")
                    else:
                        slice_states[slice_idx] = 'has_data'
                        print(f"   Slice {slice_idx+1}: Removed {slice_removed_count} samples, {remaining_count} remaining")
            
            # If this shard was affected, store its information
            if first_affected_slice is not None:
                affected_shards[shard_idx] = {
                    'first_affected_slice': first_affected_slice,
                    'affected_slices': sorted(list(affected_slices)),
                    'slice_states': slice_states,
                    'total_removed': total_removed,
                    'deleted_slices': completely_deleted_slices
                }
                
                print(f"\n   Summary for Shard {shard_idx+1}:")
                print(f"   - Total samples removed: {total_removed}")
                print(f"   - Slices affected: {[s+1 for s in sorted(affected_slices)]}")
                print(f"   - Slices completely deleted: {completely_deleted_slices}")
                print(f"   - First affected slice: {first_affected_slice+1}")
                
                # Show slice state summary
                print(f"   - Slice states:")
                for s_idx in range(self.num_slices):
                    state = slice_states.get(s_idx, 'unknown')
                    print(f"      Slice {s_idx+1}: {state}")
                
                self._update_shard_metadata(shard_idx)
        
        self.data_removal_time = time.time() - data_removal_start
        print(f"\nTotal Data Removal Time: {self.data_removal_time:.2f} seconds")
        
        # Validate that at least one shard has training data
        if affected_shards:
            has_active_shard = False
            for shard_idx, shard_info in affected_shards.items():
                has_data_in_shard = any(
                    state == 'has_data' 
                    for state in shard_info['slice_states'].values()
                )
                if has_data_in_shard:
                    has_active_shard = True
                    break
            
            if not has_active_shard:
                raise ValueError(
                    f"\n{'='*60}\n"
                    f"CRITICAL ERROR: Cannot delete classes {class_names}\n"
                    f"{'='*60}\n"
                    f"This operation would delete ALL training data from the system.\n"
                    f"At least one shard must contain training data for SISA to function.\n"
                    f"{'='*60}"
                )
        
        if not affected_shards:
            print(f"No data found for classes {class_names} in training data.")
            print("Unlearning completed (no retraining needed).")
            return
        
        # Step 2: Retrain affected shards
        print("\n" + "="*60)
        print("Step 2: Retraining affected shards incrementally...")
        print("="*60)
        
        retraining_start = time.time()
        total_retrain_time = 0.0
        total_pure_retrain_time = 0.0
        
        for shard_idx, info in affected_shards.items():
            print(f"\n{'='*70}")
            print(f"RETRAINING SHARD {shard_idx+1}")
            print(f"{'='*70}")
            print(f"First affected slice: {info['first_affected_slice']+1}")
            print(f"Slice states: {info['slice_states']}")

            retrain_time, pure_retrain_time = self._retrain_shard_incrementally(
                shard_idx,
                info['first_affected_slice'],
                info['slice_states']  # Pass slice states for proper gap handling
            )
            total_retrain_time += retrain_time
            total_pure_retrain_time += pure_retrain_time
        
        self.retraining_time = time.time() - retraining_start
        self.pure_retraining_time = total_pure_retrain_time
        print(f"\nTotal Retraining Time (with eval): {self.retraining_time:.2f} seconds")
        print(f"Total Retraining Time (pure): {self.pure_retraining_time:.2f} seconds")
        
        # Step 3: Track all unlearned classes
        print("\n" + "="*60)
        print("Step 3: Updating metadata with unlearned classes...")
        print("="*60)
        for class_idx in class_indices:
            self._track_unlearned_class(class_idx)
        
        # Calculate pure unlearning time
        pure_unlearning_time = self.data_removal_time + self.retraining_time
        
        # Step 4: Final evaluation
        print("\n" + "="*60)
        print("Step 4: Final evaluation with self-routing...")
        print("="*60)
        evaluation_start_time = time.time()
        shard_models, classified_accuracy, overall_accuracy, unlearning_report = self.final_evaluation_with_self_routing()
        evaluation_time = time.time() - evaluation_start_time
        
        total_time = time.time() - overall_start_time
        
        # Create comparison visualizations (use first class name for charts)
        self._create_comparison_visualizations(
            overall_accuracy, pure_unlearning_time, total_retrain_time, 
            f"batch_{len(class_names)}_classes",
            training_report=training_report, 
            unlearning_report=unlearning_report
        )
        
        # Evaluate forgotten samples
        if shard_models:
            self._evaluate_on_forgotten_samples(shard_models)
            # Evaluate each deleted class
            for class_idx, class_name in zip(class_indices, class_names):
                self._evaluate_deleted_class_accuracy(shard_models, class_name, class_idx)
        
        # Final summary
        print("\n" + "=" * 80)
        print("BATCH UNLEARNING COMPLETED")
        print("=" * 80)
        print(f"Unlearned Classes: {class_names}")
        print(f"Total Classes Unlearned: {len(class_names)}")
        print("\nTIMING BREAKDOWN:")
        print(f"  - Data Removal Time: {self.data_removal_time:.2f} seconds")
        print(f"  - Retraining Time Only: {self.retraining_time:.2f} seconds")
        print(f"  - Gating Retrain Time: {getattr(self, 'gating_retrain_time', 0.0):.2f} seconds")
        print(f"  - Pure Unlearning Time: {pure_unlearning_time:.2f} seconds (removal + retraining)")
        print(f"  - Evaluation & Overhead: {total_time - pure_unlearning_time:.2f} seconds")
        print(f"  - Total Process Time: {total_time:.2f} seconds")
        print("\nACCURACY RESULTS:")
        print(f"  - Final Accuracy on Classified Samples: {classified_accuracy:.4f}")
        print(f"  - Overall Model Accuracy on All Test Samples: {overall_accuracy:.4f}")
        print(f"\nAffected Shards: {list(affected_shards.keys())}")
        print(f"Output saved to: {self.base_dir}")
        print("=" * 80)

    def _remove_data_from_slice(self, shard_idx: int, slice_idx: int, class_to_remove: int) -> Tuple[int, int]:
        """Remove class data from a slice and return (removed_count, remaining_count)"""
        X, y = self._load_slice_data(shard_idx, slice_idx)
        if X is None: 
            return 0, 0
        
        original_size = len(y)
        keep_mask = (y != class_to_remove)
        x_new, y_new = X[keep_mask], y[keep_mask]
        remaining_count = len(y_new)
        removed_count = original_size - remaining_count
        
        if removed_count > 0:
            if remaining_count == 0:
                print(f"   - Shard {shard_idx+1}, Slice {slice_idx+1}: Removed {removed_count} samples (slice now EMPTY - will be deleted)")
            else:
                print(f"   - Shard {shard_idx+1}, Slice {slice_idx+1}: Removed {removed_count} samples ({remaining_count} remain)")
                # VERIFICATION: Log remaining classes after deletion
                unique_remaining = np.unique(y_new)
                remaining_class_names = [self.class_names[c] for c in unique_remaining]
                print(f"      ✓ Remaining classes in slice: {remaining_class_names}")
                # Assert deleted class is NOT in remaining data
                if class_to_remove in unique_remaining:
                    raise ValueError(f"CRITICAL ERROR: Deleted class {self.class_names[class_to_remove]} still present in slice after removal!")
            # Save the modified data (even if empty, for consistency)
            self._save_slice_data(shard_idx, slice_idx, x_new, y_new)
        
        return removed_count, remaining_count
    
    def _delete_empty_slice_files(self, shard_idx: int, slice_idx: int):
        """Delete data and model files for an empty slice"""
        shard_path = os.path.join(self.data_dir, f"shards/shard_{shard_idx+1}")
        model_path = os.path.join(self.models_dir, f"shard_{shard_idx+1}")
        
        # Delete slice data files
        x_path = os.path.join(shard_path, f"slice_{slice_idx}_x.npy")
        y_path = os.path.join(shard_path, f"slice_{slice_idx}_y.npy")
        
        for path in [x_path, y_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"      - Deleted empty slice data: {os.path.basename(path)}")
        
        # Delete slice model file
        model_file = os.path.join(model_path, f"slice_{slice_idx}_model_{self.model_name}.pth")
        if os.path.exists(model_file):
            os.remove(model_file)
            print(f"      - Deleted model weights: slice_{slice_idx}_model_{self.model_name}.pth")

    def _resize_model_head(self, old_model, head_classes: List[int]):
        """W6 dynamic head: return a model whose output head is sized to len(head_classes).

        Copies every learned weight that doesn't depend on the class count (the
        convolutional trunk and the hidden FC layer) from `old_model`; only the
        final classification layer is freshly initialized, since its width must
        match the shard's new (post-unlearning) owned-class count. Applying this
        once, before retraining, means the deleted class ends up with literally
        no output neuron rather than a masked one.
        """
        target_num_classes = len(head_classes)
        old_state = old_model.state_dict()
        old_final_classes = old_state['fc_layer.5.weight'].shape[0]
        if old_final_classes == target_num_classes:
            return old_model.to(DEVICE)

        # W21: keep the head type of the model being resized. The transplant below is
        # shape-based, so a cosine head needs no special handling -- `fc_layer.5.weight`
        # changes width and is reinitialized, while the class-independent scale carries
        # over -- but the replacement model must be built with the same head kind.
        # W28: preserve the BACKBONE as well as the head type. Same discriminator the
        # loader uses -- SISAConvNet has fc_layer.1.weight, SISAResNet pads that slot
        # with a parameterless Identity.
        old_arch = 'convnet' if 'fc_layer.1.weight' in old_state else 'resnet'
        new_model = create_sisa_model(
            num_classes=target_num_classes,
            classifier_type=getattr(old_model, 'classifier_type', None),
            arch=old_arch,
        )
        new_state = new_model.state_dict()
        transplanted = {k: v for k, v in old_state.items() if k in new_state and new_state[k].shape == v.shape}
        new_state.update(transplanted)
        new_model.load_state_dict(new_state)
        skipped = sorted(set(old_state.keys()) - set(transplanted.keys()))
        print(f"   - Resized head: {old_final_classes} -> {target_num_classes} classes (reinitialized: {skipped})")
        return new_model.to(DEVICE)

    def _retrain_shard_incrementally(self, shard_idx: int, first_affected_slice: int, slice_states: Dict = None):
        """
        Retrain shard incrementally, handling empty/deleted slices properly.
        
        Args:
            shard_idx: Index of shard to retrain
            first_affected_slice: First slice that was affected by unlearning
            slice_states: Dictionary mapping slice_idx -> state ('empty', 'has_data', 'deleted')
        """
        retrain_start_time = time.time()
        pure_retrain_time = 0  # Track pure training time (excludes evaluation/plotting)
        print("\n" + "="*20 + f" Incremental Retraining for Shard {shard_idx+1} " + "="*20)
        
        # Check class balance and determine augmentation strategy for unlearning
        augmentation_config = self.check_class_balance_and_augmentation_unlearning(shard_idx, self.class_names)
        
        # Initialize histories collection for this shard
        shard_histories = []
        
        shard_model_dir = os.path.join(self.models_dir, f"shard_{shard_idx+1}")

        # W6: the shard's current (post-unlearning) owned classes define the retrained
        # model's head size. `_update_shard_metadata` already recomputed this in Step 1,
        # before this method is ever called.
        with open(os.path.join(self.data_dir, f"shards/shard_{shard_idx+1}/metadata.json"), 'r') as f:
            current_shard_meta = json.load(f)
        head_classes = sorted(current_shard_meta.get('class_indices_present', []))
        print(f"   - Dynamic head: retraining with {len(head_classes)} output classes (was {config.get_num_classes()})")

        # If no slice_states provided, create a default one
        if slice_states is None:
            slice_states = {}
            for slice_idx in range(self.num_slices):
                X, _ = self._load_slice_data(shard_idx, slice_idx)
                if X is None or len(X) == 0:
                    slice_states[slice_idx] = 'empty'
                else:
                    slice_states[slice_idx] = 'has_data'
        
        # Delete stale model files for affected slices (from first_affected_slice onwards)
        for slice_idx in range(first_affected_slice, self.num_slices):
            stale_model_path = os.path.join(shard_model_dir, f"slice_{slice_idx}_model_{self.model_name}.pth")
            if os.path.exists(stale_model_path):
                os.remove(stale_model_path)
                print(f"   - Deleted stale model: slice_{slice_idx}_model_{self.model_name}.pth")
        
        # Find the previous available model and replay buffer (handling gaps from deleted slices)
        current_model = None
        previous_available_slice = None
        
        # Search backwards from first_affected_slice-1 to find last available model
        for slice_idx in range(first_affected_slice - 1, -1, -1):
            if slice_states.get(slice_idx, 'empty') == 'has_data':
                model_path = os.path.join(shard_model_dir, f"slice_{slice_idx}_model_{self.model_name}.pth")
                if os.path.exists(model_path):
                    try:
                        base_model, _ = load_model_pytorch(model_path)
                        current_model = self._resize_model_head(base_model, head_classes)
                        previous_available_slice = slice_idx
                        print(f"   - Loaded base model from previous available Slice {slice_idx+1}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Warning: Failed to load model from Slice {slice_idx+1}: {e}")
                        print(f"   - Will try previous slice or create new model")
                        continue  # Try the next previous slice

        if current_model is None:
            print("   - No previous model found or all models corrupted, training from scratch")
        
        # Define reports_dir for use throughout the method
        reports_dir = os.path.join(self.base_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Get deleted classes for replay buffer filtering
        deleted_classes_set = self.get_unlearned_classes_set()
        print(f"\n   🚫 UNLEARNING MODE: Will filter deleted classes {deleted_classes_set} from replay buffer")
        
        # Initialize replay buffer from previous available slices (skipping empty/deleted ones),
        # capped per class (W4) so it doesn't grow unboundedly across many slices.
        replay_buffer = {}
        replay_seen_counts = {}  # per-class "samples ever seen" count, for reservoir sampling
        replay_rng = np.random.default_rng(config.SEED)
        for slice_idx in range(first_affected_slice):
            # Only process slices that have data
            if slice_states.get(slice_idx, 'empty') != 'has_data':
                continue

            x_s, y_s = self._load_slice_data(shard_idx, slice_idx)
            if x_s is not None and len(x_s) > 0:
                add_to_replay_buffer(replay_buffer, x_s, y_s, replay_rng,
                                      config.MAX_REPLAY_SAMPLES_PER_CLASS, replay_seen_counts)
                print(f"   - Added Slice {slice_idx+1} data to replay buffer")
        
        # CRITICAL: Filter out ALL deleted classes from replay buffer
        replay_buffer = self._filter_replay_buffer_for_unlearning(replay_buffer, deleted_classes_set)
        
        # VERIFICATION: Assert replay buffer is clean
        if replay_buffer:
            buffer_classes = set(replay_buffer.keys())

            contamination = buffer_classes & deleted_classes_set
            if contamination:
                contaminated_names = [self.class_names[c] for c in contamination]
                raise ValueError(f"CRITICAL ERROR: Replay buffer still contains deleted classes {contaminated_names}!")
            else:
                print(f"   ✓ VERIFIED: Replay buffer is clean (contains {len(buffer_classes)} classes, no deleted classes)")

        # Load shard metadata (for potential future use)
        with open(os.path.join(self.data_dir, f"shards/shard_{shard_idx+1}/metadata.json"), 'r') as f:
            json.load(f)  # Load but don't use currently

        # Train from first_affected_slice to last slice, handling gaps from deleted slices
        for slice_idx in range(first_affected_slice, self.num_slices):
            # Skip empty or deleted slices
            slice_state = slice_states.get(slice_idx, 'empty')
            if slice_state in ['empty', 'deleted']:
                print(f"   - Slice {slice_idx+1} is {slice_state}, skipping...")
                continue
            
            # Double-check that slice actually has data
            x_slice, y_slice = self._load_slice_data(shard_idx, slice_idx)
            if x_slice is None or len(x_slice) == 0: 
                print(f"   - Slice {slice_idx+1} is empty after unlearning, skipping...")
                continue

            print(f"\n   --- Retraining Slice {slice_idx+1} (Incremental Logic) ---")
            print(f"   - Current model state: {'Available' if current_model is not None else 'Training from scratch'}")
            print(f"   - Replay buffer status: {len(replay_buffer)} classes available")
            
            # Get cumulative classes up to this slice (excluding unlearned classes)
            cumulative_classes = self._get_cumulative_classes_up_to_slice_unlearning(shard_idx, slice_idx)
            active_classes = cumulative_classes  # Keep track for model training
            print(f"   - Cumulative classes for validation: {[self.class_names[c] for c in cumulative_classes]}")
            
            # Create incremental validation data (like original training)
            x_val_filtered, y_val_filtered = self._get_incremental_validation_data_unlearning(
                cumulative_classes, self.validation_data
            )
            x_slice, y_slice = self._load_slice_data(shard_idx, slice_idx)
            if x_slice is None: continue

            # W18: identical class-balanced replay rule as the original training loop.
            # These two paths MUST agree -- if unlearning retrains under a different
            # replay recipe than training used, the W8 scratch-reference comparison is
            # no longer apples-to-apples and the exactness proof does not hold.
            current_replay_ratio = compute_replay_ratio(replay_buffer, y_slice)
            print(f"   - Using replay ratio: {current_replay_ratio:.3f} "
                  f"({len(replay_buffer)} replay class(es), {len(np.unique(y_slice))} in slice)")

            # START: Track pure training time (before train_model call)
            slice_train_start = time.time()
            
            # Enhanced retraining parameters for better accuracy after unlearning
            current_model, history = train_model(
                x_slice, y_slice, model=current_model,
                epochs=config.MAX_EPOCHS,  # From global config
                batch_size=config.BATCH_SIZE,  # From global config
                lr=config.UNLEARNING_LEARNING_RATE,  # From global config
                validation_data=(x_val_filtered, y_val_filtered),  # Use incremental validation
                active_classes=active_classes,
                head_classes=head_classes,  # W6: dynamic head sized to the shard's post-unlearning classes
                replay_buffer=replay_buffer,
                replay_ratio=current_replay_ratio,  # Use dynamic ratio
                dataset_mean=self.dataset_mean,
                dataset_std=self.dataset_std,
                training_type='unlearning',  # Proper early stopping for unlearning
                augmentation_config=augmentation_config,  # Use balance-based augmentation for unlearning
                device=DEVICE
            )
            
            # END: Track pure training time (after train_model call)
            slice_train_end = time.time()
            pure_retrain_time += (slice_train_end - slice_train_start)
            
            # Store history for overall training curves
            if history is not None:
                shard_histories.append(history)
            
            # Generate visualizations for the last slice of unlearning
            if slice_idx == self.slices_for_shard(shard_idx) - 1:  # Last slice (W26: per shard)
                print(f"   - Generating unlearning visualizations for final slice of Shard {shard_idx+1}...")
                
                # Create training visualizations (loss/accuracy curves)
                if history is not None:
                    create_training_visualizations(
                        history, shard_idx, slice_idx, reports_dir, 'unlearning'
                    )
                
                # Use incremental validation data for confusion matrix visualization
                if current_model is not None and len(x_val_filtered) > 0:
                    create_shard_confusion_matrix(
                        current_model, x_val_filtered, y_val_filtered, self.class_names,
                        shard_idx, reports_dir, active_classes, head_classes=head_classes
                    )
            
            # Update replay buffer with retrained slice data
            # CRITICAL: Only add classes that are NOT deleted
            if len(x_slice) > 0:
                # First, filter out deleted classes from current slice
                mask_keep = np.array([label not in deleted_classes_set for label in y_slice])
                x_slice_filtered = x_slice[mask_keep]
                y_slice_filtered = y_slice[mask_keep]
                
                if len(y_slice_filtered) > 0:
                    # Traditional replay buffer, capped per class via reservoir sampling (W4)
                    add_to_replay_buffer(replay_buffer, x_slice_filtered, y_slice_filtered, replay_rng,
                                          config.MAX_REPLAY_SAMPLES_PER_CLASS, replay_seen_counts)
                    print(f"      ✓ Updated replay buffer (filtered out deleted classes)")
                else:
                    print(f"      ⚠️  No samples to add to replay buffer after filtering deleted classes")

                if current_model is not None and len(x_val_filtered) > 0:
                    create_shard_confusion_matrix(
                        current_model, x_val_filtered, y_val_filtered, self.class_names,
                        shard_idx, reports_dir, active_classes, head_classes=head_classes
                    )

            slice_save_path = os.path.join(shard_model_dir, f"slice_{slice_idx}_model_{self.model_name}.pth")
            new_meta = {'unlearned': True}
            save_model_pytorch(current_model, slice_save_path, metadata=new_meta)

        if current_model:
            final_model_path = os.path.join(shard_model_dir, f"final_model_shard{shard_idx+1}_{self.model_name}.pth")
            save_model_pytorch(current_model, final_model_path, metadata={'unlearning_complete': True})
        
        # Store shard histories for overall visualization
        self.unlearning_histories.append(shard_histories)
        
        # Return both total time (with eval) and pure training time (without eval)
        total_retrain_time = time.time() - retrain_start_time
        return total_retrain_time, pure_retrain_time


    def final_evaluation_with_self_routing(self):
        print("\n" + "="*20 + " Final Evaluation with Self-Routing " + "="*20)
        # Positional list (index i == shard i+1); missing models stay None so
        # they line up with shard_class_indices by index.
        shard_models = [None] * self.num_shards
        for i in range(self.num_shards):
            model_path = os.path.join(self.models_dir, f"shard_{i+1}", f"final_model_shard{i+1}_{self.model_name}.pth")

            if os.path.exists(model_path):
                model, _ = load_model_pytorch(model_path)
                shard_models[i] = model.eval()
            else:
                print(f"   - Warning: No final model found for shard {i+1} at {os.path.basename(model_path)}")

        shard_class_indices = load_shard_class_indices(self.data_dir, self.num_shards)

        # W16: the gate is the real router now, so it must be retrained excluding
        # every class unlearned so far (including the one just tracked) for the
        # system to stay exact -- an untrained-on-that-class gate and a
        # retrained-to-exclude-it gate must behave indistinguishably on that
        # class's images, which is only true right after a retrain.
        gating_model = None
        gating_path = os.path.join(self.models_dir, "gating_model.pth")
        self.gating_retrain_time = 0.0
        if os.path.exists(gating_path):
            print("\n--- Retraining gating network to exclude all unlearned classes ---")
            gate_retrain_start = time.time()
            train_gating(
                num_shards=self.num_shards,
                base_dir=self.base_dir,
                num_slices=self.slices_per_shard or self.num_slices,
                dataset_mean=self.dataset_mean,
                dataset_std=self.dataset_std,
                excluded_classes=self.get_unlearned_classes(),
            )
            self.gating_retrain_time = time.time() - gate_retrain_start
            print(f"Gating Retrain Time: {self.gating_retrain_time:.2f} seconds")
            gating_model, _ = load_model_pytorch(gating_path, num_shards=self.num_shards)
            gating_model.eval()
        else:
            print(f"   - Warning: No gating model found at {os.path.basename(gating_path)}; "
                  f"falling back to confidence-based self-routing.")

        classified_accuracy, overall_accuracy, unlearning_report = self.evaluate_with_self_routing(
            shard_models, shard_class_indices, self.class_names, gating_model=gating_model
        )

        return shard_models, classified_accuracy, overall_accuracy, unlearning_report

    def evaluate_with_self_routing(self, shard_models, shard_class_indices, class_names, threshold=None, gating_model=None):
        """The primary, reported evaluation: raw self-routing, no confidence threshold
        (W5). A threshold is a deployment-time rejection knob, not exactness evidence --
        pass one explicitly only for an isolated, clearly-labeled deployment check; it
        never affects the numbers this method returns from its default (threshold=None)
        call, which is the one used for every unlearning report.
        """
        if threshold is None:
            print("\n" + "="*20 + " Self-Routing Evaluation (raw, no confidence threshold) " + "="*20)
        else:
            print("\n" + "="*20 + f" DEPLOYMENT-ONLY Evaluation with Confidence Threshold={threshold:.2f} (NOT exactness evidence) " + "="*20)
        x_test = np.load(os.path.join(self.test_data_dir, "x_test.npy"))
        y_test = np.load(os.path.join(self.test_data_dir, "y_test.npy"))

        for model in shard_models:
            if model is not None:
                model.eval()

        # Real-time self-routing (no caching - proper for new samples)
        all_final_preds = []
        all_true_labels = []
        batch_size = config.BATCH_SIZE

        # Get unlearned classes for analysis
        unlearned_classes_set = self.get_unlearned_classes_set()

        with torch.no_grad():
            for i in range(0, len(x_test), batch_size):
                batch_x = torch.from_numpy(x_test[i:i+batch_size]).float()
                batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)

                # Self-routing WITH threshold (production mode)
                # Threshold helps detect uncertain predictions on deleted/unknown classes
                batch_final_preds, _ = _run_sisa_batch(
                    batch_x_normalized, shard_models, class_names, shard_class_indices, threshold,
                    gating_model=gating_model,
                )
                
                # Keep predictions including -1 (uncertain/unknown samples)
                # This shows production behavior: reject low-confidence predictions
                batch_final_preds_np = batch_final_preds.cpu().numpy()
                
                all_final_preds.extend(batch_final_preds_np)
                all_true_labels.extend(y_test[i:i+batch_size])

        all_final_preds = np.array(all_final_preds)
        all_true_labels = np.array(all_true_labels)

        # Count rejected predictions (marked as -1 due to low confidence). Always 0
        # when threshold=None (the default, exactness-evidence path) since
        # _run_sisa_batch never rejects without a threshold.
        rejected_count = np.sum(all_final_preds == -1)
        rejected_percentage = (rejected_count / len(all_final_preds)) * 100

        if threshold is None:
            print(f"Total Test Samples: {len(all_final_preds)} (raw self-routing, nothing rejected)")
        else:
            print("\n--- DEPLOYMENT-ONLY: CONFIDENCE THRESHOLDING (not exactness evidence) ---")
            print(f"Total Test Samples: {len(all_final_preds)}")
            print(f"Rejected Predictions (low confidence): {rejected_count} ({rejected_percentage:.2f}%)")
            print(f"Accepted Predictions: {len(all_final_preds) - rejected_count} ({100 - rejected_percentage:.2f}%)")
        
        # Filter out rejected predictions for accuracy calculation
        valid_mask = all_final_preds != -1
        all_final_preds_complete = all_final_preds[valid_mask]
        all_true_labels_complete = all_true_labels[valid_mask]

        print("\n--- COMPREHENSIVE UNLEARNING EVALUATION (On Accepted Predictions) ---")
        print(f"Evaluating on {len(all_final_preds_complete)} accepted predictions")
        
        # Display total samples per class (including deleted classes)
        print("\nPer-Class Sample Distribution:")
        unique_labels, counts = np.unique(all_true_labels_complete, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            class_name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
            print(f"  {class_name}: {count} samples")

        # Calculate overall accuracy including deleted classes
        overall_accuracy = np.mean(all_final_preds_complete == all_true_labels_complete)
        
        # ============================================================================
        # MAIN CLASSIFICATION REPORT (Including Deleted Classes - Shows Unlearning)
        # ============================================================================
        print("\n" + "-"*60)
        print("Classification Report (All Classes - Including Deleted for Unlearning Analysis):")
        print("-"*60)
        report_str = classification_report(all_true_labels_complete, all_final_preds_complete, 
                                         target_names=class_names, 
                                         labels=np.arange(len(class_names)), 
                                         zero_division=0)
        report_dict = classification_report(all_true_labels_complete, all_final_preds_complete, 
                                          target_names=class_names, 
                                          labels=np.arange(len(class_names)), 
                                          zero_division=0, output_dict=True)
        print(report_str)

        print(f"SISA Overall Accuracy (All Classes): {overall_accuracy:.4f}")
        print("-"*60)
        print("Using True SISA Gating Method (specialist routing)")

        # Use overall accuracy as the main metric (includes deleted classes showing unlearning effect)
        gating_accuracy = overall_accuracy

        # ============================================================================
        # NEW: EVALUATE DELETED CLASS PERFORMANCE (Raw predictions on deleted samples)
        # ============================================================================
        print("\n" + "="*80)
        print("DELETED CLASS PERFORMANCE ANALYSIS (Raw Model Behavior)")
        print("="*80)

        # Re-run evaluation WITHOUT post-processing to see raw predictions on deleted classes
        print("Evaluating raw model predictions on ALL test samples (including deleted classes)...")

        all_raw_preds = []
        all_true_labels_full = []

        with torch.no_grad():
            for i in range(0, len(x_test), batch_size):
                batch_x = torch.from_numpy(x_test[i:i+batch_size]).float()
                batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)

                # Get RAW predictions without post-processing
                batch_raw_preds, _ = _run_sisa_batch(
                    batch_x_normalized, shard_models, class_names, shard_class_indices, threshold=None,
                    gating_model=gating_model,
                )

                all_raw_preds.extend(batch_raw_preds.cpu().numpy())
                all_true_labels_full.extend(y_test[i:i+batch_size])

        all_raw_preds = np.array(all_raw_preds)
        all_true_labels_full = np.array(all_true_labels_full)

        # Analyze deleted class performance
        unlearned_classes = self.get_unlearned_classes()
        deleted_class_results = {}

        for deleted_class_idx in unlearned_classes:
            deleted_class_name = class_names[deleted_class_idx]

            # Find samples of this deleted class
            deleted_mask = (all_true_labels_full == deleted_class_idx)
            deleted_samples = np.sum(deleted_mask)

            if deleted_samples > 0:
                deleted_preds = all_raw_preds[deleted_mask]
                deleted_true = all_true_labels_full[deleted_mask]

                # Calculate performance metrics for deleted class
                correct_predictions = np.sum(deleted_preds == deleted_true)
                accuracy = correct_predictions / deleted_samples

                # Most common wrong predictions
                wrong_preds = deleted_preds[deleted_preds != deleted_true]
                if len(wrong_preds) > 0:
                    unique_wrong, counts_wrong = np.unique(wrong_preds, return_counts=True)
                    most_common_wrong = unique_wrong[np.argmax(counts_wrong)]
                    most_common_wrong_name = class_names[most_common_wrong] if most_common_wrong < len(class_names) else f"class_{most_common_wrong}"
                else:
                    most_common_wrong_name = "None"

                deleted_class_results[deleted_class_name] = {
                    'samples': deleted_samples,
                    'accuracy': accuracy,
                    'most_confused_with': most_common_wrong_name
                }

                print(f"\nDELETED CLASS: '{deleted_class_name.upper()}'")
                print(f"   Test Samples: {deleted_samples}")
                print(f"   Raw Accuracy: {accuracy:.4f} ({correct_predictions}/{deleted_samples})")
                print(f"   Most Confused With: {most_common_wrong_name}")

                # Show prediction distribution
                unique_preds, pred_counts = np.unique(deleted_preds, return_counts=True)
                print("   Prediction Distribution:")
                for pred_idx, count in zip(unique_preds, pred_counts):
                    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
                    percentage = (count / deleted_samples) * 100
                    print(f"     {pred_name}: {count} samples ({percentage:.1f}%)")

        # Overall analysis
        total_deleted_samples = sum(result['samples'] for result in deleted_class_results.values())
        avg_deleted_accuracy = np.mean([result['accuracy'] for result in deleted_class_results.values()])

        print("\n OVERALL DELETED CLASS ANALYSIS:")
        print(f"   Total Deleted Class Samples: {total_deleted_samples}")
        print(f"   Average Raw Accuracy on Deleted Classes: {avg_deleted_accuracy:.4f}")
        print("   Expected: Near 0.0 (random chance) for successful unlearning")

        if avg_deleted_accuracy < 0.15:  # Less than 15% accuracy
            print("   Model shows strong unlearning effects")
        elif avg_deleted_accuracy < 0.30:  # Less than 30% accuracy
            print("   Model shows some unlearning effects")
        else:
            print("   Warning: Model may not have properly unlearned deleted classes")

        # ============================================================================
        # END: DELETED CLASS PERFORMANCE ANALYSIS
        # ============================================================================

        # ============================================================================
        # NEW: ACTIVE CLASSES ONLY EVALUATION (Excluding Deleted Classes)
        # This shows clean performance on classes the model should still know
        # NO CONFIDENCE MASKING - evaluate on ALL test samples for active classes
        # ============================================================================
        print("\n" + "="*80)
        print("ACTIVE CLASSES EVALUATION (Excluding Deleted Classes - NO Confidence Masking)")
        print("="*80)
        print("Evaluating model performance on remaining/active classes only...")
        print("Note: Using ALL test samples (no confidence thresholding) for active class evaluation")
        
        # Get unlearned class indices (unlearned_classes is List[int] from metadata)
        unlearned_class_indices = set(unlearned_classes)  # Already indices!
        
        # Get deleted class names for display
        deleted_class_names = [class_names[idx] for idx in unlearned_class_indices if idx < len(class_names)]
        
        # Re-run predictions WITHOUT confidence threshold for active classes.
        # W9: this single pass is also reused (via `precomputed=`) by every plot
        # call below, which previously each re-ran _run_sisa_batch over the same
        # test set independently (~7 redundant passes collapsed into this one).
        print("\nRe-evaluating with NO confidence threshold for clean active class metrics...")
        all_active_preds = []
        all_active_probs = []
        all_active_labels = []

        with torch.no_grad():
            for i in range(0, len(x_test), batch_size):
                batch_x = torch.from_numpy(x_test[i:i+batch_size]).float()
                batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)
                batch_y = y_test[i:i+batch_size]

                # NO threshold - get predictions on all samples
                batch_preds, batch_probs = _run_sisa_batch(
                    batch_x_normalized, shard_models, class_names, shard_class_indices, threshold=None,
                    gating_model=gating_model,
                )

                all_active_preds.extend(batch_preds.cpu().numpy())
                all_active_probs.append(batch_probs.cpu().numpy())
                all_active_labels.extend(batch_y)

        all_active_preds = np.array(all_active_preds)
        all_active_probs = np.concatenate(all_active_probs, axis=0)
        all_active_labels = np.array(all_active_labels)
        raw_eval_precomputed = (all_active_preds, all_active_probs, all_active_labels)
        
        # Filter to keep only active class samples (samples whose TRUE label is NOT deleted)
        active_mask = np.array([label not in unlearned_class_indices for label in all_active_labels])
        active_preds = all_active_preds[active_mask]
        active_labels = all_active_labels[active_mask]
        
        # Get active class names and indices
        active_class_indices = sorted([i for i in range(len(class_names)) if i not in unlearned_class_indices])
        active_class_names = [class_names[i] for i in active_class_indices]
        
        print(f"\nActive Classes: {active_class_names}")
        print(f"Deleted Classes (indices): {sorted(list(unlearned_class_indices))}")
        print(f"Deleted Classes (names): {deleted_class_names}")
        print(f"Total Active Class Samples (true labels only): {len(active_preds)}")
        
        # Additional filter: Remove samples where prediction is a deleted class
        # (model wrongly predicts deleted class - count as error but exclude from per-class stats)
        active_pred_mask = np.array([pred not in unlearned_class_indices for pred in active_preds])
        num_deleted_class_predictions = np.sum(~active_pred_mask)
        
        if num_deleted_class_predictions > 0:
            print(f"\nNote: {num_deleted_class_predictions} predictions were for deleted classes (counted as errors, excluded from per-class report)")
        
        # Apply filter to remove predictions of deleted classes
        active_preds_filtered = active_preds[active_pred_mask]
        active_labels_filtered = active_labels[active_pred_mask]
        
        print(f"Samples for classification report (after filtering deleted class predictions): {len(active_preds_filtered)}")
        
        # Display per-class distribution for active classes
        print("\nPer-Class Sample Distribution (Active Classes Only - True Labels):")
        unique_active_labels, active_counts = np.unique(active_labels_filtered, return_counts=True)
        for label_idx, count in zip(unique_active_labels, active_counts):
            class_name = class_names[label_idx]
            print(f"  {class_name}: {count} samples")
        
        # Calculate accuracy on active classes only (using all samples, including wrong deleted class predictions)
        active_accuracy = np.mean(active_preds == active_labels)
        
        # Classification report for active classes only (using filtered samples)
        print("\n" + "-"*60)
        print("Classification Report (Active Classes Only):")
        print("-"*60)
        active_report_str = classification_report(
            active_labels_filtered, 
            active_preds_filtered,
            target_names=active_class_names,
            labels=active_class_indices,
            zero_division=0
        )
        active_report_dict = classification_report(
            active_labels_filtered,
            active_preds_filtered,
            target_names=active_class_names,
            labels=active_class_indices,
            zero_division=0,
            output_dict=True
        )
        print(active_report_str)
        print(f"Active Classes Overall Accuracy: {active_accuracy:.4f}")
        print("-"*60)
        
        # Store active class metrics for comparison charts
        self.active_classes_accuracy = active_accuracy
        self.active_classes_report = active_report_dict
        
        # ============================================================================
        # END: ACTIVE CLASSES ONLY EVALUATION
        # ============================================================================

        # ============================================================================
        # GENERATE PLOTS FOR ACTIVE CLASSES ONLY
        # ============================================================================
        print("\n" + "=" * 50)
        print("CREATING ACTIVE CLASSES ONLY VISUALIZATIONS")
        print("=" * 50)
        
        # ROC Curves for active classes
        print("   Creating ROC curves for active classes only...")
        create_overall_sisa_roc_curve(
            shard_models,
            shard_class_indices,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'active_classes_only',
            unlearned_classes=unlearned_classes,
            precomputed=raw_eval_precomputed,
        )

        # Confusion Matrix for active classes
        print("   Creating confusion matrix for active classes only...")
        create_overall_sisa_confusion_matrix(
            shard_models,
            shard_class_indices,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'active_classes_only',
            unlearned_classes=unlearned_classes,
            precomputed=raw_eval_precomputed,
        )
        
        # Training curves for active classes (if available)
        print("   Creating training curves for active classes...")
        if hasattr(self, 'unlearning_histories') and self.unlearning_histories:
            create_overall_sisa_training_curves(
                self.unlearning_histories,
                self.reports_dir,
                'active_classes_only'
            )
        
        # ============================================================================
        # GENERATE PLOTS FOR ALL CLASSES (INCLUDING DELETED - Shows Unlearning Effect)
        # ============================================================================
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATIONS WITH ALL CLASSES (INCLUDING DELETED)")
        print(f"These plots show ALL {len(class_names)} classes to demonstrate unlearning effectiveness")
        print("=" * 50)
        
        # Get unlearned classes from metadata
        unlearned_classes = self.get_unlearned_classes()
        
        # ROC curves - ALL classes including deleted (to show they have poor performance)
        print("   Creating ROC curves with ALL classes (including deleted)...")
        create_overall_sisa_roc_curve(
            shard_models,
            shard_class_indices,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'with_deleted_classes',
            unlearned_classes=None,  # Don't filter - show ALL classes
            precomputed=raw_eval_precomputed,
        )

        # Confusion matrix - ALL classes including deleted
        print("   Creating confusion matrix with ALL classes (including deleted)...")
        create_overall_sisa_confusion_matrix(
            shard_models,
            shard_class_indices,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'with_deleted_classes',
            unlearned_classes=None,  # Don't filter - show ALL classes
            precomputed=raw_eval_precomputed,
        )
        
        # ============================================================================
        # GENERATE PLOTS FOR ACTIVE CLASSES ONLY (Clean Performance Metrics)
        # ============================================================================
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATIONS FOR UNLEARNING EVALUATION (ACTIVE CLASSES)")
        print("=" * 50)
        
        create_overall_sisa_roc_curve(
            shard_models,
            shard_class_indices,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'unlearning_evaluation',
            unlearned_classes=unlearned_classes,  # Exclude deleted classes
            precomputed=raw_eval_precomputed,
        )

        # Create overall confusion matrix showing only active classes
        print("\n" + "=" * 50)
        print("CREATING OVERALL SISA SYSTEM CONFUSION MATRIX (AFTER UNLEARNING)")
        print("=" * 50)

        create_overall_sisa_confusion_matrix(
            shard_models,
            shard_class_indices,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'unlearning_evaluation',
            unlearned_classes=unlearned_classes,  # Exclude unlearned classes
            precomputed=raw_eval_precomputed,
        )
        
        # Generate updated training curves showing unlearning impact
        print("\n" + "=" * 50)
        print("CREATING OVERALL SISA SYSTEM TRAINING CURVES (AFTER UNLEARNING)")
        print("=" * 50)
        
        # Create training curves from unlearning histories
        if hasattr(self, 'unlearning_histories') and self.unlearning_histories:
            create_overall_sisa_training_curves(
                self.unlearning_histories,
                self.reports_dir,
                'unlearning_evaluation'
            )
        
        # Return gating accuracy and classification report (true SISA approach)
        return gating_accuracy, gating_accuracy, report_dict

    def _evaluate_on_forgotten_samples(self, shard_models):
        if self.forgotten_samples_x is None or len(self.forgotten_samples_x) == 0:
            return
        
        # Consolidate samples if needed (for batch unlearning)
        if isinstance(self.forgotten_samples_x, list):
            self.forgotten_samples_x = np.concatenate(self.forgotten_samples_x)
            self.forgotten_samples_y = np.array(self.forgotten_samples_y)

        x_forgotten = self.forgotten_samples_x
        y_forgotten = self.forgotten_samples_y
        
        # Get unique classes in forgotten samples
        unique_forgotten_classes = np.unique(y_forgotten)
        class_names_str = ', '.join([self.class_names[c] for c in unique_forgotten_classes])
        
        print("\n" + "="*20 + f" Verification on Forgotten Samples ({class_names_str}) " + "="*20)
        print(f"Total forgotten samples: {len(y_forgotten)}")
        for class_idx in unique_forgotten_classes:
            count = np.sum(y_forgotten == class_idx)
            print(f"  - {self.class_names[class_idx]}: {count} samples")

        # Self-routing prediction (single source of truth, same as all other evaluation)
        shard_class_indices = load_shard_class_indices(self.data_dir, self.num_shards)
        batch_x = torch.from_numpy(x_forgotten).float()
        batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)
        with torch.no_grad():
            preds_tensor, probs_tensor = _run_sisa_batch(
                batch_x_normalized, shard_models, self.class_names, shard_class_indices, threshold=None,
                gating_model=getattr(self, 'gating_model', None),
            )
        all_preds = preds_tensor.cpu().numpy()
        all_confidences = probs_tensor.max(dim=1).values.cpu().numpy()

        # Calculate overall accuracy
        num_correct = np.sum(all_preds == y_forgotten)
        total_samples = len(y_forgotten)
        accuracy = num_correct / total_samples

        print(f"\nOverall Accuracy on forgotten samples: {num_correct}/{total_samples} = {accuracy:.2%}")
        
        # Show per-class accuracy
        for class_idx in unique_forgotten_classes:
            class_mask = (y_forgotten == class_idx)
            class_correct = np.sum(all_preds[class_mask] == y_forgotten[class_mask])
            class_total = np.sum(class_mask)
            class_acc = class_correct / class_total if class_total > 0 else 0
            print(f"  - {self.class_names[class_idx]}: {class_correct}/{class_total} = {class_acc:.2%}")
        
        if accuracy < config.UNLEARNING_SUCCESS_THRESHOLD:
            print("Unlearning completed: Model performs near random chance on forgotten samples")
        else:
            print("Unlearning may be incomplete: Model still shows significant accuracy on forgotten samples")

        # Create SEPARATE visualization for EACH forgotten class (16 samples each)
        print(f"\nCreating individual verification plots for {len(unique_forgotten_classes)} forgotten class(es)...")
        
        for class_idx in unique_forgotten_classes:
            class_name = self.class_names[class_idx]
            
            # Get samples for this specific class
            class_mask = (y_forgotten == class_idx)
            class_x = x_forgotten[class_mask]
            class_y = y_forgotten[class_mask]
            class_preds = all_preds[class_mask]
            class_confidences = all_confidences[class_mask]
            
            class_total = len(class_y)
            class_correct = np.sum(class_preds == class_y)
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            
            print(f"\n   Creating verification plot for class '{class_name}':")
            print(f"     - Samples: {class_total}")
            print(f"     - Accuracy: {class_accuracy:.2%} ({class_correct}/{class_total})")
            
            # Create 4x4 grid for this class (16 samples)
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f"Model Predictions for Forgotten Class: '{class_name}'\nAccuracy: {class_accuracy:.2%} ({class_correct}/{class_total})", 
                        fontsize=16, fontweight='bold')
            
            # Show up to 16 samples for this class
            for i, ax in enumerate(axes.flat):
                if i >= min(16, class_total):  # Show up to 16 samples
                    ax.axis('off')
                    continue
                    
                if i < class_total:
                    # Show actual sample
                    img = class_x[i].transpose((1, 2, 0))
                    ax.imshow(img)
                    true_label = self.class_names[class_y[i]]
                    pred_label = self.class_names[class_preds[i]] if class_preds[i] < len(self.class_names) else f"Class_{class_preds[i]}"
                    confidence = class_confidences[i]
                    color = "green" if true_label == pred_label else "red"
                    ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}", color=color, fontsize=8)
                ax.axis('off')
            
            plt.tight_layout(rect=config.PLOT_TIGHT_LAYOUT_RECT)
            
            # Save with class-specific filename
            save_path = os.path.join(self.reports_dir, f'SISA_Unlearning_Verification_{class_name.lower()}.png')
            plt.savefig(save_path)
            plt.close()
            
            print(f"     - Saved: {os.path.basename(save_path)}")

    def _backup_test_set_if_needed(self):
        """Create backup of original test set if it doesn't exist"""
        x_test_path = os.path.join(self.test_data_dir, "x_test.npy")
        y_test_path = os.path.join(self.test_data_dir, "y_test.npy")
        x_backup_path = os.path.join(self.test_data_dir, "x_test_original.npy")
        y_backup_path = os.path.join(self.test_data_dir, "y_test_original.npy")
        
        if not os.path.exists(x_backup_path) and os.path.exists(x_test_path):
            import shutil
            shutil.copy2(x_test_path, x_backup_path)
            shutil.copy2(y_test_path, y_backup_path)
            print("   - Created backup of original test set")

    def _permanently_remove_class_from_test_set(self, class_to_remove: int):
        """
        DISABLED: This function was modified to preserve test samples.
        Test samples are kept intact for proper evaluation and reproducibility.
        """
        print("\n" + "="*20 + " Test Set Preservation Policy " + "="*20)
        
        class_name = self.class_names[class_to_remove]
        print(f"   - Class '{class_name}' samples preserved in test set.")
        print("   - Test set remains unchanged for evaluation purposes.")
        print("   - This ensures consistent evaluation and allows for proper unlearning verification.")
        print("   - Original test samples are maintained for reproducibility.")
    
    def _get_cumulative_classes_up_to_slice_unlearning(self, shard_idx: int, target_slice_idx: int) -> List[int]:
        """
        Get all classes that should be known up to the target slice for unlearning (excluding deleted classes).
        Similar to training's get_cumulative_classes_up_to_slice but respects unlearned classes.
        """
        cumulative_classes = set()
        
        # Accumulate classes from all slices up to and including target_slice_idx
        for slice_idx in range(target_slice_idx + 1):
            x_slice, y_slice = self._load_slice_data(shard_idx, slice_idx)
            if x_slice is not None and len(x_slice) > 0:
                slice_classes = set(np.unique(y_slice))
                cumulative_classes.update(slice_classes)
        
        # Remove any classes that have been unlearned (they should not be in validation)
        # Note: If a class was completely removed from all slices, it won't be in cumulative_classes anyway
        return sorted(cumulative_classes)
    
    def _get_incremental_validation_data_unlearning(self, known_classes: List[int], validation_data: tuple) -> tuple:
        """
        Filter validation data using true labels for unlearning validation.
        """
        x_val, y_val = validation_data
        
        known_classes_set = set(known_classes)
        mask = np.array([label in known_classes_set for label in y_val])
        
        x_val_filtered = x_val[mask]
        y_val_filtered = y_val[mask]
        
        print(f"   Validation: {len(known_classes)} classes, {len(x_val_filtered)} samples")
        
        return x_val_filtered, y_val_filtered

    def check_class_balance_and_augmentation_unlearning(self, shard_idx, class_names):
        """
        Check class balance in a shard for unlearning and determine augmentation.
        Similar to training but adapted for unlearning context.
        """
        print(f"\n--- Checking Class Balance for Unlearning Shard {shard_idx+1} ---")
        
        # Load all slice data for this shard
        all_labels = []
        for slice_idx in range(self.num_slices):
            _, y_slice = self._load_slice_data(shard_idx, slice_idx)
            if y_slice is not None:
                all_labels.extend(y_slice)
        
        if not all_labels:
            print("   No data found for balance analysis")
            return None
        
        # Calculate class distribution
        from collections import Counter
        class_counts = Counter(all_labels)
        total_samples = len(all_labels)
        
        print(f"   Total samples: {total_samples}")
        print("   Class distribution:")
        
        # Calculate percentages and find min/max
        class_percentages = {}
        min_percentage = float('inf')
        max_percentage = 0
        
        for class_idx in sorted(class_counts.keys()):
            count = class_counts[class_idx]
            percentage = (count / total_samples) * 100
            class_percentages[class_idx] = percentage
            min_percentage = min(min_percentage, percentage)
            max_percentage = max(max_percentage, percentage)
            
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            print(f"     {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Calculate balance metrics
        balance_ratio = min_percentage / max_percentage if max_percentage > 0 else 1.0
        std_dev = np.std(list(class_percentages.values()))
        
        print(f"   Balance ratio: {balance_ratio:.3f}")
        print(f"   Standard deviation: {std_dev:.3f}")
        print(f"   Min/Max percentages: {min_percentage:.3f}% / {max_percentage:.3f}%")
        
        # For unlearning, be more conservative with augmentation since we're retraining
        # Only augment if severely unbalanced
        PERFECT_BALANCE_THRESHOLD = 0.90  # More conservative than training
        BALANCE_THRESHOLD = 0.75  # More conservative threshold for unlearning
        STD_THRESHOLD = 2.0       # Higher threshold for unlearning
        
        if balance_ratio >= PERFECT_BALANCE_THRESHOLD and std_dev <= STD_THRESHOLD:
            # W19: must mirror entry_training.py -- retraining under a different
            # augmentation recipe than original training would break the W8 comparison.
            print("   Classes are perfectly balanced for unlearning - using baseline augmentation")
            return config.get_augmentation_config('baseline', is_unlearning=True)
        elif balance_ratio >= BALANCE_THRESHOLD and std_dev <= 3.0:
            print("   Classes are well balanced for unlearning - using minimal augmentation")
            return config.get_augmentation_config('minimal', is_unlearning=True)
        elif balance_ratio >= 0.65 and std_dev <= 5.0:
            print("   Classes moderately unbalanced for unlearning - using light augmentation")
            return config.get_augmentation_config('light', is_unlearning=True)
        else:
            print("   Classes significantly unbalanced for unlearning - using moderate augmentation")
            return config.get_augmentation_config('moderate', is_unlearning=True)

    def _get_training_metrics(self):
        """Read training accuracy and timing breakdown from training_metrics.json
        (written by entry_training.py -- W7). No log scraping, no fallbacks: if
        the project hasn't been trained, this fails loudly rather than guessing."""
        metrics_path = os.path.join(self.data_dir, "training_metrics.json")
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        training_accuracy = metrics['final_accuracy']
        gating_training_time = metrics['gating_training_time_pure']
        base_model_time = metrics['base_model_training_time_with_eval']
        pure_training_time = metrics['base_model_training_time_pure']
        training_report = metrics['classification_report']

        return training_accuracy, gating_training_time, base_model_time, pure_training_time, training_report

    def _create_comparison_visualizations(self, unlearning_accuracy, total_time, retrain_time, unlearned_class, 
                                        training_report=None, unlearning_report=None):
        """Create comparison visualizations between training and unlearning."""
        print("\n" + "=" * 50)
        print("CREATING TRAINING VS UNLEARNING COMPARISON CHARTS")
        print("=" * 50)
        
        # Get training metrics
        training_accuracy, gating_training_time, base_model_time, pure_training_time, _ = self._get_training_metrics()
        
        # Create accuracy comparison
        create_accuracy_comparison_chart(
            training_accuracy, unlearning_accuracy, unlearned_class, self.reports_dir
        )
        
        # Create time comparison with real timing data (includes evaluation time)
        # Use the actual measured timing values from the unlearning process
        unlearning_total_time = total_time  # This is the real total unlearning time
        # Use the actual measured retraining time instead of cumulative retrain_time
        actual_retraining_time = getattr(self, 'retraining_time', retrain_time)
        create_time_comparison_chart(
            gating_training_time, base_model_time, unlearning_total_time, actual_retraining_time, self.reports_dir
        )
        
        # Create pure training time breakdown chart (excludes evaluation/plotting time)
        # Shows: Gating Training, Base Training, Find+Delete, Retraining (pure times only)
        pure_retraining_time = getattr(self, 'pure_retraining_time', actual_retraining_time)
        create_pure_training_time_chart(
            gating_training_time, pure_training_time, self.data_removal_time, pure_retraining_time, self.reports_dir
        )
        
        # Create classification metrics comparison if reports are provided
        if training_report is not None and unlearning_report is not None:
            create_classification_metrics_comparison_chart(
                training_report, unlearning_report, unlearned_class, self.reports_dir
            )
        
        # Create ACTIVE CLASSES ONLY versions of comparison charts
        print("\n   Creating active classes only comparison charts...")
        # Get active class accuracy from evaluation
        active_classes_accuracy = getattr(self, 'active_classes_accuracy', unlearning_accuracy)
        
        # Create active classes only accuracy comparison
        create_accuracy_comparison_chart(
            training_accuracy, active_classes_accuracy, f"{unlearned_class} (Active Classes Only)", self.reports_dir
        )
        
        # Create active classes only classification metrics comparison
        if training_report is not None and hasattr(self, 'active_classes_report'):
            create_classification_metrics_comparison_chart(
                training_report, self.active_classes_report, f"{unlearned_class} (Active Classes Only)", self.reports_dir
            )

    def _evaluate_deleted_class_accuracy(self, shard_models, deleted_class_name: str, deleted_class_idx: int):
        """
        Comprehensive evaluation of deleted class performance with bar chart visualization.
        
        This function evaluates:
        1. Pre-unlearning accuracy on deleted class (from training baseline)
        2. Post-unlearning accuracy on deleted class (should be ~0%)
        3. Post-unlearning accuracy on remaining classes (should maintain performance)
        4. Threshold-based logic for deleted vs unseen class classification
        
        Args:
            shard_models: List of retrained shard models after unlearning
            deleted_class_name: Name of the deleted class
            deleted_class_idx: Index of the deleted class
        """
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE DELETED CLASS EVALUATION: '{deleted_class_name.upper()}'")
        print("=" * 80)
        
        # Get training baseline metrics from training_metrics.json (W7)
        training_accuracy_for_deleted_class = self._get_pre_unlearning_class_accuracy(deleted_class_name)
            
        print(f"Pre-unlearning accuracy on '{deleted_class_name}': {training_accuracy_for_deleted_class:.4f}")
        
        # Load test data (contains deleted class samples for evaluation)
        # Use current test data since we're evaluating BEFORE permanent deletion
        x_test_current_path = os.path.join(self.data_dir, "test_data/x_test.npy")
        y_test_current_path = os.path.join(self.data_dir, "test_data/y_test.npy")
        
        # Fallback to original/backup files if current files are already modified
        if not (os.path.exists(x_test_current_path) and os.path.exists(y_test_current_path)):
            x_test_current_path = os.path.join(self.data_dir, "test_data/x_test_original.npy")
            y_test_current_path = os.path.join(self.data_dir, "test_data/y_test_original.npy")
            
        # Final fallback to backup files
        if not (os.path.exists(x_test_current_path) and os.path.exists(y_test_current_path)):
            x_test_current_path = os.path.join(self.data_dir, "test_data/x_test_backup.npy")
            y_test_current_path = os.path.join(self.data_dir, "test_data/y_test_backup.npy")
        
        if not (os.path.exists(x_test_current_path) and os.path.exists(y_test_current_path)):
            print("   Warning: Test data not found. Cannot evaluate deleted class performance.")
            return
            
        x_test_full = np.load(x_test_current_path)
        y_test_full = np.load(y_test_current_path)
        
        # Split data: deleted class vs remaining classes
        deleted_mask = (y_test_full == deleted_class_idx)
        remaining_mask = (y_test_full != deleted_class_idx)
        
        x_deleted = x_test_full[deleted_mask]
        y_deleted = y_test_full[deleted_mask]
        x_remaining = x_test_full[remaining_mask]
        y_remaining = y_test_full[remaining_mask]
        
        print(f"Test samples for '{deleted_class_name}': {len(x_deleted)}")
        print(f"Test samples for remaining classes: {len(x_remaining)}")
        
        shard_class_indices = load_shard_class_indices(self.data_dir, self.num_shards)

        # Set up evaluation
        dataset_mean, dataset_std = self._get_dataset_normalization()
        eval_transforms = T.Compose([T.Normalize(dataset_mean, dataset_std)])

        # Evaluation 1: Post-unlearning accuracy on DELETED class (should be ~0%)
        post_unlearning_deleted_accuracy = self._evaluate_sisa_on_data(
            shard_models, shard_class_indices, x_deleted, y_deleted, eval_transforms
        )

        # Evaluation 2: Post-unlearning accuracy on REMAINING classes (should maintain performance)
        post_unlearning_remaining_accuracy = self._evaluate_sisa_on_data(
            shard_models, shard_class_indices, x_remaining, y_remaining, eval_transforms
        )
        
        # Results summary
        print("\nEVALUATION RESULTS:")
        print(f"Pre-unlearning '{deleted_class_name}' accuracy: {training_accuracy_for_deleted_class:.4f}")
        print(f"Post-unlearning '{deleted_class_name}' accuracy: {post_unlearning_deleted_accuracy:.4f}")
        print(f"Post-unlearning remaining classes accuracy: {post_unlearning_remaining_accuracy:.4f}")
        
        # Create bar chart visualization
        self._create_deleted_class_bar_chart(
            deleted_class_name, 
            training_accuracy_for_deleted_class,
            post_unlearning_deleted_accuracy,
            post_unlearning_remaining_accuracy
        )
        
        # Logical evaluation with threshold
        # GDPR Compliance: Model should perform at random chance (no evidence of training on deleted data)
        random_chance = 1.0 / len(self.class_names)  # Dataset-agnostic: 1/num_classes
        success_threshold_lower = random_chance - 0.05  # 0.05 (5%) - below random is acceptable
        success_threshold_upper = random_chance + 0.05  # 0.15 (15%) - above random is still acceptable
        
        print(f"\nGDPR Exact Unlearning Evaluation (Target: Random Chance ~{random_chance:.2f}):")
        if success_threshold_lower <= post_unlearning_deleted_accuracy <= success_threshold_upper:
            print(f"✓ GDPR COMPLIANT: Deleted class '{deleted_class_name}' shows random performance at {post_unlearning_deleted_accuracy:.4f}")
            print(f"   Acceptable range: {success_threshold_lower:.2f} - {success_threshold_upper:.2f} (near random chance)")
            print(f"   Model shows NO evidence of training on deleted data ✓")
        elif post_unlearning_deleted_accuracy > success_threshold_upper:
            print(f"✗ GDPR CONCERN: Deleted class '{deleted_class_name}' accuracy too HIGH at {post_unlearning_deleted_accuracy:.4f}")
            print(f"   Model still shows memory of deleted data (expected: ≤{success_threshold_upper:.2f})")
        else:
            print(f"⚠️  WARNING: Deleted class '{deleted_class_name}' accuracy too LOW at {post_unlearning_deleted_accuracy:.4f}")
            print(f"   Below-random performance may indicate systematic bias (expected: ≥{success_threshold_lower:.2f})")
            print(f"   This could be evidence of training influence (GDPR concern)")
            
        if post_unlearning_remaining_accuracy >= (training_accuracy_for_deleted_class * 0.85):
            print(f"✓ Remaining classes maintain {post_unlearning_remaining_accuracy:.4f} accuracy")
        else:
            print(f"✗ WARNING: Remaining classes dropped to {post_unlearning_remaining_accuracy:.4f} accuracy")
        
        print("=" * 80)

    def _get_pre_unlearning_class_accuracy(self, class_name: str) -> float:
        """Pre-unlearning accuracy for a specific class, from training_metrics.json's
        classification report (W7). No hardcoded fallback: a missing metrics file or
        class entry is a real problem (project never trained / class name typo'd),
        not something to silently paper over with a guessed number."""
        metrics_path = os.path.join(self.data_dir, "training_metrics.json")
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        report = metrics['classification_report']
        if class_name not in report:
            raise KeyError(f"Class '{class_name}' not found in {metrics_path}'s classification_report")

        class_metrics = report[class_name]
        return (class_metrics['precision'] + class_metrics['recall']) / 2

    def _evaluate_sisa_on_data(self, shard_models, shard_class_indices, x_data, y_data, eval_transforms):
        """Evaluate SISA system on given data with post-processing filtering for unlearned classes."""
        if len(x_data) == 0:
            return 0.0

        all_preds = []
        batch_size = config.BATCH_SIZE

        with torch.no_grad():
            for i in range(0, len(x_data), batch_size):
                batch_x = torch.from_numpy(x_data[i:i+batch_size]).float()
                batch_x_normalized = eval_transforms(batch_x).to(DEVICE)

                # Let model predict naturally - keep raw predictions
                batch_preds, _ = _run_sisa_batch(
                    batch_x_normalized, shard_models, self.class_names, shard_class_indices, threshold=None,
                    gating_model=getattr(self, 'gating_model', None),
                )
                
                # Keep RAW predictions to see true model behavior after unlearning
                batch_preds_np = batch_preds.cpu().numpy()
                
                all_preds.extend(batch_preds_np)
        
        all_preds = np.array(all_preds)
        
        # Use RAW predictions for accuracy calculation
        # This shows what the model actually predicts after unlearning
        correct_predictions = np.sum(all_preds == y_data)
        return correct_predictions / len(y_data)
    
    def _create_deleted_class_bar_chart(self, deleted_class_name, pre_acc, post_deleted_acc, post_remaining_acc):
        """Create bar chart showing pre/post unlearning accuracy comparison for DELETED class only."""
        _, ax = plt.subplots(figsize=(10, 7))
        
        # Only show Pre-unlearning and Post-unlearning for the deleted class
        # Removed "Post-unlearning Remaining Classes" as requested
        categories = [
            f"Pre-unlearning\n'{deleted_class_name}'",
            f"Post-unlearning\n'{deleted_class_name}'"
        ]
        
        accuracies = [pre_acc, post_deleted_acc]
        colors = ['#2E8B57', '#DC143C']  # Green (before), Red (after - should be near 0)
        
        bars = ax.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'Deleted Class Unlearning Verification: "{deleted_class_name.title()}"', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add grid and styling
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=11)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.reports_dir, f'deleted_class_accuracy_evaluation_{deleted_class_name.lower()}.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - Deleted class evaluation chart saved: {os.path.basename(chart_path)}")

    def _get_dataset_normalization(self):
        """Get dataset normalization parameters."""
        return config.get_dataset_normalization()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SISA Machine Unlearning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single class unlearning
  python unlearning/sisa_unlearning.py --class-name cat

  # Batch unlearning (multiple classes)
  python unlearning/sisa_unlearning.py --class-name cat dog ship
  python unlearning/sisa_unlearning.py --class-name cat dog

  # With custom project
  python unlearning/sisa_unlearning.py --class-name cat dog --project-name my_project
        """
    )
    parser.add_argument('--class-name', type=str, nargs='+',
                      help='Name(s) of the class(es) to forget. Use spaces to separate multiple classes.')
    parser.add_argument('--index', type=int, help='Index of specific sample to forget')
    parser.add_argument('--project-name', type=str, default='cifar10_sisa_pytorch', help='Project name')
    parser.add_argument('--model-name', type=str, default='custom_cnn', help='Model architecture name')

    args = parser.parse_args()

    # Project-scoped, timestamped logging (W10) -- now that --project-name is known.
    _restore_logging, _log_path = setup_run_logging(
        os.path.join(config.PROJECTS_DIR, args.project_name, "logs"), "unlearning"
    )

    try:
        if not args.class_name and args.index is None:
            print("Error: Either --class-name or --index must be specified")
            sys.exit(1)
            
        if args.class_name and args.index is not None:
            print("Error: Cannot specify both --class-name and --index")
            sys.exit(1)
        
        print("="*80)
        print("SISA MACHINE UNLEARNING")
        print("="*80)
        print(f"Project: {args.project_name}")
        print(f"Model: {args.model_name}")
        if args.class_name:
            if len(args.class_name) == 1:
                print(f"Target: Forget class '{args.class_name[0]}'")
            else:
                print(f"Target: Forget {len(args.class_name)} classes: {args.class_name}")
        else:
            print(f"Target: Forget sample at index {args.index}")
        print("="*80)
        
        # Initialize unlearning system
        unlearner = SISAUnlearning(args.project_name, args.model_name)
        
        if args.class_name:
            if len(args.class_name) == 1:
                # Single class unlearning
                unlearner.unlearn_by_class(args.class_name[0])
            else:
                # Batch unlearning for multiple classes
                unlearner.batch_unlearn_by_classes(args.class_name)
        else:
            # Unlearn by index (not implemented yet)
            print("Error: Unlearning by index is not implemented yet.")
            sys.exit(1)
            
        print("\n✅ Unlearning completed successfully!")
        
    except Exception as e:
        print(f"\n Error during unlearning: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"Log saved to: {_log_path}")
        _restore_logging()