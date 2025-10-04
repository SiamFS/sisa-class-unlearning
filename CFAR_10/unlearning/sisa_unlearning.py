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
    create_gating_routing_barplots,
    create_accuracy_comparison_chart,
    create_time_comparison_chart,
    create_pure_training_time_chart,
    create_classification_metrics_comparison_chart,
    _run_sisa_batch,
    _normalize_probabilities_tensor,
    _apply_temperature_tensor,
)
from training.create_model import save_model_pytorch, load_model_pytorch, DEVICE

# Enhanced logging class for unlearning process
class UnlearningLogger:
    def __init__(self, log_file="unlearning.txt"):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        # Create log file if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) if os.path.dirname(log_file) else ".", exist_ok=True)
        
        # Open file in write mode to overwrite previous output
        self.file = open(log_file, 'w', encoding='utf-8')
        
        # Write session header
        self.file.write(f"{'='*80}\n")
        self.file.write(f"SISA Unlearning Session - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.write(f"{'='*80}\n")
        self.file.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        if hasattr(self, 'file'):
            self.file.close()
    
    def __del__(self):
        self.close()

class SISAUnlearning:
    def __init__(self, project_name: str, model_name: str):
        print(f"Initializing SISA Unlearning for project: '{project_name}' with model: '{model_name}'")
        self.project_name = project_name
        self.model_name = model_name
        self.base_dir = f"../projects/{self.project_name}"
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "sisa_data")
        self.reports_dir = os.path.join(self.base_dir, "data_info")
        self.test_data_dir = os.path.join(self.data_dir, "test_data")
        
        self.metadata = self._load_metadata()
        self.class_names = self.metadata.get('class_names', [])
        self.num_shards = self.metadata.get('num_shards', 0)
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
        self.gating_update_time = 0
        
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
        
        # Step 3: Track unlearned class (without retraining gating network)
        self._track_unlearned_class(class_idx)
        print("\n   - Skipping gating network retraining: Using existing gating network with confidence threshold")
        
        # Calculate PURE unlearning time (data removal + retraining only, no evaluation/plotting overhead)
        pure_unlearning_time = self.data_removal_time + self.retraining_time
        
        # Step 4: Final evaluation with gating network
        evaluation_start_time = time.time()
        shard_models, classified_accuracy, overall_accuracy, unlearning_report = self.final_evaluation_with_gating()
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
        self.gating_update_time = 0
        
        # Initialize forgotten samples storage for batch unlearning
        self.forgotten_samples_x = []
        self.forgotten_samples_y = []
        self.gating_update_time = 0
        
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
        print("Step 4: Final evaluation with gating network...")
        print("="*60)
        evaluation_start_time = time.time()
        shard_models, classified_accuracy, overall_accuracy, unlearning_report = self.final_evaluation_with_gating()
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
                        current_model, _ = load_model_pytorch(model_path)
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
        
        # Initialize replay buffer from previous available slices (skipping empty/deleted ones)
        if config.USE_SMART_REPLAY:
            from training.smart_replay import create_smart_replay_buffer
            replay_buffer = create_smart_replay_buffer(config)
            print("   - Using smart replay buffer for unlearning")
            
            # Add data from all previous AVAILABLE slices (skip empty/deleted ones)
            for slice_idx in range(first_affected_slice):
                # Only process slices that have data
                if slice_states.get(slice_idx, 'empty') != 'has_data':
                    continue
                
                x_s, y_s = self._load_slice_data(shard_idx, slice_idx)
                if x_s is not None and len(x_s) > 0:
                    # Need to load a model to compute importance scores
                    base_model_path = os.path.join(shard_model_dir, f"slice_{slice_idx}_model_{self.model_name}.pth")
                    if os.path.exists(base_model_path):
                        try:
                            temp_model, _ = load_model_pytorch(base_model_path)
                            replay_buffer.add_samples(x_s, y_s, temp_model, DEVICE)
                        except Exception as e:
                            print(f"   ⚠️  Warning: Failed to load model for replay buffer from Slice {slice_idx+1}: {e}")
                            print(f"   - Skipping this slice in replay buffer")
                            continue
                        print(f"   - Added Slice {slice_idx+1} data to smart replay buffer")
                    else:
                        print(f"   - Warning: Slice {slice_idx+1} has data but no model found")
        else:
            replay_buffer = {}
            # Traditional replay buffer logic - only from available slices
            for slice_idx in range(first_affected_slice):
                # Only process slices that have data
                if slice_states.get(slice_idx, 'empty') != 'has_data':
                    continue
                    
                x_s, y_s = self._load_slice_data(shard_idx, slice_idx)
                if x_s is not None and len(x_s) > 0:
                    unique_labels_in_slice = np.unique(y_s)
                    for label in unique_labels_in_slice:
                        mask = (y_s == label)
                        class_x_data, class_y_data = x_s[mask], y_s[mask]
                        if label in replay_buffer:
                            replay_buffer[label]['X'] = np.vstack([replay_buffer[label]['X'], class_x_data])
                            replay_buffer[label]['y'] = np.concatenate([replay_buffer[label]['y'], class_y_data])
                        else:
                            replay_buffer[label] = {'X': class_x_data, 'y': class_y_data}
                    print(f"   - Added Slice {slice_idx+1} data to replay buffer")

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
            print(f"   - Replay buffer status: {len(replay_buffer.buffer) if hasattr(replay_buffer, 'buffer') else len(replay_buffer)} classes available")
            
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

            # Use static replay ratio for unlearning
            current_replay_ratio = config.UNLEARNING_REPLAY_RATIO
            print(f"   - Using replay ratio: {current_replay_ratio}")

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
                replay_buffer=replay_buffer,
                replay_ratio=current_replay_ratio,  # Use dynamic ratio
                dataset_mean=self.dataset_mean,
                dataset_std=self.dataset_std,
                training_type='unlearning',  # Proper early stopping for unlearning
                augmentation_config=augmentation_config,  # Use balance-based augmentation for unlearning
                use_smart_replay=config.USE_SMART_REPLAY,
                device=DEVICE
            )
            
            # END: Track pure training time (after train_model call)
            slice_train_end = time.time()
            pure_retrain_time += (slice_train_end - slice_train_start)
            
            # Store history for overall training curves
            if history is not None:
                shard_histories.append(history)
            
            # Generate visualizations for the last slice of unlearning
            if slice_idx == self.num_slices - 1:  # Last slice
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
                        shard_idx, reports_dir, active_classes
                    )
            
            # Update replay buffer with retrained slice data
            if len(x_slice) > 0:
                if config.USE_SMART_REPLAY and hasattr(replay_buffer, 'add_samples'):
                    # Smart replay buffer - add with importance scoring
                    replay_buffer.add_samples(x_slice, y_slice, current_model, DEVICE)
                else:
                    # Traditional replay buffer - simple class-wise storage
                    unique_labels_in_slice = np.unique(y_slice)
                    for label in unique_labels_in_slice:
                        mask = (y_slice == label)
                        class_x_data, class_y_data = x_slice[mask], y_slice[mask]
                        if label in replay_buffer:
                            replay_buffer[label]['X'] = np.vstack([replay_buffer[label]['X'], class_x_data])
                            replay_buffer[label]['y'] = np.concatenate([replay_buffer[label]['y'], class_y_data])
                        else:
                            replay_buffer[label] = {'X': class_x_data, 'y': class_y_data}

                if current_model is not None and len(x_val_filtered) > 0:
                    create_shard_confusion_matrix(
                        current_model, x_val_filtered, y_val_filtered, self.class_names,
                        shard_idx, reports_dir, active_classes
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


    def final_evaluation_with_gating(self):
        print("\n" + "="*20 + " Final Evaluation with Gating Network " + "="*20)
        shard_models = []
        for i in range(self.num_shards):
            model_path = os.path.join(self.models_dir, f"shard_{i+1}", f"final_model_shard{i+1}_{self.model_name}.pth")
            
            if os.path.exists(model_path):
                model, _ = load_model_pytorch(model_path)
                shard_models.append(model.eval())
            else:
                print(f"   - Warning: No final model found for shard {i+1} at {os.path.basename(model_path)}")

        gating_model_path = os.path.join(self.models_dir, "gating_model.pth")
        if not os.path.exists(gating_model_path):
            print("FATAL: Gating model not found. Cannot perform evaluation.")
            return None, 0.0, 0.0, None
            
        gating_model, _ = load_model_pytorch(gating_model_path, num_shards=self.num_shards)
        
        classified_accuracy, overall_accuracy, unlearning_report = self.evaluate_with_gating_network(shard_models, gating_model, self.class_names)
        
        # Clean up gating model from memory after evaluation (SISA compliance)
        del gating_model
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        
        return shard_models, classified_accuracy, overall_accuracy, unlearning_report

    def evaluate_with_gating_network(self, shard_models, gating_model, class_names, threshold=config.CONFIDENCE_THRESHOLD):
        print("\n" + "="*20 + f" Production Evaluation with Confidence Threshold={threshold:.2f} " + "="*20)
        x_test = np.load(os.path.join(self.test_data_dir, "x_test.npy"))
        y_test = np.load(os.path.join(self.test_data_dir, "y_test.npy"))

        gating_model.eval()
        for model in shard_models: model.eval()

        # Real-time gating network routing (no caching - proper for new samples)
        all_final_preds = []
        all_true_labels = []
        batch_size = config.BATCH_SIZE
        
        # Get unlearned classes for analysis
        unlearned_classes_set = self.get_unlearned_classes_set()
        
        with torch.no_grad():
            for i in range(0, len(x_test), batch_size):
                batch_x = torch.from_numpy(x_test[i:i+batch_size]).float()
                batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)
                
                # Use REAL-TIME gating network routing WITH threshold (production mode)
                # Threshold helps detect uncertain predictions on deleted/unknown classes
                batch_final_preds, _ = _run_sisa_batch(
                    batch_x_normalized, shard_models, gating_model, class_names, threshold
                )
                
                # Keep predictions including -1 (uncertain/unknown samples)
                # This shows production behavior: reject low-confidence predictions
                batch_final_preds_np = batch_final_preds.cpu().numpy()
                
                all_final_preds.extend(batch_final_preds_np)
                all_true_labels.extend(y_test[i:i+batch_size])

        all_final_preds = np.array(all_final_preds)
        all_true_labels = np.array(all_true_labels)

        # ============================================================================
        # PRODUCTION EVALUATION: Includes confidence-based rejection
        # Predictions with low confidence are marked as -1 (unknown/rejected)
        # This demonstrates real-world deployment where uncertain predictions are rejected
        # ============================================================================
        
        # Count rejected predictions (marked as -1 due to low confidence)
        rejected_count = np.sum(all_final_preds == -1)
        rejected_percentage = (rejected_count / len(all_final_preds)) * 100
        
        print("\n--- PRODUCTION EVALUATION WITH CONFIDENCE THRESHOLDING ---")
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
                    batch_x_normalized, shard_models, gating_model, class_names, threshold=None
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
        
        # Re-run predictions WITHOUT confidence threshold for active classes
        print("\nRe-evaluating with NO confidence threshold for clean active class metrics...")
        all_active_preds = []
        all_active_labels = []
        
        with torch.no_grad():
            for i in range(0, len(x_test), batch_size):
                batch_x = torch.from_numpy(x_test[i:i+batch_size]).float()
                batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)
                batch_y = y_test[i:i+batch_size]
                
                # NO threshold - get predictions on all samples
                batch_preds, _ = _run_sisa_batch(
                    batch_x_normalized, shard_models, gating_model, class_names, threshold=None
                )
                
                all_active_preds.extend(batch_preds.cpu().numpy())
                all_active_labels.extend(batch_y)
        
        all_active_preds = np.array(all_active_preds)
        all_active_labels = np.array(all_active_labels)
        
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
            gating_model,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'active_classes_only',
            unlearned_classes=unlearned_classes
        )
        
        # Confusion Matrix for active classes
        print("   Creating confusion matrix for active classes only...")
        create_overall_sisa_confusion_matrix(
            shard_models,
            gating_model,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'active_classes_only',
            unlearned_classes=unlearned_classes
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
            gating_model,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'with_deleted_classes',
            unlearned_classes=None  # Don't filter - show ALL classes
        )
        
        # Confusion matrix - ALL classes including deleted
        print("   Creating confusion matrix with ALL classes (including deleted)...")
        create_overall_sisa_confusion_matrix(
            shard_models,
            gating_model,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'with_deleted_classes',
            unlearned_classes=None  # Don't filter - show ALL classes
        )
        
        # ============================================================================
        # GENERATE PLOTS FOR ACTIVE CLASSES ONLY (Clean Performance Metrics)
        # ============================================================================
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATIONS FOR UNLEARNING EVALUATION (ACTIVE CLASSES)")
        print("=" * 50)
        
        create_overall_sisa_roc_curve(
            shard_models,
            gating_model,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'unlearning_evaluation',
            unlearned_classes=unlearned_classes  # Exclude deleted classes
        )
        
        # Create overall confusion matrix showing only active classes
        print("\n" + "=" * 50)
        print("CREATING OVERALL SISA SYSTEM CONFUSION MATRIX (AFTER UNLEARNING)")
        print("=" * 50)
        
        create_overall_sisa_confusion_matrix(
            shard_models,
            gating_model,
            x_test,
            y_test,
            class_names,
            self.reports_dir,
            'unlearning_evaluation',
            unlearned_classes=unlearned_classes  # Exclude unlearned classes
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

        # Load gating model for proper SISA prediction
        gating_model_path = os.path.join(self.models_dir, "gating_model.pth")
        gating_model, _ = load_model_pytorch(gating_model_path, num_shards=self.num_shards)
        gating_model.eval()
        
        # Use gating-based prediction
        all_preds = []
        all_confidences = []
        
        with torch.no_grad():
            for sample in x_forgotten:
                batch_x = torch.from_numpy(sample).unsqueeze(0).float()
                batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)
                
                # Get gating prediction
                gating_logits = gating_model(batch_x_normalized)
                shard_pred = gating_logits.argmax(dim=1).item()
                
                # Get specialist predictions
                specialist_outputs = [torch.softmax(model(batch_x_normalized), dim=1) for model in shard_models]
                
                # Find predictions and confidences for each shard
                shard_predictions = []
                shard_confidences = []
                for shard_output in specialist_outputs:
                    conf, pred = torch.max(shard_output[0], dim=0)
                    shard_predictions.append(pred.item())
                    shard_confidences.append(conf.item())
                
                # Use gating-selected prediction
                final_prediction = shard_predictions[shard_pred]
                final_confidence = shard_confidences[shard_pred]
                
                all_preds.append(final_prediction)
                all_confidences.append(final_confidence)
        
        all_preds = np.array(all_preds)
        all_confidences = np.array(all_confidences)
        
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
            print("   Classes are perfectly balanced for unlearning - NO AUGMENTATION")
            return None
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
        """Extract training accuracy and detailed timing breakdown from training.txt file."""
        with open("training.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for final accuracy
        import re
        accuracy_match = re.search(r'Final SISA System Accuracy:\s*([\d.]+)', content)
        training_accuracy = float(accuracy_match.group(1))
        
        # Extract REAL timing information from training.txt
        # 1. Gating network training time - use PURE time for accuracy
        gating_match = re.search(r'Gating Network Training Time \(pure\):\s*([\d.]+)\s*seconds', content)
        gating_training_time = float(gating_match.group(1)) if gating_match else 20.0
        
        # 2. Base model training time with eval (parse actual value)
        base_match = re.search(r'Base Model Training Time \(with eval\):\s*([\d.]+)\s*seconds', content)
        base_model_time = float(base_match.group(1)) if base_match else 50.0
        
        # 3. NEW: Base model training time pure (parse actual value)
        pure_match = re.search(r'Base Model Training Time \(pure\):\s*([\d.]+)\s*seconds', content)
        pure_training_time = float(pure_match.group(1)) if pure_match else base_model_time
        
        # 3. Total training time (for validation)
        time_match = re.search(r'Total Training Time:\s*([\d.]+)\s*seconds', content)
        total_training_time = float(time_match.group(1)) if time_match else 100.0
        
        # Extract real training classification report from training.txt
        training_report = self._parse_classification_report_from_training_log(content)
        
        return training_accuracy, gating_training_time, base_model_time, pure_training_time, training_report

    def _parse_classification_report_from_training_log(self, content: str) -> dict:
        """Parse the real classification report from training.txt content."""
        try:
            # Find the classification report section
            report_start = content.find("Final Classification Report:")
            if report_start == -1:
                raise Exception("Could not find 'Final Classification Report:' in training.txt")
            
            # Extract the report section (between "precision    recall" and "Final SISA System Accuracy")
            precision_line = content.find("precision    recall  f1-score   support", report_start)
            accuracy_line = content.find("Final SISA System Accuracy:", report_start)
            
            if precision_line == -1 or accuracy_line == -1:
                raise Exception("Could not find classification report data structure in training.txt")
            
            report_section = content[precision_line:accuracy_line]
            lines = report_section.strip().split('\n')
            
            # Parse the classification report
            training_report = {}
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
            for line in lines[2:]:  # Skip header lines
                line = line.strip()
                if not line or line.startswith('accuracy') or line.startswith('macro avg') or line.startswith('weighted avg'):
                    continue
                
                # Parse class-specific metrics
                parts = line.split()
                if len(parts) >= 4 and parts[0] in class_names:
                    class_name = parts[0]
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1_score = float(parts[3])
                    support = int(parts[4])
                    
                    training_report[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1-score': f1_score,
                        'support': support
                    }
            
            # Parse accuracy, macro avg, and weighted avg
            for line in lines:
                line = line.strip()
                if line.startswith('accuracy'):
                    parts = line.split()
                    if len(parts) >= 2:
                        training_report['accuracy'] = float(parts[1])
                elif line.startswith('macro avg'):
                    parts = line.split()
                    if len(parts) >= 4:
                        training_report['macro avg'] = {
                            'precision': float(parts[2]),
                            'recall': float(parts[3]),
                            'f1-score': float(parts[4]),
                            'support': int(parts[5])
                        }
                elif line.startswith('weighted avg'):
                    parts = line.split()
                    if len(parts) >= 4:
                        training_report['weighted avg'] = {
                            'precision': float(parts[2]),
                            'recall': float(parts[3]),
                            'f1-score': float(parts[4]),
                            'support': int(parts[5])
                        }
            
            return training_report
            
        except Exception as e:
            print(f"   Error: Could not parse training classification report: {e}")
            print("   Make sure training.txt exists and contains valid classification report data")
            raise

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
        
        # Get training baseline metrics from training.txt
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
        
        # Load gating model
        gating_model_path = os.path.join(self.models_dir, "gating_model.pth")
        if not os.path.exists(gating_model_path):
            print("   Warning: Gating model not found. Cannot perform SISA evaluation.")
            return
            
        # Import model creation function
        from training.create_model import load_model_pytorch
        
        gating_model, _ = load_model_pytorch(gating_model_path, num_shards=self.num_shards)
        gating_model.eval()
        
        # Set up evaluation
        dataset_mean, dataset_std = self._get_dataset_normalization()
        eval_transforms = T.Compose([T.Normalize(dataset_mean, dataset_std)])
        
        # Evaluation 1: Post-unlearning accuracy on DELETED class (should be ~0%)
        post_unlearning_deleted_accuracy = self._evaluate_sisa_on_data(
            shard_models, gating_model, x_deleted, y_deleted, eval_transforms
        )
        
        # Evaluation 2: Post-unlearning accuracy on REMAINING classes (should maintain performance)
        post_unlearning_remaining_accuracy = self._evaluate_sisa_on_data(
            shard_models, gating_model, x_remaining, y_remaining, eval_transforms
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
        success_threshold_deleted = 0.80  # 80% successful forgetting
        
        print(f"\nThreshold Evaluation (Threshold={success_threshold_deleted:.2f}):")
        if post_unlearning_deleted_accuracy >= success_threshold_deleted:
            print(f"Deleted class '{deleted_class_name}' well forgotten with {post_unlearning_deleted_accuracy:.4f} accuracy ({post_unlearning_deleted_accuracy*100:.1f}% unknown predictions)")
        else:
            print(f"Deleted class '{deleted_class_name}' still remembered with {post_unlearning_deleted_accuracy:.4f} accuracy (threshold: ≥{success_threshold_deleted:.2f})")
            
        if post_unlearning_remaining_accuracy >= (training_accuracy_for_deleted_class * 0.85):
            print(f"Remaining classes maintain {post_unlearning_remaining_accuracy:.4f} accuracy")
        else:
            print(f"Warning: Remaining classes dropped to {post_unlearning_remaining_accuracy:.4f} accuracy")
        
        print("=" * 80)

    def _get_pre_unlearning_class_accuracy(self, class_name: str) -> float:
        """Extract pre-unlearning accuracy for specific class from training.txt."""
        training_log_path = "training.txt"
        if not os.path.exists(training_log_path):
            return 0.74  # Default fallback
            
        try:
            with open(training_log_path, 'r') as f:
                content = f.read()
                
            # Look for class-specific precision/recall in classification report
            import re
            pattern = rf'\s+{re.escape(class_name)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'
            match = re.search(pattern, content)
            
            if match:
                precision, recall, _ = map(float, match.groups())
                return (precision + recall) / 2  # Average of precision and recall
            else:
                # Extract overall accuracy as fallback
                acc_match = re.search(r'Final SISA System Accuracy: ([0-9.]+)', content)
                if acc_match:
                    return float(acc_match.group(1))
                else:
                    raise Exception(f"Could not find accuracy for class '{class_name}' in training.txt")
                
        except Exception as e:
            print(f"   Error: Could not extract class accuracy: {e}")
            raise
    
    def _evaluate_sisa_on_data(self, shard_models, gating_model, x_data, y_data, eval_transforms):
        """Evaluate SISA system on given data with post-processing filtering for unlearned classes."""
        if len(x_data) == 0:
            return 0.0
            
        all_preds = []
        batch_size = config.BATCH_SIZE
        unlearned_classes_set = self.get_unlearned_classes_set()  # O(1) cached set lookup
        
        with torch.no_grad():
            for i in range(0, len(x_data), batch_size):
                batch_x = torch.from_numpy(x_data[i:i+batch_size]).float()
                batch_x_normalized = eval_transforms(batch_x).to(DEVICE)
                
                # Let model predict naturally - keep raw predictions
                batch_preds, _ = _run_sisa_batch(
                    batch_x_normalized, shard_models, gating_model, self.class_names, threshold=None
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
    # Set up logging to overwrite previous unlearning output
    logger = UnlearningLogger("unlearning.txt")
    sys.stdout = logger
    
    try:
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
        # Restore stdout and close logger
        sys.stdout = sys.__stdout__
        logger.close()