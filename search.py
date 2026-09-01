import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import global configuration
import config
from utils.data_io import load_images

from training.create_model import load_model_pytorch, DEVICE
from plots import _run_sisa_batch  # Use TRUE SISA routing logic

class SISASearchTool:
    def __init__(self, project_name: str, model_name: str):
        print(f"Initializing SISA Search Tool for project: '{project_name}' with model: '{model_name}'")
        self.project_name = project_name
        self.model_name = model_name
        self.base_dir = os.path.join(config.PROJECTS_DIR, self.project_name)
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "sisa_data")
        self.reports_dir = os.path.join(self.base_dir, "data_info")
        
        # Load metadata first to get class names and normalization
        self.metadata = self._load_metadata()

        # Load class names dynamically from metadata (no dataset-specific fallback:
        # silently defaulting to CIFAR-10 labels would mislabel any other dataset)
        if 'class_names' not in self.metadata:
            raise KeyError(
                "metadata.json is missing 'class_names'. Re-run data_processing/entry_data_processing.py first."
            )
        self.class_names = self.metadata['class_names']
        print(f"Loaded {len(self.class_names)} class names from metadata: {self.class_names}")
        
        # Load normalization stats dynamically from metadata
        if 'normalization_mean' in self.metadata and 'normalization_std' in self.metadata:
            self.dataset_mean = self.metadata['normalization_mean']
            self.dataset_std = self.metadata['normalization_std']
            print("Loaded dynamic normalization stats from metadata.")
        else:
            print("Warning: Normalization stats not found. Using neutral fallback values.")
            self.dataset_mean = [0.5, 0.5, 0.5]  # Neutral fallback for RGB
            self.dataset_std = [0.5, 0.5, 0.5]   # Neutral fallback for RGB

        self.eval_transforms = T.Compose([
            T.Normalize(self.dataset_mean, self.dataset_std)
        ])
        
        # Load current SISA test data (the actual dataset used by models)
        self.test_data = self._load_sisa_test_data()
        
        # Load shard class constraints
        self.shard_classes = self._load_shard_class_constraints()
        
        # Load unlearned classes to determine active classes
        self.unlearned_classes = self._load_unlearned_classes()
        self.active_classes = [i for i in range(len(self.class_names)) if i not in self.unlearned_classes]
        
        print(f"Unlearned classes: {[self.class_names[i] for i in self.unlearned_classes]}")
        print(f"Active classes: {[self.class_names[i] for i in self.active_classes]}")
        
    def _load_metadata(self) -> Dict:
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found at {metadata_path}")
            return {}
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _load_sisa_test_data(self):
        """Load current SISA test data (processed dataset used by models)"""
        print("Loading current SISA test data...")
        
        x_test_path = os.path.join(self.data_dir, "test_data", "x_test.npy")
        y_test_path = os.path.join(self.data_dir, "test_data", "y_test.npy")
        
        if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
            raise FileNotFoundError(f"SISA test data not found at {x_test_path} or {y_test_path}")
        
        # Load the processed test data
        test_data = load_images(x_test_path)
        test_labels = np.load(y_test_path)
        
        print(f"Loaded {len(test_data)} current test samples")
        print(f"Available classes: {sorted(np.unique(test_labels))}")
        return test_data, test_labels

    def _load_shard_class_constraints(self):
        """Load which classes each shard is responsible for"""
        shard_classes = {}
        num_shards = self.metadata.get('num_shards', 2)
        
        for shard_idx in range(num_shards):
            shard_metadata_path = os.path.join(self.data_dir, "shards", f"shard_{shard_idx+1}", "metadata.json")
            if os.path.exists(shard_metadata_path):
                with open(shard_metadata_path, 'r') as f:
                    shard_meta = json.load(f)
                    shard_classes[shard_idx] = shard_meta.get('class_indices_present', [])
            else:
                print(f"Warning: Shard {shard_idx+1} metadata not found")
                shard_classes[shard_idx] = []
        
        print(f"Loaded shard class constraints:")
        for shard_idx, classes in shard_classes.items():
            class_names = [self.class_names[i] for i in classes]
            print(f"   - Shard {shard_idx+1}: {class_names}")
        
        return shard_classes

    def _load_unlearned_classes(self) -> List[int]:
        """Load the list of unlearned class indices"""
        unlearned_classes_file = os.path.join(self.models_dir, "unlearned_classes.json")
        if os.path.exists(unlearned_classes_file):
            with open(unlearned_classes_file, 'r') as f:
                return json.load(f)
        return []

    def _load_models(self):
        """Load all shard models (self-routing: no gating network needed)."""
        print("Loading SISA models...")

        num_shards = self.metadata.get('num_shards', 2)  # Default to 2 shards
        # Positional list (index i == shard i+1); missing models stay None so
        # they line up with self.shard_classes / shard_class_indices by index.
        shard_models = [None] * num_shards

        for i in range(num_shards):
            model_path = os.path.join(self.models_dir, f"shard_{i+1}", f"final_model_shard{i+1}_{self.model_name}.pth")

            if os.path.exists(model_path):
                model, _ = load_model_pytorch(model_path)
                shard_models[i] = model.eval()
                print(f"   - Loaded shard {i+1} model")
            else:
                print(f"   - Warning: No model found for shard {i+1}")

        return shard_models

    def _load_gating_model(self):
        """W16: the gating network is the real router now if one exists for this
        project; falls back to confidence-based self-routing (gating_model=None)
        for older projects trained before this switch."""
        num_shards = self.metadata.get('num_shards', 2)
        gating_path = os.path.join(self.models_dir, "gating_model.pth")
        if os.path.exists(gating_path):
            gating_model, _ = load_model_pytorch(gating_path, num_shards=num_shards)
            gating_model.eval()
            print("   - Loaded gating network (real router)")
            return gating_model
        print("   - No gating network found; falling back to confidence-based self-routing")
        return None

    def search_class_predictions(self, class_name: str, num_samples: int = config.DEFAULT_SEARCH_SAMPLES, threshold: float = config.CONFIDENCE_THRESHOLD):
        """Search for predictions on a specific class with visualization"""
        
        if class_name not in self.class_names:
            raise ValueError(f"Class '{class_name}' not found. Available classes: {self.class_names}")
        
        class_idx = self.class_names.index(class_name)
        print(f"\n" + "="*70)
        print(f"SEARCHING FOR CLASS: '{class_name.upper()}' PREDICTIONS")
        print("="*70)
        print("Searching in current SISA test dataset...")
        
        # Load models
        shard_models = self._load_models()
        gating_model = self._load_gating_model()

        if not shard_models or all(model is None for model in shard_models):
            print("ERROR: Could not load required models!")
            return
        
        # Get samples of the target class from current test data
        test_data, test_labels = self.test_data
        class_mask = (test_labels == class_idx)
        class_samples = test_data[class_mask]
        class_labels = test_labels[class_mask]
        
        if len(class_samples) == 0:
            print(f"No samples found for class '{class_name}' in current test dataset!")
            print("This likely means the class has been unlearned and removed from test data.")
            print(f"Available classes in current test set: {[self.class_names[i] for i in sorted(np.unique(test_labels))]}")
            return
        
        # Take first num_samples
        samples_to_analyze = min(num_samples, len(class_samples))
        selected_samples = class_samples[:samples_to_analyze]
        selected_labels = class_labels[:samples_to_analyze]
        
        print(f"Found {len(class_samples)} total '{class_name}' samples in current test set")
        print(f"Analyzing first {samples_to_analyze} samples...")
        
        # Make predictions using TRUE SISA self-routing (same as training/unlearning evaluation)
        num_shards = self.metadata.get('num_shards', 2)
        shard_class_indices = [self.shard_classes.get(i, []) for i in range(num_shards)]
        predictions, confidences = self._predict_with_true_sisa_batch(selected_samples, shard_models, shard_class_indices, threshold, gating_model)
        
        # Calculate accuracy
        correct_predictions = 0
        unknown_predictions = 0
        
        print(f"\n" + "-"*50)
        print("PREDICTION RESULTS:")
        print("-"*50)
        
        for i in range(samples_to_analyze):
            true_label = self.class_names[selected_labels[i]]
            pred_idx = predictions[i]
            confidence = confidences[i]
            
            if pred_idx == -1:  # Unknown prediction
                pred_label = "UNKNOWN"
                unknown_predictions += 1
            else:
                pred_label = self.class_names[pred_idx]
                if pred_idx == selected_labels[i]:
                    correct_predictions += 1
            
            status = "✓ CORRECT" if pred_idx == selected_labels[i] else ("? UNKNOWN" if pred_idx == -1 else "✗ WRONG")
            print(f"Sample {i+1:2d}: True={true_label:10s} | Pred={pred_label:10s} | Conf={confidence:.3f} | {status}")
        
        # Remove old detailed analysis section - using TRUE SISA routing now
        accuracy = correct_predictions / samples_to_analyze
        unknown_rate = unknown_predictions / samples_to_analyze
        
        print("-"*50)
        print(f"Accuracy on '{class_name}': {correct_predictions}/{samples_to_analyze} = {accuracy:.2%}")
        print(f"Unknown rate: {unknown_predictions}/{samples_to_analyze} = {unknown_rate:.2%}")
        
        if class_idx in self.unlearned_classes:  # Deleted class
            if accuracy < config.UNLEARNING_SUCCESS_THRESHOLD and unknown_rate > 0.5:
                print("✓ Model has successfully unlearned this class")
            elif unknown_rate > 0.3:
                print("⚠ Model shows some unlearning effects")
            else:
                print("✗ Warning: Model may not have properly unlearned this class")
        
        # Create visualization
        self._create_prediction_visualization(selected_samples, selected_labels, predictions, confidences, class_name)
        
        print(f"\nVisualization saved to: {os.path.join(self.reports_dir, f'Class_Search_{class_name}_Analysis.png')}")
        print("="*70)

    def _predict_with_true_sisa_batch(self, samples, shard_models, shard_class_indices, threshold, gating_model=None):
        """Use TRUE SISA routing (_run_sisa_batch) for consistency with evaluation --
        gating-network routing (W16) if available, confidence-based self-routing otherwise."""
        predictions = []
        confidences = []

        with torch.no_grad():
            for sample in samples:
                # Convert single sample to batch format
                batch_x = torch.from_numpy(sample).unsqueeze(0).float()
                batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)

                # Use TRUE SISA routing (same as evaluation)
                batch_preds, batch_probs = _run_sisa_batch(
                    batch_x_normalized,
                    shard_models,
                    self.class_names,
                    shard_class_indices,
                    threshold,
                    gating_model=gating_model,
                )
                
                pred = batch_preds[0].item()
                # Get confidence for predicted class (or max prob if prediction is -1)
                if pred == -1:
                    conf = 0.0
                else:
                    conf = batch_probs[0, pred].item() if pred < batch_probs.size(1) else 0.0
                
                predictions.append(pred)
                confidences.append(conf)
        
        return predictions, confidences

    def _create_prediction_visualization(self, samples, true_labels, predictions, confidences, class_name):
        """Create a 4x4 grid visualization similar to unlearning verification"""
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f"Class Search Analysis: '{class_name.upper()}' ", fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            if i >= len(samples):
                ax.axis('off')
                continue
                
            # Display image
            img = samples[i].transpose((1, 2, 0))
            ax.imshow(img)
            
            # Get labels and colors
            true_label = self.class_names[true_labels[i]]
            pred_idx = predictions[i]
            confidence = confidences[i]
            
            if pred_idx == -1:
                pred_label = "UNKNOWN"
                color = "orange"
            else:
                pred_label = self.class_names[pred_idx]
                color = "green" if pred_label == true_label else "red"
            
            # Set title with confidence
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}"
            ax.set_title(title, color=color, fontsize=9, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs(self.reports_dir, exist_ok=True)
        save_path = os.path.join(self.reports_dir, f'Class_Search_{class_name}_Analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='SISA Class Search and Prediction Analysis Tool')
    parser.add_argument('--project', type=str, default=config.PROJECT_NAME, help='Name of the project directory.')
    parser.add_argument('--model', type=str, default=config.MODEL_TYPE, help='Model architecture name.')
    parser.add_argument('--class-name', type=str, required=True, help='Name of the class to analyze (e.g., "cat", "dog").')
    parser.add_argument('--samples', type=int, default=config.DEFAULT_SEARCH_SAMPLES, help=f'Number of samples to analyze (default: {config.DEFAULT_SEARCH_SAMPLES}, max: 16 for visualization).')
    parser.add_argument('--threshold', type=float, default=config.CONFIDENCE_THRESHOLD, help=f'Confidence threshold for predictions (default: {config.CONFIDENCE_THRESHOLD}).')
    
    args = parser.parse_args()
    
    # Limit samples for visualization
    if args.samples > config.DEFAULT_SEARCH_SAMPLES:
        print(f"Warning: Limiting samples to {config.DEFAULT_SEARCH_SAMPLES} for visualization (requested: {args.samples})")
        args.samples = config.DEFAULT_SEARCH_SAMPLES
    
    print("SISA Class Search and Prediction Analysis Tool")
    print("=" * 50)
    print(f"Target class: {args.class_name}")
    print(f"Samples to analyze: {args.samples}")
    print(f"Confidence threshold: {args.threshold}")
    print("=" * 50)
    
    try:
        search_tool = SISASearchTool(project_name=args.project, model_name=args.model)
        search_tool.search_class_predictions(args.class_name, args.samples, args.threshold)
        
        print("\nAnalysis completed")
        print(f"Check the generated visualization in: {os.path.join(config.PROJECTS_DIR, args.project, 'data_info')}")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
