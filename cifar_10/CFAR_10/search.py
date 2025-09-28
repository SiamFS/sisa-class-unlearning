import sys
import os
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import global configuration
import config

from training.create_model import load_model_pytorch, DEVICE

class SISASearchTool:
    def __init__(self, project_name: str, model_name: str):
        print(f"Initializing SISA Search Tool for project: '{project_name}' with model: '{model_name}'")
        self.project_name = project_name
        self.model_name = model_name
        self.base_dir = f"../{config.PROJECTS_DIR}/{self.project_name}"
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "sisa_data")
        self.reports_dir = os.path.join(self.base_dir, "data_info")
        
        # Load metadata first to get class names and normalization
        self.metadata = self._load_metadata()
        
        # Load class names dynamically from metadata
        if 'class_names' in self.metadata:
            self.class_names = self.metadata['class_names']
            print(f"Loaded {len(self.class_names)} class names from metadata: {self.class_names}")
        else:
            print("Warning: Class names not found in metadata. Using CIFAR-10 defaults.")
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
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
        test_data = np.load(x_test_path)
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
        """Load all shard models and gating model"""
        print("Loading SISA models...")
        
        # Load shard models
        shard_models = []
        num_shards = self.metadata.get('num_shards', 2)  # Default to 2 shards
        
        for i in range(num_shards):
            model_path = os.path.join(self.models_dir, f"shard_{i+1}", f"final_model_shard{i+1}_{self.model_name}.pth")
            
            if os.path.exists(model_path):
                model, _ = load_model_pytorch(model_path)
                shard_models.append(model.eval())
                print(f"   - Loaded shard {i+1} model")
            else:
                print(f"   - Warning: No model found for shard {i+1}")

        # Load gating model
        gating_model_path = os.path.join(self.models_dir, "gating_model.pth")
        gating_model = None
        if os.path.exists(gating_model_path):
            gating_model, _ = load_model_pytorch(gating_model_path, num_shards=num_shards)
            gating_model = gating_model.eval()
            print("   - Loaded gating model")
        else:
            print("   - Warning: Gating model not found")
            
        return shard_models, gating_model

    def search_class_predictions(self, class_name: str, num_samples: int = config.DEFAULT_SEARCH_SAMPLES, threshold: float = config.CONFIDENCE_THRESHOLD):
        """Search for predictions on a specific class with visualization"""
        
        if class_name not in self.class_names:
            raise ValueError(f"Class '{class_name}' not found. Available classes: {self.class_names}")
        
        class_idx = self.class_names.index(class_name)
        print(f"\n" + "="*70)
        print(f"SEARCHING FOR CLASS: '{class_name.upper()}' PREDICTIONS")
        print("="*70)
        print(f"Using confidence threshold: {threshold}")
        print("Searching in current SISA test dataset...")
        
        # Load models
        shard_models, gating_model = self._load_models()
        
        if not shard_models or not gating_model:
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
        
        # Make predictions
        predictions, confidences, detailed_results = self._predict_with_gating(selected_samples, shard_models, gating_model, threshold)
        
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
            detail = detailed_results[i]
            
            if pred_idx == -1:  # Unknown prediction
                pred_label = "UNKNOWN"
                unknown_predictions += 1
            else:
                pred_label = self.class_names[pred_idx]
                if pred_idx == selected_labels[i]:
                    correct_predictions += 1
            
            status = "CORRECT" if pred_label == true_label else "WRONG"
            print(f"Sample {i+1:2d}: True={true_label:10s} | Pred={pred_label:10s} | Conf={confidence:.3f} {status}")
        
        # Show detailed analysis for all samples
        print("-"*50)
        print("DETAILED ANALYSIS:")
        print("-"*50)
        
        for i in range(samples_to_analyze):
            detail = detailed_results[i]
            true_label = self.class_names[selected_labels[i]]
            pred_idx = predictions[i]
            
            status = "CORRECT" if pred_idx == selected_labels[i] else ("UNKNOWN" if pred_idx == -1 else "WRONG")
            pred_label = self.class_names[pred_idx] if pred_idx != -1 else "UNKNOWN"
            
            print(f"\nSample {detail['sample_idx']}: {true_label} → {pred_label} {status}")
            print(f"  Method used: {detail['final_method'].upper()}")
            
            if 'gating_confidence' in detail:
                print(f"  Gating confidence: {detail['gating_confidence']:.3f}")
            
            # Dynamic gating weights display based on actual number of shards
            gating_weights_str = ", ".join([f"Shard{i+1}={weight:.3f}" for i, weight in enumerate(detail['gating_weights'])])
            print(f"  Gating weights: {gating_weights_str}")
            print("  Individual shard predictions:")
            
            for shard_idx, (shard_pred, shard_conf) in enumerate(detail['shard_predictions']):
                gating_marker = "→" if shard_idx == (detail['gating_selected_shard'] - 1) else " "
                override_marker = "*" if detail.get('override_shard') == shard_idx + 1 else " "
                marker = override_marker if override_marker != " " else gating_marker
                print(f"    {marker} Shard {shard_idx+1}: {shard_pred} ({shard_conf})")
                
            if detail.get('override_shard'):
                print(f"  Override: Used Shard {detail['override_shard']} instead of gating choice")
                if detail.get('override_reason'):
                    print(f"  Reason: {detail['override_reason']}")
            
            print(f"  Final confidence: {detail['final_confidence']:.3f}")
        
        accuracy = correct_predictions / samples_to_analyze
        unknown_rate = unknown_predictions / samples_to_analyze
        
        print("-"*50)
        print(f"Accuracy on '{class_name}': {correct_predictions}/{samples_to_analyze} = {accuracy:.2%}")
        print(f"Unknown rate: {unknown_predictions}/{samples_to_analyze} = {unknown_rate:.2%}")
        
        if class_name == 'cat':  # Special analysis for deleted class
            if accuracy < config.UNLEARNING_SUCCESS_THRESHOLD and unknown_rate > 0.5:
                print(">>> SUCCESS: Model appears to have successfully unlearned this class!")
            elif unknown_rate > 0.3:
                print(">>> PARTIAL: Model shows some unlearning effects.")
            else:
                print(">>> WARNING: Model may not have properly unlearned this class.")
        
        # Create visualization
        self._create_prediction_visualization(selected_samples, selected_labels, predictions, confidences, class_name, threshold)
        
        print(f"\nVisualization saved to: {os.path.join(self.reports_dir, f'Class_Search_{class_name}_Analysis.png')}")
        print("="*70)

    def _predict_with_gating(self, samples, shard_models, gating_model, threshold):
        """Make predictions using an improved gating network approach with ensemble fallback"""
        predictions = []
        confidences = []
        detailed_results = []
        
        with torch.no_grad():
            for idx, sample in enumerate(samples):
                # Convert single sample to batch format
                batch_x = torch.from_numpy(sample).unsqueeze(0).float()
                batch_x_normalized = self.eval_transforms(batch_x).to(DEVICE)
                
                # Get gating prediction
                gating_logits = gating_model(batch_x_normalized)
                gating_probs = torch.softmax(gating_logits, dim=1)
                shard_pred = gating_logits.argmax(dim=1).item()
                
                # Get specialist predictions for all shards
                specialist_outputs = [torch.softmax(model(batch_x_normalized), dim=1) for model in shard_models]
                
                # Analyze what each shard predicts
                shard_predictions = []
                shard_confidences = []
                for shard_idx, shard_output in enumerate(specialist_outputs):
                    conf, pred = torch.max(shard_output[0], dim=0)
                    shard_predictions.append(pred.item())
                    shard_confidences.append(conf.item())
                
                # IMPROVED DECISION LOGIC:
                # 1. Check if gating selection is confident AND the selected shard is confident
                gating_confidence = gating_probs[0, shard_pred].item()
                selected_shard_confidence = shard_confidences[shard_pred]
                
                # Store detailed analysis
                detailed_info = {
                    'sample_idx': idx + 1,
                    'gating_selected_shard': shard_pred + 1,
                    'gating_confidence': gating_confidence,
                    'gating_weights': gating_probs[0].cpu().numpy().tolist(),
                    'shard_predictions': [(self.class_names[pred], f"{conf:.3f}") for pred, conf in zip(shard_predictions, shard_confidences)],
                }
                
                final_prediction = None
                final_confidence = 0.0
                method_used = "unknown"
                
                # Find the most confident shard (but only consider valid class predictions)
                best_shard = -1
                best_confidence = 0.0
                
                # Check each shard and only consider predictions for classes they were trained on
                for shard_idx, (pred_class, confidence) in enumerate(zip(shard_predictions, shard_confidences)):
                    # Only consider this shard if it predicts a class it was trained on
                    if pred_class in self.shard_classes.get(shard_idx, []):
                        if confidence > best_confidence:
                            best_shard = shard_idx
                            best_confidence = confidence
                
                # Strategy 1: Use best shard if it's significantly more confident AND predicting valid class
                if (best_shard != -1 and 
                    best_confidence >= threshold and 
                    best_confidence > selected_shard_confidence + config.CONFIDENCE_BOOST_THRESHOLD and
                    shard_predictions[best_shard] in self.shard_classes.get(best_shard, [])):
                    
                    final_prediction = shard_predictions[best_shard]
                    final_confidence = best_confidence
                    method_used = "best_shard"
                    detailed_info['override_shard'] = best_shard + 1
                    detailed_info['override_reason'] = f"Better confidence: {best_confidence:.3f} vs {selected_shard_confidence:.3f} (valid class)"
                
                # Strategy 2: Use gating if selected shard is confident enough AND predicting valid class
                elif (selected_shard_confidence >= threshold and 
                      shard_predictions[shard_pred] in self.shard_classes.get(shard_pred, [])):
                    final_prediction = shard_predictions[shard_pred]
                    final_confidence = selected_shard_confidence
                    method_used = "gating"
                
                # If gating confidence is too low, mark as uncertain but still use selected shard
                else:
                    final_prediction = shard_predictions[shard_pred]
                    final_confidence = selected_shard_confidence
                    method_used = "gating_uncertain"
                
                # Store results
                predictions.append(final_prediction)
                confidences.append(final_confidence)
                
                detailed_info['final_method'] = method_used
                detailed_info['final_prediction'] = self.class_names[final_prediction] if final_prediction != -1 else 'UNKNOWN'
                detailed_info['final_confidence'] = final_confidence
                detailed_results.append(detailed_info)
        
        return predictions, confidences, detailed_results

    def _create_prediction_visualization(self, samples, true_labels, predictions, confidences, class_name, threshold):
        """Create a 4x4 grid visualization similar to unlearning verification"""
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f"Class Search Analysis: '{class_name.upper()}' (Threshold={threshold})", fontsize=16, fontweight='bold')
        
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
        
        print(f"\nAnalysis completed successfully!")
        print(f"Check the generated visualization in: ../projects/{args.project}/data_info/")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
