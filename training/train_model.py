import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import List, Optional

# Import global configuration
import config
from training.create_model import create_sisa_model, DEVICE
from training.early_stopping import SISAEarlyStopping
from training.augmentation import build_augmenter

# Import utility functions from plots.py (single source of truth)
from plots import (
    _normalize_probabilities_tensor,
    _apply_temperature_tensor,
    _apply_temperature_numpy,
    _run_sisa_batch,
    _scatter_local_to_global,
)

TRUE_LABEL_TITLE = 'True Label'
PREDICTED_LABEL_TITLE = 'Predicted Label'
SISA_METADATA_PATH = os.path.join(config.PROJECTS_DIR, config.PROJECT_NAME, "sisa_data", "metadata.json")


def _filter_data_by_class(X, y, active_classes):
    """Filters a dataset to only include samples from active_classes."""
    if active_classes is None:
        return X, y, None
    
    mask = np.isin(y, active_classes)
    label_map = {original_label: new_label for new_label, original_label in enumerate(sorted(active_classes))}
    
    y_filtered = y[mask]
    y_remapped = np.array([label_map[label] for label in y_filtered])
    
    return X[mask], y_remapped, label_map

def train_model(X, y, model=None, epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
                  validation_data=None, active_classes=None, replay_buffer=None, replay_ratio=0.2,
                  dataset_mean=None, dataset_std=None, training_type='fresh', augmentation_config=None,
                  device=None, head_classes=None):
    """
    Trains a SISA model with specialized validation and an optional replay mechanism.

    `head_classes` (W6, dynamic head): the fixed, sorted list of global class ids the
    model's output head is sized to. When None (default), the model keeps its original
    global-width head and every label stays in global-id space, exactly as before this
    parameter existed. When provided, a fresh model is created with `num_classes=len(head_classes)`,
    every training/replay label is remapped from global id to its position in
    `sorted(head_classes)` before the loss, and validation output is scattered back to
    global space so the existing `active_classes`-based column selection keeps working
    unchanged. `active_classes` continues to mean "classes visible/evaluated so far" and
    must always be a subset of `head_classes` when both are given.
    """
    head_label_map = None
    if head_classes is not None:
        head_label_map = {orig: new for new, orig in enumerate(sorted(head_classes))}

    def _remap_to_head(labels: np.ndarray) -> np.ndarray:
        """Map global class ids to local head-column indices (identity if no dynamic head)."""
        if head_label_map is None:
            return labels
        return np.array([head_label_map[int(v)] for v in labels], dtype=np.int64)

    if dataset_mean is None or dataset_std is None:
        print("Warning: Normalization stats not provided. Using default RGB values.")
        dataset_mean = [0.5, 0.5, 0.5]
        dataset_std = [0.5, 0.5, 0.5]

    # Build augmentation transforms based on configuration.
    # W19: geometric augmentation (crop + flip) is handled by PerSampleAugmenter so each
    # sample in a batch gets its own random draw -- applying a torchvision Compose to an
    # already-batched tensor gives the whole batch one shared decision. Colour jitter and
    # normalization stay in the Compose (jitter only appears on unbalanced-shard paths).
    augmenter = build_augmenter(augmentation_config, config.SEED)
    augmentations = []

    if augmentation_config is None:
        print("   - No augmentation applied")
    else:
        print(f"   - Using augmentation: {augmentation_config.get('reason', 'custom')}")

        if augmenter.crop_padding > 0:
            print(f"   - Random crop: padding={augmenter.crop_padding} (per-sample)")
        if augmenter.flip_prob > 0:
            print(f"   - Horizontal flip: {augmenter.flip_prob:.2f} (per-sample)")

        # Color jitter (only if any parameter > 0)
        color_params = [
            augmentation_config.get('color_jitter_brightness', 0),
            augmentation_config.get('color_jitter_contrast', 0),
            augmentation_config.get('color_jitter_saturation', 0),
            augmentation_config.get('color_jitter_hue', 0)
        ]
        if any(p > 0 for p in color_params):
            augmentations.append(T.ColorJitter(
                brightness=color_params[0],
                contrast=color_params[1],
                saturation=color_params[2],
                hue=color_params[3]
            ))
            print(f"   - Color jitter: brightness={color_params[0]:.2f}, contrast={color_params[1]:.2f}")

    # Always add normalization at the end
    augmentations.append(T.Normalize(dataset_mean, dataset_std))

    train_transforms = T.Compose(augmentations)
    val_transforms = T.Compose([
        T.Normalize(dataset_mean, dataset_std)
    ])
    
    if validation_data:
        x_val_full, y_val_full = validation_data
        x_val, y_val_remapped, label_map = _filter_data_by_class(x_val_full, y_val_full, active_classes)
        print(f"   - Validating on {len(x_val)} samples from {len(active_classes)} active classes.")
        if len(x_val) == 0:
            print(f"   âš ï¸  WARNING: No validation samples found for specialist classes {active_classes}")
            print(f"   ðŸ“Š Available classes in validation: {sorted(np.unique(y_val_full))}")
            # Create empty validation tensors to avoid errors
            x_val = np.empty((0, *x_val_full.shape[1:]))
            y_val_remapped = np.empty(0, dtype=np.int64)
    else:
        x_train, x_val, y_train, y_val_original = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X, y = x_train, y_train
        label_map = {original_label: new_label for new_label, original_label in enumerate(sorted(active_classes))}
        y_val_remapped = np.array([label_map[label] for label in y_val_original])
        print(f"   - Created internal validation set with {len(x_val)} samples.")

    x_train_t = torch.from_numpy(X.astype(np.float32))
    y_train_t = torch.from_numpy(_remap_to_head(y.astype(np.int64))).long()

    x_val_t = torch.from_numpy(x_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val_remapped.astype(np.int64)).long()

    if model is None:
        model = create_sisa_model(num_classes=len(head_classes) if head_classes is not None else None)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
    
    # Use appropriate patience based on training type
    if training_type == 'unlearning':
        patience = config.UNLEARNING_PATIENCE
        min_delta = config.UNLEARNING_MIN_DELTA
    else:
        patience = config.TRAINING_PATIENCE
        min_delta = config.TRAINING_MIN_DELTA
        
    # W20: with fewer than 2 active classes the filtered validation output has a single
    # column, so softmax is trivially 1.0 and cross-entropy is exactly 0.0 on every
    # epoch. Early stopping can never improve on zero, so it used to restore the epoch-0
    # weights and silently discard the whole run (shard 1's single-class slice 1 did
    # exactly this). Fall back to the training loss, which still carries signal.
    num_active = len(active_classes) if active_classes is not None else 0
    degenerate_validation = num_active < 2 or len(x_val_t) == 0
    if degenerate_validation:
        print(f"   - Degenerate validation ({num_active} active class(es), {len(x_val_t)} samples): "
              f"early stopping will monitor training loss instead")

    early_stopping = SISAEarlyStopping(
        patience=patience,
        min_delta=min_delta,
        monitor='train_loss' if degenerate_validation else 'val_loss',
        mode='min',
        restore_best_weights=True,
        verbose=True
    )

    # Simple loss and optimization - Adam handles adaptive learning rates
    # CRITICAL: Use 0.0 label smoothing during unlearning to allow deleted class neurons to die
    label_smoothing_value = config.UNLEARNING_LABEL_SMOOTHING if training_type == 'unlearning' else config.LABEL_SMOOTHING
    print(f"   - Using label smoothing: {label_smoothing_value:.3f} ({'unlearning mode' if training_type == 'unlearning' else 'normal training'})")
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)  # Simple Adam optimizer

    if replay_buffer:
        num_replay_samples = sum(len(d['y']) for d in replay_buffer.values())
        print(f"   - Using replay buffer with {num_replay_samples} previous samples from {len(replay_buffer)} classes.")

    rng = np.random.default_rng(config.SEED)  # Deterministic, derived from the global seed (W1)
    
    # W18: the current-slice portion of each batch is also the loop stride. Previously
    # the loop advanced by `batch_size` while consuming only the first `main_batch_size`
    # indices of each chunk, silently dropping the remainder every epoch -- 31% of a
    # slice at the old fixed 0.3 ratio, and ~72% at the class-balanced ratios W18
    # produces, which would have undercut the very fix it delivers. Striding by
    # `main_batch_size` means every current-slice sample is used exactly once per epoch,
    # with replay topping each batch up to `batch_size`.
    uses_replay = bool(replay_buffer) and replay_ratio > 0
    main_batch_size = max(1, int(batch_size * (1 - replay_ratio))) if uses_replay else batch_size
    replay_batch_size = batch_size - main_batch_size if uses_replay else 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        num_steps = 0
        indices = rng.permutation(len(X))

        for i in range(0, len(X), main_batch_size):
            batch_indices = indices[i:i+main_batch_size]
            if len(batch_indices) == 0:
                continue
            num_steps += 1

            if uses_replay:
                main_x = x_train_t[batch_indices]
                main_y = y_train_t[batch_indices]

                available_classes = list(replay_buffer.keys())

                if available_classes and replay_batch_size > 0:
                    replay_x_list, replay_y_list = [], []
                    chosen_classes = rng.choice(available_classes, replay_batch_size)
                    for class_idx in chosen_classes:
                        class_data = replay_buffer[class_idx]
                        sample_idx = rng.integers(0, len(class_data['X']))
                        replay_x_list.append(class_data['X'][sample_idx])
                        replay_y_list.append(class_data['y'][sample_idx])

                    replay_x = torch.from_numpy(np.array(replay_x_list).astype(np.float32))
                    replay_y = torch.from_numpy(_remap_to_head(np.array(replay_y_list).astype(np.int64))).long()

                    batch_x = torch.cat((main_x, replay_x))
                    batch_y = torch.cat((main_y, replay_y))
                else:
                    batch_x = main_x
                    batch_y = main_y
            else:
                batch_x = x_train_t[batch_indices]
                batch_y = y_train_t[batch_indices]

            # Per-sample geometric augmentation first (W19), then colour jitter + normalize.
            batch_x = augmenter(batch_x)
            batch_x = train_transforms(batch_x).to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            
            # Simple forward pass without mixed precision
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / num_steps if num_steps > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        if len(x_val_t) > 0:
            with torch.no_grad():
                for i in range(0, len(x_val_t), batch_size):
                    batch_x_val = x_val_t[i:i+batch_size]
                    batch_y_val = y_val_t[i:i+batch_size]

                    batch_x_val = val_transforms(batch_x_val).to(DEVICE)
                    batch_y_val = batch_y_val.to(DEVICE)
                    
                    output_full = model(batch_x_val)
                    if head_classes is not None:
                        # Reduced/dynamic head: scatter local head columns back to global
                        # class space so the active_classes column selection below is valid.
                        output_full = _scatter_local_to_global(
                            output_full, sorted(head_classes), config.get_num_classes(), DEVICE
                        )

                    # Specialist-only validation (filtered to active classes)
                    active_class_indices = torch.tensor(sorted(active_classes), device=DEVICE)
                    output_filtered = output_full[:, active_class_indices]

                    loss = criterion(output_filtered, batch_y_val)
                    val_loss += loss.item()
                    _, predicted_filtered = torch.max(output_filtered.data, 1)
                    val_total += batch_y_val.size(0)
                    val_correct += (predicted_filtered == batch_y_val).sum().item()

        val_epoch_loss = val_loss / (len(x_val_t) / batch_size) if len(x_val_t) > 0 else 0
        val_epoch_acc = val_correct / val_total if val_total > 0 else 0
        
        # No lr_scheduler needed - Adam handles adaptive learning rates
        history['loss'].append(epoch_loss); history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss); history['val_accuracy'].append(val_epoch_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f'   Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        monitored_loss = epoch_loss if degenerate_validation else val_epoch_loss
        if early_stopping(monitored_loss, model, epoch):
            print("   - Early stopping triggered.")
            break
            
    early_stopping.restore_best_model(model)
    return model, history