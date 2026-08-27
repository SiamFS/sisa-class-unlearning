import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

# Import global configuration
import config
from training.create_model import create_sisa_model, DEVICE
from training.early_stopping import SISAEarlyStopping

# Import utility functions from plots.py (single source of truth)
from plots import (
    _normalize_probabilities_tensor,
    _apply_temperature_tensor, 
    _run_sisa_batch
)



def _filter_data_by_class(X, y, active_classes):
    """Filter a dataset to active_classes, keeping labels in the original space so
    validation scores the same objective training optimises."""
    if active_classes is None:
        return X, y

    mask = np.isin(y, active_classes)
    return X[mask], y[mask]

def train_model(X, y, model=None, epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, 
                  validation_data=None, active_classes=None, replay_buffer=None, replay_ratio=0.2,
                  dataset_mean=None, dataset_std=None, training_type='fresh', augmentation_config=None,
                  use_smart_replay=False, device=None):
    """
    Trains a SISA model with specialized validation and an optional replay mechanism.
    """
    if dataset_mean is None or dataset_std is None:
        print("Warning: Normalization stats not provided. Using default RGB values.")
        dataset_mean = [0.5, 0.5, 0.5]
        dataset_std = [0.5, 0.5, 0.5]

    # Build augmentation transforms based on configuration
    augmentations = []
    
    if augmentation_config is None:
        # NO AUGMENTATION - Classes are balanced
        print("   - No augmentation applied (balanced classes)")
    else:
        # Apply augmentation for imbalanced classes
        print(f"   - Using augmentation: {augmentation_config.get('reason', 'custom')}")
        
        # Horizontal flip
        if augmentation_config.get('random_horizontal_flip', 0) > 0:
            augmentations.append(T.RandomHorizontalFlip(p=augmentation_config['random_horizontal_flip']))
            print(f"   - Horizontal flip: {augmentation_config.get('random_horizontal_flip', 0):.1f}")
        
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
        x_val, y_val = _filter_data_by_class(x_val_full, y_val_full, active_classes)
        print(f"   - Validating on {len(x_val)} samples from {len(active_classes)} active classes.")
        if len(x_val) == 0:
            print(f"   WARNING: No validation samples found for specialist classes {active_classes}")
            print(f"   Available classes in validation: {sorted(np.unique(y_val_full))}")
            # Create empty validation tensors to avoid errors
            x_val = np.empty((0, *x_val_full.shape[1:]))
            y_val = np.empty(0, dtype=np.int64)
    else:
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X, y = x_train, y_train
        print(f"   - Created internal validation set with {len(x_val)} samples.")

    x_train_t = torch.from_numpy(X.astype(np.float32))
    y_train_t = torch.from_numpy(y.astype(np.int64)).long()

    x_val_t = torch.from_numpy(x_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).long()
    
    if model is None:
        model = create_sisa_model()
        if training_type == 'fresh':
            pass  # Already set as parameter default
    # Note: training_type is now passed as parameter, no need to reassign

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
    
    # Use appropriate patience based on training type
    if training_type == 'unlearning':
        patience = config.UNLEARNING_PATIENCE
        min_delta = config.UNLEARNING_MIN_DELTA
    else:
        patience = config.TRAINING_PATIENCE
        min_delta = config.TRAINING_MIN_DELTA
        
    early_stopping = SISAEarlyStopping(
        patience=patience,
        min_delta=min_delta,
        monitor='val_loss',
        mode='min',
        restore_best_weights=True,
        verbose=True
    )

    # Same label smoothing for training and unlearning (see config) so the
    # before/after comparison is not confounded by the objective changing.
    label_smoothing_value = config.UNLEARNING_LABEL_SMOOTHING if training_type == 'unlearning' else config.LABEL_SMOOTHING
    print(f"   - Using label smoothing: {label_smoothing_value:.3f} ({'unlearning mode' if training_type == 'unlearning' else 'normal training'})")
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)  # Simple Adam optimizer

    if replay_buffer:
        # Handle both traditional dict replay buffer and SmartReplayBuffer
        if hasattr(replay_buffer, 'buffer'):
            # SmartReplayBuffer case
            buffer_dict = replay_buffer.buffer
            buffer_type = "smart replay"
        else:
            # Traditional dict case
            buffer_dict = replay_buffer
            buffer_type = "traditional replay"
        
        num_replay_samples = sum(len(d['y']) for d in buffer_dict.values())
        print(f"   - Using {buffer_type} buffer with {num_replay_samples} previous samples from {len(buffer_dict)} classes.")

    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Replay supplements each batch: striding by main_batch_size keeps every sample
    # seen once per epoch, with replay_batch_size samples appended on top.
    use_replay = bool(replay_buffer) and replay_ratio > 0
    if use_replay:
        main_batch_size = max(1, int(batch_size * (1 - replay_ratio)))
        replay_batch_size = batch_size - main_batch_size
    else:
        main_batch_size = batch_size
        replay_batch_size = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total, num_batches = 0.0, 0, 0, 0
        indices = rng.permutation(len(X))

        for i in range(0, len(X), main_batch_size):
            batch_indices = indices[i:i+main_batch_size]
            if len(batch_indices) == 0: continue

            if use_replay:
                main_x = x_train_t[batch_indices]
                main_y = y_train_t[batch_indices]

                # Choose replay strategy
                if use_smart_replay and hasattr(replay_buffer, 'sample_for_replay'):
                    # Smart replay buffer
                    replay_x_np, replay_y_np = replay_buffer.sample_for_replay(replay_batch_size)
                    if len(replay_x_np) > 0:
                        replay_x = torch.from_numpy(replay_x_np.astype(np.float32))
                        replay_y = torch.from_numpy(replay_y_np.astype(np.int64)).long()
                        batch_x = torch.cat((main_x, replay_x))
                        batch_y = torch.cat((main_y, replay_y))
                    else:
                        batch_x = main_x
                        batch_y = main_y
                else:
                    # Traditional random replay buffer
                    replay_x_list, replay_y_list = [], []
                    
                    # Handle both dict and SmartReplayBuffer
                    if hasattr(replay_buffer, 'buffer'):
                        buffer_dict = replay_buffer.buffer
                    else:
                        buffer_dict = replay_buffer
                    
                    available_classes = list(buffer_dict.keys())
                    
                    if available_classes and replay_batch_size > 0:
                        chosen_classes = rng.choice(available_classes, replay_batch_size)
                        for class_idx in chosen_classes:
                            class_data = buffer_dict[class_idx]
                            sample_idx = rng.integers(0, len(class_data['X']))
                            replay_x_list.append(class_data['X'][sample_idx])
                            replay_y_list.append(class_data['y'][sample_idx])
                        
                        replay_x = torch.from_numpy(np.array(replay_x_list).astype(np.float32))
                        replay_y = torch.from_numpy(np.array(replay_y_list).astype(np.int64)).long()
                        
                        batch_x = torch.cat((main_x, replay_x))
                        batch_y = torch.cat((main_y, replay_y))
                    else:
                        batch_x = main_x
                        batch_y = main_y
            else:
                batch_x = x_train_t[batch_indices]
                batch_y = y_train_t[batch_indices]

            batch_x = train_transforms(batch_x).to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            
            # Simple forward pass without mixed precision
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            _, predicted = torch.max(output.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / num_batches if num_batches > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        model.eval()
        val_loss, val_correct, val_total, val_batches = 0.0, 0, 0, 0

        if len(x_val_t) > 0:
            with torch.no_grad():
                for i in range(0, len(x_val_t), batch_size):
                    batch_x_val = x_val_t[i:i+batch_size]
                    batch_y_val = y_val_t[i:i+batch_size]

                    batch_x_val = val_transforms(batch_x_val).to(DEVICE)
                    batch_y_val = batch_y_val.to(DEVICE)

                    # Same objective the optimizer minimises: full cross-entropy on
                    # original labels, argmax over all classes as at inference time.
                    output = model(batch_x_val)

                    loss = criterion(output, batch_y_val)
                    val_loss += loss.item()
                    val_batches += 1
                    _, predicted = torch.max(output.data, 1)
                    val_total += batch_y_val.size(0)
                    val_correct += (predicted == batch_y_val).sum().item()

        val_epoch_loss = val_loss / val_batches if val_batches > 0 else 0
        val_epoch_acc = val_correct / val_total if val_total > 0 else 0
        
        # No lr_scheduler needed - Adam handles adaptive learning rates
        history['loss'].append(epoch_loss); history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss); history['val_accuracy'].append(val_epoch_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f'   Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        if early_stopping(val_epoch_loss, model, epoch):
            print("   - Early stopping triggered.")
            break
            
    early_stopping.restore_best_model(model)
    return model, history