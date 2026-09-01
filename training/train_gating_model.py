import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T

# Import global configuration
import config
from utils.seeding import seeded_generator
from utils.data_io import load_images

from training.create_model import create_gating_model, save_model_pytorch, DEVICE
from training.augmentation import PerSampleAugmenter

def train_gating(num_shards, base_dir, num_slices, dataset_mean, dataset_std, excluded_classes=None,
                 save_dir=None):
    """Train the shard router.

    `save_dir` (W31) redirects the checkpoint away from the project's real
    models/gating_model.pth. Required by anything that trains a throwaway gate --
    a hyperparameter trial or the W8 scratch reference -- so a side experiment can
    never overwrite the deployed router.
    """

    # W19/W22: geometric augmentation is per-sample (a torchvision Compose applied to a
    # batched tensor gives the whole batch one shared decision) and now includes the
    # random crop the gate previously lacked entirely.
    gating_augmenter = PerSampleAugmenter(
        crop_padding=config.GATING_CROP_PADDING,
        flip_prob=config.GATING_FLIP_PROB,
        seed=config.SEED,
    )
    train_transforms = T.Compose([
        T.Normalize(dataset_mean, dataset_std)
    ])
    val_transforms = T.Compose([
        T.Normalize(dataset_mean, dataset_std)
    ])
    
    # W26: num_slices may be a per-shard list (shards can differ in slice count) or a
    # legacy scalar. Normalise to a list so the reconstruction loop works for both.
    if isinstance(num_slices, int):
        num_slices = [num_slices] * num_shards

    sisa_data_dir = os.path.join(base_dir, "sisa_data")
    models_dir = save_dir if save_dir else os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Build a map from class index to shard index
    class_to_shard_map = {}
    for i in range(num_shards):
        shard_meta_path = os.path.join(sisa_data_dir, f"shards/shard_{i+1}/metadata.json")
        with open(shard_meta_path, 'r') as f:
            shard_meta = json.load(f)
        for class_idx in shard_meta['class_indices_present']:
            class_to_shard_map[class_idx] = i
    
    print(f"   - Built class-to-shard map: {class_to_shard_map}")
    if excluded_classes:
        print(f"   - Excluding unlearned classes from gating training: {excluded_classes}")

    # 2. Reconstruct the full training data from the sharded and sliced files
    all_x_data = []
    all_y_data = []
    for shard_idx in range(num_shards):
        for slice_idx in range(num_slices[shard_idx]):
            slice_x_path = os.path.join(sisa_data_dir, f"shards/shard_{shard_idx+1}/slice_{slice_idx}_x.npy")
            if os.path.exists(slice_x_path):
                x_data = load_images(slice_x_path)
                y_data = np.load(slice_x_path.replace('_x.npy', '_y.npy'))
                
                # CRITICAL FIX: Filter out excluded/unlearned classes
                if excluded_classes:
                    mask = ~np.isin(y_data, excluded_classes)
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                
                if len(x_data) > 0:
                    all_x_data.append(x_data)
                    all_y_data.append(y_data)
    
    if not all_x_data:
        print("   - ERROR: No training data available after filtering")
        return None
        
    x_train_full = np.concatenate(all_x_data)
    y_train_full = np.concatenate(all_y_data)
    
    # 3. Create labels for the Gating Network (the target is the shard index)
    # Filter out classes that are not in class_to_shard_map (unlearned classes)
    valid_mask = np.array([label in class_to_shard_map for label in y_train_full])
    x_train_full = x_train_full[valid_mask]
    y_train_full = y_train_full[valid_mask]
    
    y_gating = np.array([class_to_shard_map[label] for label in y_train_full])
    print(f"   - Created {len(y_gating)} labels for Gating Network training.")
    
    # 4. Create DataLoader
    # W22: was a hardcoded literal 42 -- harmless only while config.SEED happens to be
    # 42, and the exact class of stray-literal reproducibility bug W4 already fixed once.
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train_full, y_gating, test_size=0.2, stratify=y_gating, random_state=config.SEED
    )
    
    train_dataset = TensorDataset(torch.from_numpy(x_train_split).float(), torch.from_numpy(y_train_split).long())
    val_dataset = TensorDataset(torch.from_numpy(x_val_split).float(), torch.from_numpy(y_val_split).long())
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.GATING_BATCH_SIZE, shuffle=True, num_workers=0,
        generator=seeded_generator(config.SEED),
    )
    val_loader = DataLoader(val_dataset, batch_size=config.GATING_BATCH_SIZE, num_workers=0)
    
    # 5. Training Loop with Early Stopping
    model = create_gating_model(num_shards)

    # W22: inverse-frequency class weights. Shard labels are imbalanced whenever shards
    # own different numbers of classes -- W17's semantic clustering made this 18000 vs
    # 27000 (40/60) -- and unweighted cross-entropy drifts toward the majority shard.
    class_weights = None
    if getattr(config, 'GATING_CLASS_WEIGHTS', False):
        counts = np.bincount(y_train_split, minlength=num_shards).astype(np.float64)
        if (counts > 0).all():
            weights = counts.sum() / (num_shards * counts)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
            print(f"   - Shard label counts: {counts.astype(int).tolist()}")
            print(f"   - Applying class weights: {[round(float(w), 4) for w in weights]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # W23: same decoupled-weight-decay fix as the specialists (Adam folds L2 into the
    # adaptive update; AdamW applies it as actual weight decay).
    if getattr(config, 'OPTIMIZER', 'adam').lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.GATING_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.GATING_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # W25: the gate now trains under the SAME regime as the specialists. It previously
    # had none of W23's fixes -- no LR schedule, no minimum-epoch floor, patience 4 and
    # a 25-epoch cap -- so it stopped at its first plateau while the LR was still at its
    # initial value. Measured on the 83.26% run: it stopped at epoch 10/25 with its best
    # at epoch 6, while training accuracy was still climbing monotonically (0.888 ->
    # 0.954). That is exactly the premature-stopping defect W23 fixed everywhere else,
    # and it matters more here than anywhere: routing is now the system bottleneck.
    best_val_acc = 0.0
    best_model_state = None
    epochs = config.GATING_MAX_EPOCHS
    patience = config.GATING_EARLY_STOPPING_PATIENCE
    min_epochs = getattr(config, 'GATING_MIN_EPOCHS', 0)
    min_improvement = config.TRAINING_MIN_DELTA  # same accuracy threshold as the specialists
    patience_counter = 0

    scheduler = None
    if getattr(config, 'LR_SCHEDULER', 'none') == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR,
        )
    lr_reductions = 0
    min_lr_reductions = getattr(config, 'MIN_LR_REDUCTIONS_BEFORE_STOP', 0) if scheduler is not None else 0
    print(f"   - Gate schedule: max_epochs={epochs}, patience={patience}, min_epochs={min_epochs}, "
          f"LR plateau x{min_lr_reductions} before stopping")


    # START: Track pure gating training time
    import time
    pure_gating_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = gating_augmenter(x_batch)
            x_batch = train_transforms(x_batch).to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0.0
        
        with torch.no_grad():
            for x_batch_val, y_batch in val_loader:
                x_batch_val = val_transforms(x_batch_val).to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(x_batch_val)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"   - Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

        # W25: anneal the LR on plateau, mirroring the specialists.
        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < prev_lr:
                lr_reductions += 1
                gate_note = (f" (reduction {lr_reductions} of {min_lr_reductions} required before stopping)"
                             if lr_reductions < min_lr_reductions else "")
                print(f"   - LR reduced: {prev_lr:.2e} -> {new_lr:.2e}{gate_note}")

        # Enhanced early stopping for gating network
        if val_acc > best_val_acc + min_improvement:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience counter
            print(f"   - New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # W25: same two guards the specialists get -- honour the minimum-epoch
                # floor, and do not stop until the LR has actually been annealed. A
                # plateau at the starting LR says nothing about the lower ones.
                if (epoch + 1) < min_epochs:
                    pass
                elif lr_reductions < min_lr_reductions:
                    print(f"   - Plateau reached, but only {lr_reductions}/{min_lr_reductions} LR "
                          f"reductions so far -- continuing at a lower LR instead of stopping.")
                    patience_counter = 0
                else:
                    print(f"   - Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
    
    # END: Track pure gating training time
    pure_gating_end = time.time()
    pure_gating_time = pure_gating_end - pure_gating_start
    print(f"   - Pure Gating Network Training Time: {pure_gating_time:.2f} seconds")
    
    # Save only the best model at the end
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_path = os.path.join(models_dir, "gating_model.pth")
        # PRIVACY: Metadata only contains shard count, NOT class information
        metadata = {
            'best_val_acc': best_val_acc,
            'num_shards': num_shards,
            # W24: routing resolution is invisible in the weight shapes (the adaptive
            # pool hides it), so the loader has to read it back from here.
            'input_size': model.input_size,
            'training_type': 'retrained' if excluded_classes else 'initial',
            'excluded_class_count': len(excluded_classes) if excluded_classes else 0,
            # NOTE: We do NOT store class names, indices, or any class-identifying information
            # The gating network is a pure shard router with NO knowledge of classes
        }
        save_model_pytorch(model, save_path, metadata=metadata)
        print(f"   - Saved final best Gating Network model (Val Acc: {best_val_acc:.4f})")
    else:
        print("   - No improvement found, saving final model state")
        save_path = os.path.join(models_dir, "gating_model.pth")
        metadata = {
            'best_val_acc': 0.0,
            'num_shards': num_shards,
            # W24: routing resolution is invisible in the weight shapes (the adaptive
            # pool hides it), so the loader has to read it back from here.
            'input_size': model.input_size,
            'training_type': 'retrained' if excluded_classes else 'initial',
            'excluded_class_count': len(excluded_classes) if excluded_classes else 0,
        }
        save_model_pytorch(model, save_path, metadata=metadata)
            
    return os.path.join(models_dir, "gating_model.pth"), pure_gating_time