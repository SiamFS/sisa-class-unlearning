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

from training.create_model import create_gating_model, save_model_pytorch, DEVICE

def train_gating(num_shards, base_dir, num_slices, dataset_mean, dataset_std, excluded_classes=None):

    # Basic augmentation to prevent overfitting
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),  # Only horizontal flip
        T.Normalize(dataset_mean, dataset_std)
    ])
    val_transforms = T.Compose([
        T.Normalize(dataset_mean, dataset_std)
    ])
    
    sisa_data_dir = os.path.join(base_dir, "sisa_data")
    models_dir = os.path.join(base_dir, "models")
    
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
        for slice_idx in range(num_slices):
            slice_x_path = os.path.join(sisa_data_dir, f"shards/shard_{shard_idx+1}/slice_{slice_idx}_x.npy")
            if os.path.exists(slice_x_path):
                x_data = np.load(slice_x_path)
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
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train_full, y_gating, test_size=0.2, stratify=y_gating, random_state=42
    )
    
    train_dataset = TensorDataset(torch.from_numpy(x_train_split).float(), torch.from_numpy(y_train_split).long())
    val_dataset = TensorDataset(torch.from_numpy(x_val_split).float(), torch.from_numpy(y_val_split).long())
    
    train_loader = DataLoader(train_dataset, batch_size=config.GATING_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.GATING_BATCH_SIZE, num_workers=0)
    
    # 5. Training Loop with Early Stopping
    model = create_gating_model(num_shards)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.GATING_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Early stopping setup for gating network
    best_val_acc = 0.0
    best_model_state = None
    epochs = config.GATING_MAX_EPOCHS
    patience = config.GATING_EARLY_STOPPING_PATIENCE  # Use config parameter for early stopping
    patience_counter = 0
    min_improvement = 0.001  # Minimum improvement threshold
    
    # START: Track pure gating training time
    import time
    pure_gating_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_batch, y_batch in train_loader:
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
        
        # Enhanced early stopping for gating network
        if val_acc > best_val_acc + min_improvement:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience counter
            print(f"   - New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
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
            'training_type': 'retrained' if excluded_classes else 'initial',
            'excluded_class_count': len(excluded_classes) if excluded_classes else 0,
        }
        save_model_pytorch(model, save_path, metadata=metadata)
            
    return os.path.join(models_dir, "gating_model.pth"), pure_gating_time