import torch
import torch.nn as nn
import torch.nn.functional as F

# Import global configuration
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _infer_flat_features(conv_layer, input_shape):
    """Size the first FC layer from a dummy forward pass, so the architecture is
    independent of image resolution and channel count. eval() lets BatchNorm take n=1."""
    was_training = conv_layer.training
    conv_layer.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape)
        num_features = conv_layer(dummy).reshape(1, -1).size(1)
    conv_layer.train(was_training)
    return num_features


# --- Simplified SISANet to prevent overfitting ---
class SISANet(nn.Module):
    def __init__(self, num_classes=None, input_shape=None):
        super(SISANet, self).__init__()
        if num_classes is None:
            num_classes = config.get_num_classes()
        if input_shape is None:
            input_shape = config.get_input_shape()
        in_channels = input_shape[0]

        # Much simpler conv layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )
        # Configurable classifier using config parameters
        flat_features = _infer_flat_features(self.conv_layer, input_shape)
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=config.FC_LAYER_DROPOUT),
            nn.Linear(flat_features, config.FC_LAYER_1_HIDDEN),
            nn.BatchNorm1d(config.FC_LAYER_1_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.FC_LAYER_DROPOUT),
            nn.Linear(config.FC_LAYER_1_HIDDEN, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layer(x)
        return x

class GatingNetwork(nn.Module):
    """
    Lightweight gating network for SISA shard routing.
    Simplified architecture: 2 conv layers + 2 FC layers
    Purpose: Route samples to appropriate shards (NOT classify into classes)
    """
    def __init__(self, num_shards, input_shape=None):
        super(GatingNetwork, self).__init__()
        if input_shape is None:
            input_shape = config.get_input_shape()
        in_channels = input_shape[0]

        # Lightweight conv layers - only 2 layers for routing
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Simplified FC layers - only 2 layers for routing decision
        self._flat_features = _infer_flat_features(
            nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.pool,
                          self.conv2, self.bn2, nn.ReLU(), self.pool),
            input_shape,
        )
        self.fc1 = nn.Linear(self._flat_features, 128)
        self.dropout = nn.Dropout(config.GATING_DROPOUT_RATE)
        self.fc2 = nn.Linear(128, num_shards)  # Direct output to shards

    def forward(self, x):
        # Lightweight forward pass for routing
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # e.g. 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # e.g. 16x16 -> 8x8
        x = x.reshape(-1, self._flat_features)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output: shard probabilities
        return x

class PyTorchModelManager:
    def __init__(self, device=None):
        self.device = device if device is not None else DEVICE
        
    def save_model_complete(self, model, filepath, metadata=None):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_cpu = model.to('cpu')
        save_dict = {
            'model_state_dict': model_cpu.state_dict(),
            'model_class': model.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
        }
        if metadata:
            save_dict['metadata'] = metadata
        torch.save(save_dict, filepath)
        model.to(self.device)
        print(f" Model saved: {filepath}")
        return filepath
    
    def load_model_complete(self, filepath, num_classes=None, device=None, num_shards=None):
        device = device if device is not None else self.device
        checkpoint = torch.load(filepath, map_location=device)

        model_class_name = checkpoint.get('model_class', 'SISANet')
        if model_class_name == 'GatingNetwork' and num_shards is not None:
             model = create_gating_model(num_shards=num_shards)
        else:
            model = create_sisa_model(num_classes=num_classes)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        metadata = checkpoint.get('metadata', {})
        return model, metadata

# Backwards-compatible alias for checkpoints saved under the previous class name
SISACIFAR10Net = SISANet

def create_sisa_model(num_classes=None, input_shape=None):
    return SISANet(num_classes, input_shape).to(DEVICE)

def create_gating_model(num_shards, input_shape=None):
    return GatingNetwork(num_shards, input_shape).to(DEVICE)

def save_model_pytorch(model, filepath, metadata=None):
    manager = PyTorchModelManager(device=DEVICE)
    return manager.save_model_complete(model=model, filepath=filepath, metadata=metadata)

def load_model_pytorch(filepath, device=None, num_classes=None, num_shards=None):
    device = device if device is not None else DEVICE
    manager = PyTorchModelManager(device=device)
    return manager.load_model_complete(filepath, device=device, num_classes=num_classes, num_shards=num_shards)