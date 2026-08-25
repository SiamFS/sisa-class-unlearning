import torch
import torch.nn as nn
import torch.nn.functional as F

# Import global configuration
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from datetime import datetime
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CosineClassifier(nn.Module):
    """Scaled cosine classifier (W21, LUCIR-style): logits = s * cos(f, w_c).

    Two properties matter here, both measured problems in this project:

    1. **Routing for free.** Because features and class vectors are L2-normalized,
       `cos(f, w_c)` is bounded in [-1, 1] and directly comparable *across independently
       trained specialists* -- which raw softmax confidence and logsumexp energy are not
       (W15 rejected both; the gate-free probe measured 61.4% and 65.6% routing for them
       versus 75.1% for cosine-to-prototype). Routing score = max over owned classes.
    2. **Recency-bias mitigation.** Plain `nn.Linear` lets the weight vectors of
       recently-trained classes grow in magnitude, so they dominate the argmax. This is
       the horse/frog effect in the confusion matrix (0.91 recall, 0.55 precision).
       Normalizing the class vectors removes magnitude as a degree of freedom entirely.

    Deliberately quacks like `nn.Linear`: exposes `in_features`/`out_features` and
    registers its parameter as `weight` with shape (num_classes, in_features), so
    W6's `_resize_model_head` and the checkpoint head-width inference in
    `load_model_complete` both keep working with no changes.
    """

    def __init__(self, in_features, out_features, scale=None, learnable_scale=None):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

        scale = config.COSINE_SCALE_INIT if scale is None else scale
        learnable = config.COSINE_SCALE_LEARNABLE if learnable_scale is None else learnable_scale
        if learnable:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer('scale', torch.tensor(float(scale)))

    def cosine(self, x):
        """Raw cosine similarities in [-1, 1] -- the routing score, unscaled."""
        return F.normalize(x, p=2, dim=1) @ F.normalize(self.weight, p=2, dim=1).t()

    def forward(self, x):
        return self.scale * self.cosine(x)


# --- Simplified SISAConvNet to prevent overfitting ---
class SISAConvNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, classifier_type=None):
        super(SISAConvNet, self).__init__()
        if classifier_type is None:
            classifier_type = getattr(config, 'CLASSIFIER_TYPE', 'linear')
        self.classifier_type = classifier_type
        # Much simpler conv layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fixes the FC input size to 128*4*4 regardless of input resolution
            # (previously assumed exactly 32x32 input -> 4x4 after 3 poolings)
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # Configurable classifier using config parameters. Index 5 is the final
        # classifier in both modes -- W6's head resize and the loader's head-width
        # inference (`fc_layer.5.weight`) rely on that position being stable.
        if classifier_type == 'cosine':
            final_layer = CosineClassifier(config.FC_LAYER_1_HIDDEN, num_classes)
        else:
            final_layer = nn.Linear(config.FC_LAYER_1_HIDDEN, num_classes)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=config.FC_LAYER_DROPOUT),
            nn.Linear(config.FC_LAYER_1_INPUT, config.FC_LAYER_1_HIDDEN),
            nn.BatchNorm1d(config.FC_LAYER_1_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.FC_LAYER_DROPOUT),
            final_layer
        )

    def penultimate(self, x):
        """Feature vector fed to the final classifier -- the routing representation (W21)."""
        x = self.conv_layer(x)
        x = x.reshape(x.size(0), -1)
        return self.fc_layer[:5](x)

    def routing_cosine(self, x):
        """Cosine similarity of each sample to every class vector, in [-1, 1].

        Defined for both head types: with a cosine head these are the head's own
        normalized similarities; with a linear head it is the same geometry computed
        against the (unnormalized) weight rows, which is the `ncm_cosine` score the
        gate-free probe measured. Either way the value is bounded and therefore
        comparable across independently trained specialists.
        """
        features = self.penultimate(x)
        final_layer = self.fc_layer[5]
        if isinstance(final_layer, CosineClassifier):
            return final_layer.cosine(features)
        return F.normalize(features, p=2, dim=1) @ F.normalize(final_layer.weight, p=2, dim=1).t()

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
    def __init__(self, num_shards, in_channels=3, pool_size=None):
        super(GatingNetwork, self).__init__()
        # Lightweight conv layers - only 2 layers for routing
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # W22: the 8x8 pool made fc1 a Linear(4096, 128) -- 524k of the network's ~544k
        # parameters, nearly as large as a full specialist (~620k), to make what is only
        # a K-way routing decision. Global average pooling (pool_size=1) cuts that to
        # Linear(64, 128) at ~8k, shrinking the whole gate roughly 19x. Routing is a
        # coarse decision, so the retained 8x8 spatial grid was not earning its cost.
        if pool_size is None:
            pool_size = getattr(config, 'GATING_POOL_SIZE', 8)
        self.pool_size = int(pool_size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self._flat_features = 64 * self.pool_size * self.pool_size

        # Simplified FC layers - only 2 layers for routing decision
        self.fc1 = nn.Linear(self._flat_features, 128)
        self.dropout = nn.Dropout(config.GATING_DROPOUT_RATE)
        self.fc2 = nn.Linear(128, num_shards)  # Direct output to shards

    def forward(self, x):
        # Lightweight forward pass for routing
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
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
        
        model_class_name = checkpoint.get('model_class', 'SISAConvNet')
        if model_class_name == 'GatingNetwork' and num_shards is not None:
            # W22: infer the checkpoint's own pooling size from fc1's input width, so
            # gates saved before GATING_POOL_SIZE existed still load after the change.
            gate_pool_size = None
            fc1_weight = checkpoint['model_state_dict'].get('fc1.weight')
            if fc1_weight is not None:
                gate_pool_size = int(round((fc1_weight.shape[1] / 64) ** 0.5))
            model = create_gating_model(num_shards=num_shards, pool_size=gate_pool_size)
        else:
            state_dict = checkpoint['model_state_dict']
            if num_classes is None:
                # W6 dynamic head: a checkpoint's own final-layer shape is authoritative
                # about its class count, whether that's the full global count (a shard
                # never unlearned) or a reduced count (retrained after unlearning).
                final_layer_weight = state_dict.get('fc_layer.5.weight')
                if final_layer_weight is not None:
                    num_classes = final_layer_weight.shape[0]
            # W21: the checkpoint also decides its own head type, so linear checkpoints
            # saved before the cosine head existed still load correctly even when
            # config.CLASSIFIER_TYPE has since been switched to 'cosine'. A linear head
            # has a bias term; a cosine head has no bias and carries a scale instead.
            if 'fc_layer.5.bias' in state_dict:
                checkpoint_classifier = 'linear'
            elif 'fc_layer.5.scale' in state_dict:
                checkpoint_classifier = 'cosine'
            else:
                checkpoint_classifier = getattr(config, 'CLASSIFIER_TYPE', 'linear')
            model = create_sisa_model(num_classes=num_classes, classifier_type=checkpoint_classifier)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        metadata = checkpoint.get('metadata', {})
        return model, metadata

def create_sisa_model(num_classes=None, in_channels=None, classifier_type=None):
    if num_classes is None:
        num_classes = config.get_num_classes()
    if in_channels is None:
        in_channels = config.IN_CHANNELS
    return SISAConvNet(num_classes, in_channels, classifier_type=classifier_type).to(DEVICE)

def create_gating_model(num_shards, in_channels=None, pool_size=None):
    if in_channels is None:
        in_channels = config.IN_CHANNELS
    return GatingNetwork(num_shards, in_channels, pool_size=pool_size).to(DEVICE)

def save_model_pytorch(model, filepath, metadata=None):
    manager = PyTorchModelManager(device=DEVICE)
    return manager.save_model_complete(model=model, filepath=filepath, metadata=metadata)

def load_model_pytorch(filepath, device=None, num_classes=None, num_shards=None):
    device = device if device is not None else DEVICE
    manager = PyTorchModelManager(device=device)
    return manager.load_model_complete(filepath, device=device, num_classes=num_classes, num_shards=num_shards)