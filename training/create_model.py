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
        """Feature vector fed to the final classifier -- the routing representation (W21).

        W34: Dropout layers in the pre-classifier chain are SKIPPED, so this is
        deterministic regardless of the module's train/eval mode. `fc_layer[:5]` used to
        be applied wholesale, and position 4 is an `nn.Dropout` in both backbones
        (FC_LAYER_DROPOUT=0.5 for the ConvNet), so calling routing_cosine() on a module
        left in train mode returned randomly-zeroed features and a correspondingly random
        routing decision. Inert today -- routing runs under eval()/no_grad -- but a
        routing score must not depend on the module's mode.
        """
        x = self.conv_layer(x)
        x = x.reshape(x.size(0), -1)
        for layer in self.fc_layer[:5]:
            if isinstance(layer, nn.Dropout):
                continue
            x = layer(x)
        return x

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
    def __init__(self, num_shards, in_channels=3, pool_size=None, input_size=None):
        super(GatingNetwork, self).__init__()

        # W24: optionally route on a downsampled image. The gate's cost is dominated by
        # its two conv layers (4.7M of its 6.1M MACs sit in conv2 alone), and conv cost
        # scales with spatial area -- so halving each side cuts conv FLOPs ~4x, whereas
        # shrinking the FC layer (GATING_POOL_SIZE) removes 96% of the *parameters* but
        # only ~8% of the *compute*. Routing is a coarse "which shard" decision, so full
        # 32x32 detail may not be earning its cost. 0/None = no downsampling.
        if input_size is None:
            input_size = getattr(config, 'GATING_INPUT_SIZE', 0) or 0
        self.input_size = int(input_size)
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
        # W24: downsample before the convs, so the saving lands where the cost is.
        # adaptive_avg_pool2d (rather than a fixed stride) keeps this resolution-agnostic,
        # matching how the rest of the pipeline avoids hardcoding 32x32.
        if self.input_size and x.shape[-1] != self.input_size:
            x = F.adaptive_avg_pool2d(x, (self.input_size, self.input_size))
        # Lightweight forward pass for routing
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.reshape(-1, self._flat_features)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output: shard probabilities
        return x

class BasicBlock(nn.Module):
    """Standard two-conv residual block (He et al. 2015), CIFAR variant: 3x3 / 3x3 with
    a 1x1 projection shortcut only when stride or width changes."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class SISAResNet(nn.Module):
    """CIFAR-style residual backbone whose depth is derived from input resolution (W28).

    Stages are added until the feature map reaches `RESNET_TARGET_SPATIAL` before global
    average pooling, so 32x32 yields widths [16, 32, 64] at depth 20 -- reproducing
    He et al.'s ResNet-20 exactly -- and 64x64 yields [16, 32, 64, 128] at depth 26.
    The architecture is therefore a function of the dataset, matching how NUM_SHARDS,
    NUM_SLICES_PER_SHARD and MAX_SLICES_PER_SHARD are already derived.

    `fc_layer` deliberately mirrors SISAConvNet's six-element shape, padding the
    positions a residual net does not need with `Identity`. That is what lets every
    existing contract keep working unchanged:

      * `fc_layer[:5]` holds only Identity and Dropout, and `penultimate()` skips the
        Dropout, so it returns the GAP output unchanged;
      * `fc_layer.5.weight` exists, so W6's `_resize_model_head` and the loader's
        class-count inference need no special case;
      * `Identity` holds no parameters, so the ABSENCE of `fc_layer.1.weight` (the
        ConvNet's Linear(2048, 256)) identifies the architecture from a checkpoint
        alone -- no new metadata field, and old checkpoints still load.
    """

    def __init__(self, num_classes=10, in_channels=3, input_size=None,
                 blocks_per_stage=None, base_width=None, classifier_type=None,
                 stage_widths=None):
        super(SISAResNet, self).__init__()
        if classifier_type is None:
            classifier_type = getattr(config, 'CLASSIFIER_TYPE', 'linear')
        self.classifier_type = classifier_type

        if input_size is None:
            input_size = config.get_input_size()
        if blocks_per_stage is None:
            blocks_per_stage = config.RESNET_BLOCKS_PER_STAGE
        if base_width is None:
            base_width = config.RESNET_BASE_WIDTH

        # `stage_widths` is an explicit override used when rebuilding from a checkpoint,
        # so a saved model reconstructs exactly even if the config has since changed.
        widths = list(stage_widths) if stage_widths else config.resnet_stage_widths(input_size, base_width)
        self.input_size = int(input_size)
        self.blocks_per_stage = int(blocks_per_stage)
        self.stage_widths = list(widths)
        self.feature_dim = widths[-1]
        self.depth = 2 * self.blocks_per_stage * len(widths) + 2

        layers = [
            nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        ]
        current = widths[0]
        for stage_idx, width in enumerate(widths):
            for block_idx in range(self.blocks_per_stage):
                # Downsample once at the start of every stage after the first.
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                layers.append(BasicBlock(current, width, stride))
                current = width
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv_layer = nn.Sequential(*layers)

        if classifier_type == 'cosine':
            final_layer = CosineClassifier(self.feature_dim, num_classes)
        else:
            final_layer = nn.Linear(self.feature_dim, num_classes)

        # Positions 0-3 are Identity: see the class docstring for why the shape matters.
        self.fc_layer = nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Dropout(p=config.RESNET_DROPOUT),
            final_layer,
        )

    def penultimate(self, x):
        """Feature vector fed to the final classifier -- the routing representation (W21).

        W34: Dropout layers in the pre-classifier chain are SKIPPED, so this is
        deterministic regardless of the module's train/eval mode. `fc_layer[:5]` used to
        be applied wholesale, and position 4 is an `nn.Dropout` in both backbones
        (FC_LAYER_DROPOUT=0.5 for the ConvNet), so calling routing_cosine() on a module
        left in train mode returned randomly-zeroed features and a correspondingly random
        routing decision. Inert today -- routing runs under eval()/no_grad -- but a
        routing score must not depend on the module's mode.
        """
        x = self.conv_layer(x)
        x = x.reshape(x.size(0), -1)
        for layer in self.fc_layer[:5]:
            if isinstance(layer, nn.Dropout):
                continue
            x = layer(x)
        return x

    def routing_cosine(self, x):
        """Cosine similarity of each sample to every class vector, in [-1, 1]. Same
        contract as SISAConvNet.routing_cosine."""
        features = self.penultimate(x)
        final_layer = self.fc_layer[5]
        if isinstance(final_layer, CosineClassifier):
            return final_layer.cosine(features)
        return F.normalize(features, p=2, dim=1) @ F.normalize(final_layer.weight, p=2, dim=1).t()

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.reshape(x.size(0), -1)
        return self.fc_layer(x)


def _infer_resnet_shape(state_dict):
    """Recover a SISAResNet's geometry from its own state dict (W28/W29).

    Specialists and the router share the architecture but differ in depth
    (RESNET_BLOCKS_PER_STAGE vs GATING_BLOCKS_PER_STAGE), and either may have been
    saved under a config that has since changed. Reading the shape back from the
    checkpoint makes saved models self-describing, so they always reconstruct exactly.

    Returns (stage_widths, blocks_per_stage, num_classes).
    """
    stem = state_dict['conv_layer.0.weight'].shape[0]
    feature_dim = state_dict['fc_layer.5.weight'].shape[1]
    num_classes = state_dict['fc_layer.5.weight'].shape[0]

    # Widths double once per stage, from the stem width up to the final feature width.
    widths, w = [], stem
    while w < feature_dim:
        widths.append(w)
        w *= 2
    widths.append(feature_dim)

    n_blocks = sum(1 for k in state_dict if k.endswith('.conv1.weight') and k.startswith('conv_layer.'))
    blocks_per_stage = max(1, n_blocks // len(widths))
    return widths, blocks_per_stage, num_classes


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

        # W29: a residual gate and a residual specialist are both SISAResNet, so the
        # class name cannot tell them apart -- they differ only in depth. `num_shards`
        # is the existing signal that a gate is being loaded (callers pass it only for
        # the router), and the geometry is read back from the checkpoint itself so the
        # gate rebuilds at ITS depth rather than the specialist's.
        if model_class_name == 'SISAResNet' and num_shards is not None:
            sd = checkpoint['model_state_dict']
            widths, blocks, n_out = _infer_resnet_shape(sd)
            model = SISAResNet(
                num_classes=n_out,
                stage_widths=widths,
                blocks_per_stage=blocks,
                classifier_type='linear' if 'fc_layer.5.bias' in sd else 'cosine',
            ).to(device)
        elif model_class_name == 'GatingNetwork' and num_shards is not None:
            # W22: infer the checkpoint's own pooling size from fc1's input width, so
            # gates saved before GATING_POOL_SIZE existed still load after the change.
            gate_pool_size = None
            fc1_weight = checkpoint['model_state_dict'].get('fc1.weight')
            if fc1_weight is not None:
                gate_pool_size = int(round((fc1_weight.shape[1] / 64) ** 0.5))
            # W24: the routing input size is not recoverable from any weight shape (the
            # adaptive pool hides it), so it travels in the checkpoint metadata. Absent
            # metadata means a gate saved before W24, which routed on full-size input.
            gate_input_size = checkpoint.get('metadata', {}).get('input_size', 0)
            model = create_gating_model(
                num_shards=num_shards, pool_size=gate_pool_size, input_size=gate_input_size
            )
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
            # W28: the checkpoint also decides its own backbone. SISAConvNet has
            # fc_layer.1.weight (its Linear(2048, 256)); SISAResNet pads that position
            # with Identity, which holds no parameters -- so the key's absence is an
            # exact architecture discriminator, and a ConvNet checkpoint still loads
            # while MODEL_ARCH is set to 'resnet'.
            checkpoint_arch = 'convnet' if 'fc_layer.1.weight' in state_dict else 'resnet'
            if checkpoint_arch == 'resnet':
                widths, blocks, _ = _infer_resnet_shape(state_dict)
                model = SISAResNet(num_classes=num_classes, stage_widths=widths,
                                   blocks_per_stage=blocks,
                                   classifier_type=checkpoint_classifier).to(device)
            else:
                model = create_sisa_model(num_classes=num_classes,
                                          classifier_type=checkpoint_classifier,
                                          arch=checkpoint_arch)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        metadata = checkpoint.get('metadata', {})
        return model, metadata

def create_sisa_model(num_classes=None, in_channels=None, classifier_type=None, arch=None):
    """Build a specialist. `arch` overrides config.MODEL_ARCH -- used by the loader so a
    checkpoint's own architecture wins over the current config setting (W28)."""
    if num_classes is None:
        num_classes = config.get_num_classes()
    if in_channels is None:
        in_channels = config.IN_CHANNELS
    if arch is None:
        arch = getattr(config, 'MODEL_ARCH', 'convnet')

    if str(arch).lower() == 'resnet':
        return SISAResNet(num_classes, in_channels, classifier_type=classifier_type).to(DEVICE)
    return SISAConvNet(num_classes, in_channels, classifier_type=classifier_type).to(DEVICE)

def create_gating_model(num_shards, in_channels=None, pool_size=None, input_size=None, arch=None):
    """Build the router.

    W29: under MODEL_ARCH='resnet' the gate is the SAME residual family as the
    specialists, just shallower (GATING_BLOCKS_PER_STAGE) -- one architecture with a
    depth parameter rather than two unrelated networks. Measured at 1 block/stage,
    base 16: 77,522 params at 32x32, versus GatingNetwork's 544,258, which was 200% of
    a ResNet-20 specialist -- a router twice the size of the model it routes to.

    Note `pool_size`/`input_size` are GatingNetwork-only knobs (W22/W24); a residual
    gate derives its own stages from resolution.
    """
    if in_channels is None:
        in_channels = config.IN_CHANNELS
    if arch is None:
        arch = getattr(config, 'MODEL_ARCH', 'convnet')

    if str(arch).lower() == 'resnet':
        return SISAResNet(
            num_classes=num_shards,
            in_channels=in_channels,
            input_size=input_size,
            blocks_per_stage=config.GATING_BLOCKS_PER_STAGE,
            base_width=config.GATING_BASE_WIDTH,
            classifier_type='linear',  # routing is a plain K-way decision
        ).to(DEVICE)

    return GatingNetwork(num_shards, in_channels, pool_size=pool_size, input_size=input_size).to(DEVICE)

def save_model_pytorch(model, filepath, metadata=None):
    manager = PyTorchModelManager(device=DEVICE)
    return manager.save_model_complete(model=model, filepath=filepath, metadata=metadata)

def load_model_pytorch(filepath, device=None, num_classes=None, num_shards=None):
    device = device if device is not None else DEVICE
    manager = PyTorchModelManager(device=device)
    return manager.load_model_complete(filepath, device=device, num_classes=num_classes, num_shards=num_shards)