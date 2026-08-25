"""Dataset factory (W12): maps config.DATASET to a torchvision dataset class.

Both currently-registered datasets are 32x32x3 RGB with a `.classes`
attribute (the same shape/channel assumptions the rest of the pipeline
already makes explicit via config.IN_CHANNELS and the model's adaptive
pooling), so switching between them is a config-only change -- no code path
elsewhere hardcodes "10 classes" or "CIFAR-10" anymore (num_classes and class
names are always derived from sisa_data/metadata.json).
"""
import torchvision.datasets as tv_datasets

DATASET_REGISTRY = {
    'cifar10': tv_datasets.CIFAR10,
    'cifar100': tv_datasets.CIFAR100,
}


def get_dataset_class(name: str):
    """Resolve a config.DATASET string (e.g. "cifar10", "CIFAR-10", "cifar_10")
    to the torchvision dataset class that loads it."""
    key = name.lower().replace('-', '').replace('_', '')
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[key]
