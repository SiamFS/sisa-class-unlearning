"""Global determinism for the SISA pipeline (W1).

Call `set_seed` once at the top of every entry point (data processing,
training, unlearning, tuning, scratch-reference) before any model, data,
or RNG is touched. This is required for the exactness proof: the unlearned
model can only be shown to equal a from-scratch model if training is
reproducible given the same seed and inputs.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) and force deterministic ops."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seeded_generator(seed: int) -> torch.Generator:
    """A torch.Generator seeded for use with DataLoader(generator=...)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
