"""Global determinism for the SISA pipeline (W1).

Call `set_seed` once at the top of every entry point (data processing,
training, unlearning, tuning, scratch-reference) before any model, data,
or RNG is touched. This is required for the exactness proof: the unlearned
model can only be shown to equal a from-scratch model if training is
reproducible given the same seed and inputs.
"""
import os
import random
import sys

# W23: set before `import torch` below, and therefore before anything can create a
# cuBLAS handle. Deterministic cuBLAS GEMMs on CUDA >= 10.2 require this variable to
# be present in the environment at handle-creation time -- setting it later is
# silently ignored. Doing it at module import (rather than only inside set_seed)
# means any caller that imports this module at the top of a file is covered, even if
# it calls set_seed further down.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

# W31: make stdout/stderr tolerant of non-ASCII at the process level.
#
# 25 print() lines across the codebase contain emoji or box-drawing characters, and
# Windows consoles default to cp1252, which cannot encode them -- so a run would die
# with UnicodeEncodeError partway through, mid-unlearning. run_logging's tee already
# handled this, but only for entry points that set logging up; anything calling the
# library directly (experiments/, a notebook, a test) still crashed. Reconfiguring
# here covers every caller, because every entry point imports this module -- directly
# or transitively -- before it prints anything.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass  # already wrapped, or a stream that doesn't support reconfigure

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) and force deterministic ops."""
    os.environ['PYTHONHASHSEED'] = str(seed)

    # W23: required for deterministic cuBLAS GEMMs on CUDA >= 10.2. Without it,
    # `use_deterministic_algorithms(warn_only=True)` only *warns* that every matmul
    # (i.e. every Linear layer, including the cosine head) is non-deterministic, and
    # runs it anyway -- so identical seeds did not in fact reproduce identical weights.
    # That silently invalidated W1's acceptance criterion and condition 3 of the
    # exactness argument in section 4.2. Must be set before the cuBLAS handle is
    # created, which is why set_seed() has to run before any CUDA work: every entry
    # point already calls it immediately after importing config.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

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
