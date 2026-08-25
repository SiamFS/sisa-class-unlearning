"""Per-sample batched augmentation (W19).

The previous pipeline applied a torchvision `Compose` directly to an already
batched tensor: `train_transforms(batch_x)` where `batch_x` is (N, C, H, W).
torchvision v1 transforms draw **one** random decision per call, so every sample
in a 64-image batch received the identical flip and the identical jitter -- far
weaker than per-sample augmentation, and not what any of the augmentation configs
intended.

This module implements the two geometric augmentations that matter for CIFAR
(random crop with reflect padding, and horizontal flip) as vectorised per-sample
tensor ops, driven by an explicitly seeded `torch.Generator` so runs stay bit-for-bit
reproducible (W1). Colour jitter is left to torchvision, since it only appears on
the unbalanced-shard paths and is secondary to the geometric transforms.

Augmentation runs on CPU tensors before the `.to(DEVICE)` transfer, which keeps the
generator device-independent and the results identical on CPU and GPU runs.
"""
from typing import Optional

import torch
import torch.nn.functional as F


class PerSampleAugmenter:
    """Vectorised per-sample random crop + horizontal flip.

    Args:
        crop_padding: pixels of reflect padding added before cropping back to the
            original size. 0 disables cropping.
        flip_prob: per-sample probability of a horizontal flip. 0 disables flipping.
        seed: seed for the internal `torch.Generator`.
    """

    def __init__(self, crop_padding: int = 0, flip_prob: float = 0.0, seed: int = 0):
        self.crop_padding = int(crop_padding or 0)
        self.flip_prob = float(flip_prob or 0.0)
        self._generator = torch.Generator(device='cpu')
        self._generator.manual_seed(int(seed))

    @property
    def enabled(self) -> bool:
        return self.crop_padding > 0 or self.flip_prob > 0.0

    def reset(self, seed: int) -> None:
        """Re-seed, so a fresh training run reproduces the same augmentation stream."""
        self._generator.manual_seed(int(seed))

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if not self.enabled or batch.numel() == 0:
            return batch

        batch = self._random_flip(batch)
        batch = self._random_crop(batch)
        return batch

    def _random_flip(self, batch: torch.Tensor) -> torch.Tensor:
        if self.flip_prob <= 0.0:
            return batch
        n = batch.size(0)
        draws = torch.rand(n, generator=self._generator, device='cpu')
        mask = (draws < self.flip_prob).to(batch.device)
        if not bool(mask.any()):
            return batch
        flipped = torch.flip(batch, dims=[3])
        # torch.where broadcasts the per-sample mask across C, H, W.
        return torch.where(mask.view(-1, 1, 1, 1), flipped, batch)

    def _random_crop(self, batch: torch.Tensor) -> torch.Tensor:
        pad = self.crop_padding
        if pad <= 0:
            return batch

        n, c, h, w = batch.shape
        padded = F.pad(batch, (pad, pad, pad, pad), mode='reflect')

        # One independent (top, left) offset per sample.
        max_offset = 2 * pad
        tops = torch.randint(0, max_offset + 1, (n,), generator=self._generator, device='cpu')
        lefts = torch.randint(0, max_offset + 1, (n,), generator=self._generator, device='cpu')
        tops = tops.to(batch.device)
        lefts = lefts.to(batch.device)

        rows = tops.view(n, 1) + torch.arange(h, device=batch.device).view(1, h)  # (n, h)
        cols = lefts.view(n, 1) + torch.arange(w, device=batch.device).view(1, w)  # (n, w)

        # Advanced indexing broadcasts (n,1,1,1) x (1,c,1,1) x (n,1,h,1) x (n,1,1,w) -> (n,c,h,w)
        batch_idx = torch.arange(n, device=batch.device).view(n, 1, 1, 1)
        chan_idx = torch.arange(c, device=batch.device).view(1, c, 1, 1)
        return padded[batch_idx, chan_idx, rows.view(n, 1, h, 1), cols.view(n, 1, 1, w)]


def build_augmenter(augmentation_config: Optional[dict], seed: int) -> PerSampleAugmenter:
    """Construct a `PerSampleAugmenter` from an augmentation config dict."""
    if not augmentation_config:
        return PerSampleAugmenter(0, 0.0, seed)
    return PerSampleAugmenter(
        crop_padding=augmentation_config.get('random_crop_padding', 0),
        flip_prob=augmentation_config.get('random_horizontal_flip', 0.0),
        seed=seed,
    )
