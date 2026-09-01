"""Image-array disk I/O (W34): store pixels as uint8, hand them out as float32.

The pipeline works in float32 [0, 1] end to end -- normalization stats, augmentation
fill values, and every model input assume it. But CIFAR pixels only ever take 256
distinct values, so writing them as float32 spends 4 bytes to store one of 256
possibilities: 3 of every 4 bytes on disk carry no information at all.

entry_data_processing.py used to bake the `/255.0` in BEFORE saving, so sisa_data was
738 MB where 184 MB holds exactly the same content. That inflated every rewrite, every
project clone, and every re-run of data processing by 4x.

These two functions move the conversion to the disk boundary instead. Callers are
unaffected: `load_images` returns float32 [0, 1] exactly as `np.load` did before, and
the encode/decode round trip is bit-for-bit lossless -- for every n in 0..255,
`rint((n/255.0) * 255) == n`, and decoding back reproduces the identical float32 bit
pattern. Verified against the real slice and test arrays, not just in principle.

Deliberately NOT changed: the in-memory pipeline. Sharding, slicing, the train_mean /
train_std computation and the plots all still see float32 [0, 1]. Only the bytes on
disk differ, which is what keeps this change incapable of moving any training result.

`load_images` also reads legacy float32 arrays unchanged, so old sisa_data keeps
working and the migration needs no flag day.
"""
import numpy as np

__all__ = ['load_images', 'save_images', 'encoded_dtype']


def encoded_dtype(arr: np.ndarray) -> np.dtype:
    """The dtype `save_images` would actually write for `arr`, without encoding it.

    Lets the idempotency guard in entry_data_processing.py compare what is on disk
    against what would be written, without materialising the encoded copy.
    """
    return np.dtype(np.uint8) if arr.dtype.kind == 'f' else arr.dtype


def save_images(path: str, arr: np.ndarray) -> None:
    """Write a float32 [0, 1] image array to `path` as uint8.

    Non-float arrays (labels, indices) are written through untouched, so this is safe
    to use for any array in the sisa_data tree.
    """
    if arr.dtype.kind != 'f':
        np.save(path, arr)
        return

    if arr.size:
        lo, hi = float(arr.min()), float(arr.max())
        if lo < 0.0 or hi > 1.0:
            # Loud failure rather than silent wraparound on the uint8 cast. Nothing in
            # the pipeline should reach here with normalized (mean-subtracted) data --
            # normalization happens per batch at train time, never on what is stored.
            raise ValueError(
                f"save_images expects float pixels in [0, 1], got [{lo}, {hi}] for {path}. "
                "Storing normalized data would not round-trip through uint8."
            )

    np.save(path, np.rint(arr * 255.0).astype(np.uint8))


def load_images(path: str) -> np.ndarray:
    """Read an image array as float32 [0, 1].

    Handles both storage generations: uint8 (current) is decoded, float32 (legacy
    sisa_data written before W34) is returned as-is.
    """
    arr = np.load(path)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr
