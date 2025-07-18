"""
Noise quilting algorithm, for stitching together the tiles of noise that we generate.

Based on "Image Quilting for Texture Synthesis and Transfer" (Efros and Freeman 2001)

"""

import numpy as np

from typing import Iterable


class PatchError(Exception):
    """
    General exception for an issue with the patches

    """


def _ceildiv(a: int, b: int) -> int:
    """
    Return the ceiling of the division of a by b.

    :param a: the numerator
    :param b: the denominator
    :return: the ceiling of a / b
    """
    return -(a // -b)


def _patch_layout(
    target_size: tuple[int, ...], patch_size: tuple[int, ...], patch_overlap: int
) -> tuple[int, ...]:
    """
    Calculate how many patches are needed to fill the target size, given the patch size and overlap.

    If the patch size does not fit exactly into the target size, this will give the number of patches needed
    to fill the target size, with some extra which should be dealt with by the caller.

    :param target_size: the size of the desired quilt of patches
    :param patch_size: the size of each patch
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)

    :raises ValueError: if the patch size is larger than the target size, if the overlap is negative
                        or if the target and patch have different dimensions
    """
    if patch_overlap < 0:
        raise ValueError(f"Patch overlap must be non-negative, got {patch_overlap}")
    if patch_overlap >= min(patch_size):
        raise ValueError(
            f"Patch overlap must be smaller than the patch size, got {patch_overlap} and {patch_size}"
        )
    if len(target_size) != len(patch_size):
        raise ValueError(
            f"Target size {target_size} and patch size {patch_size} must have the same number of dimensions"
        )

    return tuple(
        _ceildiv(t - patch_overlap, p - patch_overlap)
        for t, p in zip(target_size, patch_size)
    )


def quilt(
    patches: Iterable[np.ndarray], *, target_size: tuple[int, int], patch_overlap: int
) -> np.ndarray:
    """
    Quilt together a collection of patches to give an array of the provided size.

    Chooses patches from `patches` randomly (with replacement), stitching them together until
    the desired `target_size` is reached.
    Uses the quilting algorithm in [1] to choose which patches should be aligned next to each other
    and the optimal cut through each patch.

    :param patches: the patches to stitch together
    :param target_size: the size of the desired quilt of patches
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)

    :raises UserWarning: if the provided target_size, patch size and overlap
    :raises ValueError: if the patches are not all the same size or if they are not 2d
    :return: the optimal quilt of patches

    """
    # Check that the patches are all the same size
    patch_sizes = {patch.shape for patch in patches}
    try:
        (patch_size,) = patch_sizes
    except ValueError:
        raise PatchError(f"All patches must be the same size; got {patch_sizes}")

    # Check the patches are all 2d
    if len(patch_size) != 2:
        raise PatchError(f"Patches must be 2d, not {patch_size}")

    # Find how many patches we need to build up to the target size
    # Emit a warning if it doesn't fit exactly - we'll have to crop the last rows/columns

    # Init a list of lists holding the patches
    # Choose a random patch to start with
    # Choose optimal patches to build up the first row, applying vertical stitches
    # Choose optimal patches to build up the rest of the array

    # Add vertical stitches in the first row
    # Add horizontal/diagonal seams to the rest of the patches
    # build this into the ideal quilt

    # Check that the quilt is the right size
    return
