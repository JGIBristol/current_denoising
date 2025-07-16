"""
Noise quilting algorithm, for stitching together the tiles of noise that we generate.

Based on "Image Quilting for Texture Synthesis and Transfer" (Efros and Freeman 2001)

"""

import warnings
from typing import Iterable

import numpy as np


class PatchError(Exception):
    """
    General exception for an issue with the patches

    """


class PatchSizeMismatchError(PatchError):
    """
    Patches are not all the same size, somehow
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


def _verify_patches(patches: Iterable[np.ndarray]) -> None:
    """
    Perform some basic checks on the patches to ensure they are suitable for quilting.

    Checks that they are all the same size and 2d.

    """
    if not all((patch.shape == patches[0].shape for patch in patches)):
        raise PatchSizeMismatchError("All patches must be the same size")
    if not all((len(patch.shape) == 2 for patch in patches)):
        raise PatchSizeMismatchError("All patches must be 2d")


def randomly_choose_patches(
    patches: Iterable[np.ndarray],
    target_size: tuple[int, int],
    patch_overlap: int,
    allow_rotation: bool = False,
    *,
    rng: np.random.Generator,
) -> list[list[np.ndarray]]:
    """
    Randomly choose patches that will at least fill the target size when stitched together.

    Chooses patches with replacement.

    :param patches: the patches to choose from
    :param target_size: the size of the desired quilt of patches
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)
    :param allow_rotation: whether to allow patches to be rotated when matching them
    :param rng: a random number generator

    :raises PatchSizeMismatchError: if the patches are not all the same size or they are not all 2d
    :return: patches that will at least fill the target size when stitched together.
             Note that after stitching, the resulting array may be larger than the target size.

    """
    if allow_rotation:
        raise NotImplementedError(
            "Randomly choosing patches with rotation is not yet implemented"
        )

    _verify_patches(patches)
    n_col, n_row = _patch_layout(target_size, patches[0].shape, patch_overlap)

    out_list = [[None for _ in range(n_col)] for _ in range(n_row)]

    for i in range(n_row):
        for j in range(n_col):
            out_list[i][j] = rng.choice(patches)

    return out_list


def _get_best_patch(
    patches: Iterable[np.ndarray],
    comparison_patch: np.ndarray,
    patch_slice: tuple[slice, slice],
) -> np.ndarray:
    """
    Iterate over patches, and find the one that best matches the comparison patch
    in the specified slice.

    Comparison patch should be already sliced
    """
    score = float("-inf")
    best_patch = None
    for patch in patches:
        overlap_region = patch[patch_slice]
        mse = np.sum((overlap_region - comparison_patch) ** 2)
        if mse > score:
            score = mse
            best_patch = patch

    return best_patch


def _best_patch_compare_left(
    patches: Iterable[np.ndarray],
    comparison_patch: np.ndarray,
    patch_overlap: int,
) -> np.ndarray:
    """
    Iterate over patches, and find the one that best matches the comparison patch
    along its right edge
    """
    # We just want to take the right bit of the comparison patch
    comparison_patch = comparison_patch[:, -patch_overlap:]

    return _get_best_patch(
        patches, comparison_patch, (slice(None), slice(None, patch_overlap))
    )


def _best_patch_compare_top(
    patches: Iterable[np.ndarray], comparison_patch: np.ndarray, patch_overlap: int
) -> np.ndarray:
    """ """
    # We just want to take the bottom bit of the comparison patch
    comparison_patch = comparison_patch[-patch_overlap:, :]

    return _get_best_patch(
        patches, comparison_patch, (slice(None, patch_overlap), slice(None))
    )


def _best_patch_compare_top_left(
    patches: Iterable[np.ndarray],
    left_comparison_patch: np.ndarray,
    top_comparison_patch: np.ndarray,
    patch_overlap: int,
) -> np.ndarray:
    """
    Find the best patch that matches both the left and top edges of the comparison patches.

    This double-counts the overlap region, since it compares it to both the left and top edges,
    but I think this is fine
    """
    # We want the right of the patch on the left, and the bottom of the patch on top
    left_comparison_patch = left_comparison_patch[:, -patch_overlap:]
    top_comparison_patch = top_comparison_patch[-patch_overlap:, :]

    score = float("-inf")
    best_patch = None
    for patch in patches:
        left_overlap_region = patch[:, :patch_overlap]
        top_overlap_region = patch[patch_overlap:, :]
        mse = np.sum((left_overlap_region - left_comparison_patch) ** 2) + np.sum(
            (top_overlap_region - top_comparison_patch) ** 2
        )
        if mse > score:
            score = mse
            best_patch = patch

    return best_patch


def optimally_choose_patches(
    patches: Iterable[np.ndarray],
    target_size: tuple[int, int],
    patch_overlap: int,
    allow_rotation: bool = False,
    *,
    rng: np.random.Generator,
    repeat_penalty: float = 0.0,
) -> list[np.ndarray]:
    """
    Choose patches that will at least fill the target size when stitched together, such that the overlap
    between patches is optimal.

    Chooses the first patch randomly, then builds up the first row by matching the left edge of each patch
    to the right edge of the previous patch.
    The first patch in subsequent rows are chosen according to its match with the bottom edge of the first patch,
    then the rest of the patches in the row are chosen by matching the top and left edges of the patches.

    Patches are chosen with replacement, but repeatedly using the same patch can be penalised by setting
    `repeat_penalty` to a positive value.

    :param patches: the patches to choose from
    :param target_size: the size of the desired quilt of patches
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)
    :param allow_rotation: whether to allow patches to be rotated when matching them
    :param rng: a random number generator, used to choose the first patch
    :param repeat_penalty: a penalty for using the same patch multiple times, to encourage diversity in the patches used

    :return: patches that will at least fill the target size when stitched together.
             Note that after stitching, the resulting array may be larger than the target size.

    """
    if allow_rotation:
        raise NotImplementedError(
            "Randomly choosing patches with rotation is not yet implemented"
        )

    if patch_overlap == 0:
        raise PatchError(
            "Patch overlap must be non-zero; if you want no overlap, use `randomly_choose_patches`"
        )

    if repeat_penalty < 0:
        warnings.warn(
            f"Repeat penalty {repeat_penalty} is negative; this will not penalise repeated patches",
            RuntimeWarning,
        )

    _verify_patches(patches)
    n_col, n_row = _patch_layout(target_size, patches[0].shape, patch_overlap)

    out_list = [[None for _ in range(n_col)] for _ in range(n_row)]

    # Choose the first patch randomly
    out_list[0][0] = rng.choice(patches)

    # Choose the first row by matching the left edge of each patch to the right edge of the previous patch
    for i in range(1, n_col):
        # We want to compare the overlap region on the right of the previous patch
        # with the overlap region on the left of the current patch
        comparison_patch = out_list[0][i - 1]

        # Iterate over all the patches and find the best one
        out_list[0][i] = _best_patch_compare_left(
            patches, comparison_patch, patch_overlap
        )

    # For the next rows, choose the first patch according to its match with the bottom edge of the first patch
    for i in range(1, n_row):
        # Compare the first one to the bottom edge of the first patch
        out_list[i][0] = _best_patch_compare_top(patches, out_list[0][0], patch_overlap)

        # Compare the rest of them to the top and left edges of the previous patches
        for j in range(1, n_col):
            out_list[i][j] = _best_patch_compare_top_left(
                patches, out_list[i][j - 1], out_list[i - 1][j], patch_overlap
            )

    return out_list


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
