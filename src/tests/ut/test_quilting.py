"""
Tests for the quilting algorithm

"""

import pytest
import numpy as np

from current_denoising.generation import quilting


def test_ceildiv():
    """
    Test ceiling division

    """
    assert quilting._ceildiv(5, 2) == 3
    assert quilting._ceildiv(4, 2) == 2
    assert quilting._ceildiv(0, 2) == 0
    assert quilting._ceildiv(1, -2) == 0


def test_negative_patch_overlap():
    """
    Check we get an error if the patch overlap is negative

    """
    with pytest.raises(ValueError):
        quilting._patch_layout(target_size=(4, 4), patch_size=(2, 2), patch_overlap=-1)


def test_large_overlap_error():
    """
    Check we get an error if the patch overlap is larger than the patch size

    """
    with pytest.raises(ValueError):
        quilting._patch_layout(target_size=(4, 4), patch_size=(2, 2), patch_overlap=3)


def test_patch_layout_mismatch():
    """
    Check we get an error if the target size and patch size have different dimensions

    """
    with pytest.raises(ValueError):
        quilting._patch_layout(
            target_size=(4, 4, 4), patch_size=(2, 2), patch_overlap=1
        )


def test_patch_layout_exact():
    """
    Check we get the correct number of patches for an exact fit
    """
    assert quilting._patch_layout((34, 26), (10, 10), 2) == (4, 3)


def test_patch_layout_large():
    """
    Check we get the correct number of patches if the requested size
    is slightly larger than an exact fit
    """
    assert quilting._patch_layout((35, 27), (10, 10), 2) == (5, 4)


def test_patch_layout_small():
    """
    Check we get the correct number of patches if the requested size
    is slightly smaller than the patch size
    """
    assert quilting._patch_layout((33, 25), (10, 10), 2) == (4, 3)


def test_patch_size_mismatch():
    """
    Check we get an error if we pass in patches of different sizes

    """
    patches = [np.ones((2, 2)), np.ones((3, 3))]

    with pytest.raises(quilting.PatchError):
        quilting.quilt(patches, target_size=None, patch_overlap=None)


def test_3d_patches():
    """
    Check we get an error if the patches are not 2d

    """
    patches = [np.ones((2, 2, 2)), np.ones((2, 2, 2))]

    with pytest.raises(quilting.PatchError):
        quilting.quilt(patches, target_size=None, patch_overlap=None)
