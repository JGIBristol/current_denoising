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


def test_patch_verification():
    """
    Check that _verify_patches raises an error if patches are not all the same size or not 2d

    """
    quilting._verify_patches([np.ones((2, 2)), np.ones((2, 2))])

    with pytest.raises(quilting.PatchSizeMismatchError):
        quilting._verify_patches([np.ones((2, 2)), np.ones((3, 3))])

    with pytest.raises(quilting.PatchSizeMismatchError):
        quilting._verify_patches([np.ones((2, 2, 2)), np.ones((2, 2, 2))])


@pytest.fixture
def simple_existing_patch():
    """Initialise a 10x10 patch with unfilled pixels"""
    return quilting._unfilled_image((10, 10))


@pytest.fixture
def simple_candidate_patch():
    """patch with values increasing"""
    return np.arange(9).reshape((3, 3))


def test_vertical_overlap_cost(simple_existing_patch, simple_candidate_patch):
    """
    Check we get the right cost map for a vertical overlap
    """
    # Add a square of numbers in the top left corner of the existing patch
    # This will look like
    # 0 2 4
    # 6 8 10
    # 12 14 16
    simple_existing_patch[0:3, 0:3] = simple_candidate_patch * 2

    # Overlap the new patch on the right of the existing one
    pos = (0, 2)

    cost_map = quilting.overlap_cost(simple_existing_patch, simple_candidate_patch, pos)

    expected_cost = np.ones_like(simple_candidate_patch) * np.inf
    expected_cost[:, 2] = [4, 25, 64]

    np.testing.assert_array_equal(cost_map, expected_cost)


def test_horizontal_overlap_cost(simple_existing_patch, simple_candidate_patch):
    """
    Check we get the right cost map for a horizontal overlap
    """
    # Add a square of numbers in the top left corner of the existing patch
    # This will look like
    # 0 2 4
    # 6 8 10
    # 12 14 16
    simple_existing_patch[0:3, 0:3] = simple_candidate_patch * 2

    pos = (2, 0)

    cost_map = quilting.overlap_cost(simple_existing_patch, simple_candidate_patch, pos)

    expected_cost = np.full((3, 3), np.inf)
    expected_cost[2] = [36, 49, 64]

    np.testing.assert_array_equal(cost_map, expected_cost)


def test_corner_overlap_cost(simple_existing_patch, simple_candidate_patch):
    """
    Check we get the right cost map for a corner overlap
    """
    # Add a square of numbers in the top left corner of the existing patch
    # This will look like
    # 0 2 4
    # 6 8 10
    # 12 14 16
    simple_existing_patch[0:3, 0:3] = simple_candidate_patch * 2

    pos = (1, 1)

    cost_map = quilting.overlap_cost(simple_existing_patch, simple_candidate_patch, pos)

    expected_cost = np.array(
        [
            [16, 25, np.inf],
            [49, 64, np.inf],
            [np.inf, np.inf, np.inf],
        ]
    )

    np.testing.assert_array_equal(cost_map, expected_cost)
