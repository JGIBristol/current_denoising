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
def unfilled_image():
    """Initialise a 10x10 patch with unfilled pixels"""
    return quilting._unfilled_image((10, 10))


@pytest.fixture
def simple_candidate_patch():
    """patch with values increasing"""
    return np.arange(9, dtype=np.float32).reshape((3, 3))


def test_vertical_overlap_cost(unfilled_image, simple_candidate_patch):
    """
    Check we get the right cost map for a vertical overlap
    """
    # Add a square of numbers in the top left corner of the existing patch
    # This will look like
    # 0 2 4
    # 6 8 10
    # 12 14 16
    unfilled_image[0:3, 0:3] = simple_candidate_patch * 2

    # Overlap the new patch on the right of the existing one
    pos = (0, 2)

    cost_map = quilting.overlap_cost(unfilled_image, simple_candidate_patch, pos)

    expected_cost = np.ones_like(simple_candidate_patch) * np.inf
    expected_cost[:, 0] = [16, 49, 100]

    np.testing.assert_array_equal(cost_map, expected_cost)


def test_horizontal_overlap_cost(unfilled_image, simple_candidate_patch):
    """
    Check we get the right cost map for a horizontal overlap
    """
    # Add a square of numbers in the top left corner of the existing patch
    # This will look like
    # 0 2 4
    # 6 8 10
    # 12 14 16
    unfilled_image[0:3, 0:3] = simple_candidate_patch * 2

    pos = (2, 0)

    cost_map = quilting.overlap_cost(unfilled_image, simple_candidate_patch, pos)

    expected_cost = np.full((3, 3), np.inf)
    expected_cost[0] = [144, 169, 196]

    np.testing.assert_array_equal(cost_map, expected_cost)


def test_corner_overlap_cost(unfilled_image, simple_candidate_patch):
    """
    Check we get the right cost map for a corner overlap
    """
    # Add a square of numbers in the top left corner of the existing patch
    # This will look like
    # 0 2 4
    # 6 8 10
    # 12 14 16
    unfilled_image[0:3, 0:3] = simple_candidate_patch * 2

    pos = (1, 1)

    cost_map = quilting.overlap_cost(unfilled_image, simple_candidate_patch, pos)

    expected_cost = np.array(
        [
            [64, 81, np.inf],
            [121, 144, np.inf],
            [np.inf, np.inf, np.inf],
        ]
    )

    np.testing.assert_array_equal(cost_map, expected_cost)


def test_invalid_graphs():
    """
    Check we raise an error if the graph is invalid
    """
    # Invalid start
    with pytest.raises(quilting.GraphConstructionError):
        quilting.cost_to_graph(
            np.array([[np.inf, 2], [np.inf, 4]]), start="left", end="right"
        )

    # Invalid end
    with pytest.raises(quilting.GraphConstructionError):
        quilting.cost_to_graph(
            np.array([[1, 2], [np.inf, np.inf]]), start="top", end="bottom"
        )


def test_simple_cost_to_graph(simple_candidate_patch):
    """
    Check we get the right graph from a simple cost matrix with no infinities

    Horizontal traversal
    """
    expected_graph = {
        "START": {
            ((0, 0), 0),
            ((1, 0), 3),
            ((2, 0), 6),
        },
        (0, 0): {
            ((1, 0), 3),
            ((0, 1), 1),
        },
        (0, 1): {
            ((0, 0), 0),
            ((0, 2), 2),
            ((1, 1), 4),
        },
        (0, 2): {
            ((0, 1), 1),
            ((1, 2), 5),
            ("END", 0),
        },
        (1, 0): {
            ((0, 0), 0),
            ((1, 1), 4),
            ((2, 0), 6),
        },
        (1, 1): {
            ((0, 1), 1),
            ((1, 0), 3),
            ((1, 2), 5),
            ((2, 1), 7),
        },
        (1, 2): {
            ((0, 2), 2),
            ((1, 1), 4),
            ((2, 2), 8),
            ("END", 0),
        },
        (2, 0): {
            ((1, 0), 3),
            ((2, 1), 7),
        },
        (2, 1): {
            ((1, 1), 4),
            ((2, 0), 6),
            ((2, 2), 8),
        },
        (2, 2): {
            ((1, 2), 5),
            ((2, 1), 7),
            ("END", 0),
        },
    }

    graph = quilting.cost_to_graph(simple_candidate_patch, start="left", end="right")

    assert graph == expected_graph


def test_cost_to_graph(simple_candidate_patch):
    """
    Check we get the right graph if the cost matrix has infinities
    """
    # Add some infinities to make the graph more complex
    simple_candidate_patch[1, 1] = np.inf
    simple_candidate_patch[1, 2] = np.inf
    simple_candidate_patch[2, 2] = np.inf

    expected_graph = {
        "START": {
            ((2, 0), 6),
            ((2, 1), 7),
        },
        (2, 0): {
            ((2, 1), 7),
            ((1, 0), 3),
        },
        (2, 1): {
            ((2, 0), 6),
        },
        (1, 0): {
            ((0, 0), 0),
            ((2, 0), 6),
        },
        (0, 0): {
            ("END", 0),
            ((1, 0), 3),
            ((0, 1), 1),
        },
        (0, 1): {
            ("END", 0),
            ((0, 0), 0),
            ((0, 2), 2),
        },
        (0, 2): {
            ("END", 0),
            ((0, 1), 1),
        },
    }

    graph = quilting.cost_to_graph(simple_candidate_patch, start="bottom", end="top")

    assert graph == expected_graph


def test_diagonal_graph(simple_candidate_patch):
    """
    Check we can correctly build a graph for a cost matrix where we will traverse from
    the left to the top edge
    """
    # Add some infinities to make the graph viable
    simple_candidate_patch[0, 0] = np.inf
    simple_candidate_patch[1, 2] = np.inf
    simple_candidate_patch[2, 1] = np.inf
    simple_candidate_patch[2, 2] = np.inf

    expected_graph = {
        "START": {
            ((1, 0), 3.0),
            ((2, 0), 6.0),
        },
        (0, 1): {
            ("END", 0.0),
            ((0, 2), 2.0),
            ((1, 1), 4.0),
        },
        (0, 2): {
            ("END", 0.0),
            ((0, 1), 1.0),
        },
        (1, 0): {
            ((1, 1), 4.0),
            ((2, 0), 6.0),
        },
        (1, 1): {
            ((0, 1), 1.0),
            ((1, 0), 3.0),
        },
        (2, 0): {
            ((1, 0), 3.0),
        },
    }

    graph = quilting.cost_to_graph(simple_candidate_patch, start="left", end="top")

    assert graph == expected_graph


def test_traversal():
    """
    Test that we can traverse the graph correctly
    """
    # Traversing from bottom to top:
    # 2 0 0
    # 2 . 9
    # 1 3 0
    graph = {
        "START": {((2, 0), 1), ((2, 1), 3), ((2, 2), 0)},
        (0, 0): {((0, 1), 0), ((1, 0), 2), ("END", 0)},
        (0, 1): {((0, 0), 2), ((0, 2), 0), ("END", 0)},
        (0, 2): {((0, 1), 0), ((1, 2), 9), ("END", 0)},
        (1, 0): {((0, 0), 2), ((2, 0), 1)},
        (1, 2): {((0, 2), 0), ((2, 2), 0)},
        (2, 0): {((1, 0), 2), ((2, 1), 3)},
        (2, 1): {((2, 0), 1), ((2, 2), 0)},
        (2, 2): {((1, 2), 9), ((2, 1), 3)},
    }

    expected_path = [(2, 0), (1, 0), (0, 0)]

    assert quilting.shortest_path(graph) == expected_path


def test_traverse_array():
    """
    Check that we get the expected path from an array
    """
    a = np.inf
    cost_array = np.array(
        [
            [8, 0, 2, 1, 1, 1],
            [4, 3, 1, 1, 3, 1],
            [2, 2, 1, a, a, a],
            [1, 1, 1, a, a, a],
            [3, 1, 4, a, a, a],
        ],
        dtype=np.float32,
    )

    expected_path = [
        (4, 1),
        (3, 1),
        (3, 2),
        (2, 2),
        (1, 2),
        (1, 3),
        (0, 3),
        (0, 4),
        (0, 5),
    ]

    assert quilting.seam_nodes(cost_array, start="bottom", end="right") == expected_path


def test_add_patch_left(unfilled_image):
    """
    Check we can correctly add a patch to the left of an existing image
    The seam here has 0 cost

    """
    # Add a patch to the top left of the existing image
    unfilled_image[0:3, 0:3] = np.arange(1, 10, dtype=np.float32).reshape((3, 3))

    patch = np.array(
        [
            [3, 3, 1],
            [5, 6, 1],
            [8, 7, 1],
        ]
    )

    expected_array = quilting._unfilled_image(unfilled_image.shape)
    expected_array[0:3, 0:4] = np.array(
        [
            [1, 2, 3, 1],
            [4, 5, 6, 1],
            [7, 8, 7, 1],
        ]
    )

    np.testing.assert_array_equal(
        quilting.add_patch(unfilled_image, patch, (0, 1)), expected_array
    )


def test_add_patch_below(unfilled_image):
    """
    Check we can correctly add a patch below an existing image
    The seam here has nonzero cost, so pixels along it should be averaged

    """
    unfilled_image[0:3, 0:3] = np.arange(1, 10, dtype=np.float32).reshape((3, 3))
    patch = np.array(
        [
            [2, 3, 100],
            [100, 6, 7],
            [1, 1, 1],
        ]
    )

    expected_array = quilting._unfilled_image(unfilled_image.shape)
    expected_array[0:4, 0:3] = np.array(
        [
            [1, 2, 3],
            [3, 4, 6],
            [7, 7, 8],
            [1, 1, 1],
        ]
    )

    np.testing.assert_array_equal(
        quilting.add_patch(unfilled_image, patch, (1, 0)), expected_array
    )


def test_add_patch_diag(unfilled_image):
    """
    Check we can correctly add a patch diagonally to an existing image

    """
    # Build up a slightly more complicated image
    unfilled_image[0:6, 0:6] = np.arange(36, dtype=np.float32).reshape((6, 6))
    unfilled_image[6:9, 0:3] = np.arange(36, 45).reshape((3, 3))

    position = (4, 0)
    patch = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 30, 31, 34, 35],
            [0, 0, 36, 10, 10, 10],
            [0, 40, 41, 10, 10, 10],
            [0, 41, 0, 10, 10, 10],
        ]
    )

    expected_array = quilting._unfilled_image(unfilled_image.shape)
    expected_array[0:5, 0:6] = np.arange(30).reshape((5, 6))
    expected_array[5:9, 0:6] = np.array(
        [
            [30, 31, 31, 32, 34, 35],
            [36, 37, 37, 10, 10, 10],
            [39, 40, 41, 10, 10, 10],
            [42, 42, 0, 10, 10, 10],
        ]
    )

    actual_array = quilting.add_patch(unfilled_image, patch, position)

    np.testing.assert_array_equal(actual_array, expected_array)
