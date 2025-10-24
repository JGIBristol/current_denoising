"""
Tests for the MDT specific utilities module
"""

import pytest
import numpy as np

from current_denoising.generation import mdt


def test_wrong_dimension():
    """
    Check we get the correct error if we pass a 1- or 3-d array
    """
    with pytest.raises(mdt.FillNaNError):
        mdt.fill_nan_with_nearest(np.ones(3))
    with pytest.raises(mdt.FillNaNError):
        mdt.fill_nan_with_nearest(np.ones((3, 3, 3)))


def test_all_nan():
    """
    Check we get the right exception
    """
    with pytest.raises(mdt.FillNaNError):
        mdt.fill_nan_with_nearest(np.ones((3, 3), dtype=float) * np.nan)


def test_mdt_nonans():
    """
    Check we get the same array back if there are no NaNs present
    """
    in_arr = np.random.random((5, 5))
    np.testing.assert_array_equal(mdt.fill_nan_with_nearest(in_arr), in_arr)


def test_simple_filling():
    """
    Check we get the right value for a constant array where one value is NaN
    """
    in_arr = np.ones((3, 3), dtype=float)
    in_arr[1, 1] = np.nan

    np.testing.assert_array_equal(
        mdt.fill_nan_with_nearest(in_arr), np.ones_like(in_arr)
    )


def test_square_filling():
    """
    Check we get the expected values for an array with multiple NaNs, some of
    which are equidistant from different points
    """
    in_arr = np.ones((5, 5), dtype=float) * np.nan
    in_arr[:, -1] = 3
    in_arr[-1] = 2
    in_arr[:, 0] = 1
    in_arr[0] = 0

    # Priority order is left/top/bottom/right
    expected_out = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 3],
            [1, 1, 1, 3, 3],
            [1, 1, 2, 2, 3],
            [1, 2, 2, 2, 2],
        ],
        dtype=float,
    )

    np.testing.assert_array_equal(mdt.fill_nan_with_nearest(in_arr), expected_out)
