"""
Utilities for dealing with the MDT directly
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


class MDTError(Exception):
    """General base exception"""


class FillNaNError(Exception):
    """Error filling in NaN values"""


def fill_nan_with_nearest(input_arr: np.ndarray) -> np.ndarray:
    """
    Returns a copy of the input array where NaN values are populated by their nearest
    non-NaN neighbour.

    If two pixels are equally close (e.g. on a corner) the priority ordering is
    left-top-bottom-right, but this might be implementation dependent and cause
    the tests to fail sometimes. If this happens, it's fine. But maybe change the test.

    Uses Euclidean distance on the grid (so is slighly inaccurate for a global gridded
    field), but should be good enough since this is only really important around coastlines
    for when we perform the Gaussian smoothing.

    :param input_arr: 2d input array
    :return: an equal-shaped array, where NaN values replaced with their nearest neighbours
    :raises: FillNaNError if the input is not 2-d or if the entire array is NaN

    """
    if input_arr.ndim != 2:
        raise FillNaNError(f"Array must be 2d, got {input_arr.ndim=}")

    nan_mask = np.isnan(input_arr)

    if np.all(nan_mask):
        raise FillNaNError(f"Image only contains NaN; cannot fill anything in!")

    if not np.any(nan_mask):
        return input_arr

    _, indices = distance_transform_edt(nan_mask, return_indices=True)

    filled = input_arr.copy()
    filled[nan_mask] = input_arr[tuple(indices[:, nan_mask])]
    return filled
