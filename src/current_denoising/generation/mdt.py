"""
Utilities for dealing with the MDT directly
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter1d

from ..utils import util


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


def gauss_smooth(grid: np.ndarray, kernel_size_km: float) -> np.ndarray:
    """
    Approximately smooth a grid with the provided kernel size in km.
    Assumes the grid is a global gridded field (i.e. corresponds to a full 180 degrees of latitude, and 360 of longitude)

    Uses a variable-sized kernel that shrinks near the poles to try to maintain an equal spatial size
    as latitude varies. This is not exact - the shape of the kernel will be distorted, especially near the poles,
    as the kernel is radially symmetric on the Cartesian grid, but it should be good enough for our purposes.

    To smooth grids containing NaNs, consider using `fill_nan_with_nearest` first.

    :param grid: the grid to smooth
    :param kernel_size: the width of the Gaussian kernel in km.

    :return: the smoothed grid.

    """
    lats, longs = util.lat_long_grid(grid.shape)

    dlat = lats[1] - lats[0]
    dlong = longs[1] - longs[0]

    sigma_lat = kernel_size_km / (util.KM_PER_DEG * dlat)
    sigma_long = kernel_size_km / (
        util.KM_PER_DEG * dlong * util.cos_latitudes(grid.shape[0])
    )
    assert len(sigma_long) == grid.shape[0], f"{len(sigma_long)}, {grid.shape=}"

    tmp = gaussian_filter1d(grid, sigma=sigma_lat, axis=0, mode="wrap")

    smoothed = np.empty_like(grid)
    for i, sigma in enumerate(sigma_long):
        smoothed[i, :] = gaussian_filter1d(tmp[i, :], sigma=sigma, mode="wrap")

    return smoothed
