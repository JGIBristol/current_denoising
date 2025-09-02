"""
Utilities for applying noise to data

"""

import numpy as np
from scipy.ndimage import gaussian_filter


def noise_strength_map(current_grid: np.ndarray, *, filter_size: int) -> np.ndarray:
    """
    Get the expected noise strength map from a gridded field of current
    velocities.

    :param current_grid: gridded field showing current strength; land should be indicated
                         with np.nan.
    :param filter_size: size of Gaussian filter to apply, in grid points
    :returns: a grid of the same shape as current_grid, showing the expected noise strength
              at that grid point. Land values are set to np.nan.
    """
    land_mask = np.isnan(current_grid)

    current_grid = np.where(land_mask, 0.0, current_grid)

    filtered = gaussian_filter(current_grid, sigma=filter_size)
    filtered[land_mask] = np.nan

    return filtered
