"""
Utilities for plotting maps of things
"""

import numpy as np


def lat_long_grid(img_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate latitude and longitude arrays for an image of the provided shape

    :param img_shape: The shape of the image (height, width). Assumes that this image
                      contains equally spaced grid points, and that they are centred
                      such that the grid centres run from e.g. -90+1/8 to 90-1/8 for a
                      quarter-degree grid.
    :returns: latitudes
    :returns: longitudes

    """
    height, width = img_shape

    # Find how many degrees each grid point corresponds to
    lat_point_size = 180.0 / height
    long_point_size = 360.0 / width

    return np.linspace(
        -90 + lat_point_size / 2, 90 - lat_point_size / 2, height, endpoint=True
    ), np.linspace(
        -180 + long_point_size / 2, 180 - long_point_size / 2, width, endpoint=True
    )
