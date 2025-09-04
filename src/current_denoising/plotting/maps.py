"""
Utilities for plotting maps of things
"""

import numpy as np
import matplotlib.pyplot as plt


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


def imshow(
    current_grid: np.ndarray, axis: plt.Axes = None, cmap: str = "turbo"
) -> plt.Figure:
    """
    Plot a gridded field of current velocities.

    Adds a colorbar, unless we're plotting on an existing axis (in which case the user
    has control of this sort of thing).

    :param current_grid: gridded field showing current strength; land should be indicated
                         with np.nan.
    :param axis: axis to plot on; if None, a new figure and axis will be created.
    :param cmap: colormap to use. Defaults to the same one used by Laura in her paper,
                 doi:10.1017/eds.2023.41

    :returns: the figure containing the plot
    """
    new_axis = axis is None
    if new_axis:
        fig, axis = plt.subplots(figsize=(10, 5))

    lat, long = lat_long_grid(current_grid.shape)
    extent = [long[0], long[-1], lat[0], lat[-1]]
    imshow_kw = {
        "origin": "upper",
        "vmin": 0,
        "vmax": 1.4,
        "cmap": cmap,
        "extent": extent,
    }
    im = axis.imshow(current_grid, **imshow_kw)

    im.set_extent(extent)

    if new_axis:
        fig.colorbar(im)
        fig.tight_layout()

    return axis.get_figure()
