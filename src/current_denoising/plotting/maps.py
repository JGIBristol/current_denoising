"""
Utilities for plotting maps of things
"""

import functools

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from ..utils import util


def clear2black_cmap() -> colors.Colormap:
    """
    Colormap that varies from clear to black
    """
    c_white = colors.colorConverter.to_rgba("white", alpha=0)
    c_black = colors.colorConverter.to_rgba("black", alpha=1)
    return colors.ListedColormap([c_white, c_black], "clear2black")


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

    lat, long = util.lat_long_grid(current_grid.shape)
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
