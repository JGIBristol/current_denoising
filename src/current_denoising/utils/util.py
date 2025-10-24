"""
General utilities
"""

import functools
import numpy as np

KM_PER_DEG = 111.32
"""Size of a degree on the grid (at the equator) in km"""


class LatLongError(Exception):
    """General exception for latitude/longitude calculation errors"""


def grid_point_size(height: int, width: int) -> tuple[float, float]:
    """
    Calculate the size of each grid point in degrees, given the shape of the grid.

    :param height: number of grid points in the latitude direction
    :param width: number of grid points in the longitude direction

    :returns: (lat_point_size, long_point_size)
    """
    lat_point_size = 180.0 / height
    long_point_size = 360.0 / width

    return lat_point_size, long_point_size


def tile(input_img: np.ndarray, start: tuple[int, int], size: int) -> np.ndarray:
    """
    Extract a tile at the provided location + size from a 2d array

    """
    return input_img[start[0] : start[0] + size, start[1] : start[1] + size]


def get_tile(
    square_grid: np.ndarray, co_ords: tuple[int, int], tile_size: int
) -> np.ndarray:
    """
    Extract a tile from a square grid, given its location and size.

    The tile will start at the closest grid point to the provided coordinates;
    if two points are equally close, it will choose the northernmost/westernmost point.

    :param square_grid: The input grid from which to extract the tile. Must be a square grid; i.e.
                 the spacing in latitude and longitude must be the same.
    :param co_ords: A tuple of (lat, long) for the top-left corner of the tile.
                    Must be in ([-90, 90], [-180, 180]).
                    Requesting a lat/long right on the edge (e.g. 90 or -180) will return
                    the edgemost grid point, which is not at exactly that lat/long - the first
                    grid point is e.g. at (90 - grid_size/2).
    :param tile_size: size of the tile in degrees

    :returns: the tile as a view into `grid`.
    """
    lat_point_size, long_point_size = grid_point_size(*square_grid.shape)
    if lat_point_size != long_point_size:
        raise LatLongError(
            f"Grid points are not square: {lat_point_size} x {long_point_size} deg"
        )

    if (tile_size % lat_point_size) or (tile_size % long_point_size):
        raise LatLongError(
            f"Tile size {tile_size} is not a multiple of grid point size "
            f"({lat_point_size}, {long_point_size}) for grid shape {square_grid.shape}"
        )

    lat, long = co_ords
    if abs(lat) > 90 or abs(long) > 180:
        raise LatLongError(
            f"Requested lat/long {co_ords} is out of range (+-90, +-180)"
        )

    lats, longs = lat_long_grid(square_grid.shape)

    lat_idx = int(np.argmin(np.abs(-lats - lat)))
    long_idx = int(np.argmin(np.abs(longs - long)))

    # Convert tile size in degrees to number of grid points
    extent = int(tile_size / lat_point_size)

    tile_ = tile(square_grid, (lat_idx, long_idx), extent)

    assert tile_.shape == (extent, extent), (
        f"Tile shape {tile_.shape} is not as expected {(tile_size, tile_size)}\n"
        f"Out of bounds for grid shape {square_grid.shape}, co_ords {co_ords}?"
        f" (got index {lat_idx, long_idx}, extent {extent})"
    )

    return tile_


@functools.cache
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
    lat_point_size, long_point_size = grid_point_size(height, width)

    return np.linspace(
        -90 + lat_point_size / 2, 90 - lat_point_size / 2, height, endpoint=True
    ), np.linspace(
        -180 + long_point_size / 2, 180 - long_point_size / 2, width, endpoint=True
    )


def cos_latitudes(n_points: int) -> np.ndarray:
    """
    Given a number of latitude points to cover the Earth, find the cosine of each of these latitudes.

    E.g. if we pass 720 (for a quarter-degree grid), returns cos(-90+1/8 deg), ... , cos(90-1/8deg)

    :param n_points: number of points in the second axis of a gridded field (i.e., number of latitude points)
    :return: cosine of all the latitudes
    """
    lat, _ = lat_long_grid((n_points, 1))

    return np.cos(np.deg2rad(lat))
