"""
General utilities
"""

import inspect
import functools
from typing import Callable

import numpy as np

KM_PER_DEG = 111.32
"""Size of a degree on the grid (at the equator) in km"""


class UtilError(Exception):
    """General base class"""


class LatLongError(UtilError):
    """General exception for latitude/longitude calculation errors"""


class TileError(UtilError):
    """Something went wrong splitting into tiles"""


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


def latlong2index(
    lat: float, long: float, grid_size: tuple[int, int]
) -> tuple[int, int]:
    """
    Convert a latitude and longitude to indices on a whole-Earth grid of the specified size.

    :param lat: latitude
    :param long:
    :param grid_size: the shape of the grid array, e.g. (1440, 720) for a quarter-degree grid.

    :returns: indices corresponding to the closest grid point at the provided latitude and longitude.
    """
    lats, longs = lat_long_grid(grid_size)

    # Reverse lats since the grid runs south -> north but the
    # image is stored north -> south
    lat_idx = int(np.argmin(np.abs(-lats - lat)))
    long_idx = int(np.argmin(np.abs(longs - long)))

    return (lat_idx, long_idx)


def tile(input_img: np.ndarray, start: tuple[int, int], size: int) -> np.ndarray:
    """
    Extract a tile at the provided location + size from a 2d array

    """
    return input_img[start[0] : start[0] + size, start[1] : start[1] + size]


def tiles_from_indices(
    input_img: np.ndarray, indices: list[tuple[float, float]], size: int
) -> np.ndarray:
    """
    Extract a series of square tiles from an image,
    given their size and the indices of their top-left corners.

    :param input_img: the 2d numpy array to extract tiles from.
    :param indices: the (y, x) co-ordinates of each tile's top left corner,
                    in grid points
    :param size: the side length of the tile, in grid points

    :returns: an (N, size, size) shaped array holding the extracted tiles
    :raises TileError: if any of the co-ordinates are too close to the edges
                       such that the extracted tiles will come out the wrong shape
    """
    max_y = [i + size for i, _ in indices]
    max_x = [j + size for _, j in indices]
    if any(y > len(input_img) for y in max_y) or any(
        x > input_img.shape[1] for x in max_x
    ):
        raise TileError(
            f"Cannot extract {size} sized tiles from {input_img.shape} array\n"
            f"\t indices:\n{indices}"
        )

    return np.stack([tile(input_img, i, size) for i in indices])


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

    lat_idx, long_idx = latlong2index(lat, long, square_grid.shape)

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


def _arg_in_signature(fcn: Callable, arg: str) -> bool:
    """
    Check whether the provided arg is in the list of function args
    """
    return arg in inspect.signature(fcn).parameters


def apply_to_sliding_window(
    array: np.ndarray, fcn: Callable[[np.ndarray], float], window_size: int
) -> np.ndarray:
    """
    Apply a function to a 2d array in sliding windows.

    Applies the given function to (window_size x window_size) square sliding
    windows and returns the result. fcn must take a square window as its only argument
    and return a number.

    Pads the result on the right and bottom edges (i.e. the last rows and columns) with NaN.

    :param array: the array to apply the function to
    :param fcn: the function to apply to the windows. Must take an "axis" argument,
                and have a result that can be cast to a float.
    :param window_size: size of the square windows to be extracted from the array

    :return: an array with the same shape as `array` where each entry is the result of
             applying `fcn` to each window in `array`. Padded with NaNs.
    :raises UtilError: if the window size is too large, or the function does not take an
                       `axis` argument
    """
    if any([window_size > x for x in array.shape]):
        raise UtilError(
            f"Cannot create windows with size {window_size} from {array.shape=}"
        )
    if not _arg_in_signature(fcn, "axis"):
        raise UtilError(
            f"function {fcn} does not accept an 'axis' argument; required to broadcast over a sliding window"
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        array, (window_size, window_size)
    )

    retval = fcn(windows, axis=(-2, -1)).astype(float)

    pad_width = window_size - 1
    return np.pad(
        retval,
        ((0, pad_width), (0, pad_width)),
        mode="constant",
        constant_values=np.nan,
    )


def split_into_tiles(
    input_img: np.ndarray, tile_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop and reshape an (N, M) shaped image into an (K, tile_size, tile_size) array,
    cropping the image on its right and bottom edges.

    Splits the image into non-overlapping square tiles; doesn't share memory with the
    underlying array.
    Returns an array holding the 2d tiles, and also the indices giving their original
    locations (using their top-left corner) so the input image can be reconstructed from the tiles.

    :param input_img: the input grided field (e.g. an MDT) to split into tiles.
    :param tile_size: the side length of the tile to split the image into.

    :returns: an (k, tile_size, tile_size) reshaped array.
    :returns: a list of indices giving the location in `input_img` of the top-left corner of each tile
    :raises TileError: if the tile size is larger than the input (along any axis), if the
                       input image is not 2d or if `tile_size` is not an integer.
    """
    if input_img.ndim != 2:
        raise TileError(f"Got {input_img.shape=}")
    if any((tile_size > s for s in input_img.shape)):
        raise TileError(f"Got {input_img.shape=} but {tile_size=}; tiles too big")

    # Otherwise the reshape might break
    if not isinstance(tile_size, int):
        raise TileError(f"`tile_size` must be an integer, got {type(tile_size)=}")

    n_vert = input_img.shape[0] // tile_size
    n_horiz = input_img.shape[1] // tile_size

    # Crop the array to the right size
    cropped_img = input_img[0 : n_vert * tile_size, 0 : n_horiz * tile_size]

    # Get the locations of the tiles
    rr, cc = np.meshgrid(
        np.arange(0, n_vert * tile_size, tile_size),
        np.arange(0, n_horiz * tile_size, tile_size),
        indexing="ij",
    )
    locations = np.column_stack((rr.ravel(), cc.ravel()))

    # Get the tiles
    tiles = (
        cropped_img.reshape((n_vert, tile_size, n_horiz, tile_size))
        .swapaxes(1, 2)
        .reshape(n_vert * n_horiz, tile_size, tile_size)
    )

    return tiles, locations
