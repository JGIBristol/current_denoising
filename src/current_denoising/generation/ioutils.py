"""
Input and output utilities
"""

import pathlib

import numpy as np


class IOError(Exception):
    """General error with I/O"""


def read_currents(path: pathlib.Path) -> np.ndarray:
    """
    Read a .dat file holding current data and return a 720x1440 shaped numpy array giving
    the current in m/s (I think)

    :param path: location of the .dat file; current data is located in
                 data/projects/SING/richard_stuff/Table2/currents/ on the RDSF
    :returns: a numpy array holding current speed
    :raises IOError: if the file is malformed
    """
    dtype = np.dtype(
        "=f4"
    )  # Not sure about the endianness, but it doesn't seem to matter...
    shape = 720, 1440  # Assuming a quarter degree grid

    type_size = np.dtype(dtype).itemsize

    with open(path, "rb") as f:
        first_record_len = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=dtype, count=first_record_len // type_size)

        closing_record_len = np.fromfile(f, dtype=np.int32, count=1)[0]
        if closing_record_len != first_record_len:
            raise IOError("Close marker does not match the opener.")

    return data.reshape(shape)


def _included_indices(
    n_rows: int, tile_size: int, max_latitude: float
) -> tuple[int, int]:
    """
    Find the range of y-indices to select from, given the size of the input image
    and the maximum latitude.

    Assumes the input image is centred on the equator and ranges from -90 to 90 degrees latitude.
    """
    if max_latitude <= 0:
        raise IOError("Maximum latitude must be > 0")

    # Given the number of rows in the image, find which latitude each row corresponds to
    latitudes = np.linspace(90, -90, n_rows, endpoint=True)

    # Check if we have enough allowed latitudes
    allowed_latitudes = np.sum((latitudes < max_latitude) & (latitudes > -max_latitude))
    if allowed_latitudes < tile_size:
        raise IOError(
            f"Not enough allowed latitudes ({allowed_latitudes}) to fit a tile of size {tile_size}"
        )

    # Find the first latitude that is <= to the provided maximum
    for min_row in range(n_rows):
        if max_latitude >= latitudes[min_row]:
            break
    else:
        raise RuntimeError("No rows found below the provided maximum latitude")

    # Find the last latitude that is >= - the provided minimum
    for max_row in range(min_row + 1, n_rows - tile_size + 2):
        index = max_row + tile_size - 1
        # If we have fallen off the bottom of the image, take the last row
        if index == n_rows:
            break

        # If the bottom of the tile lies on exactly the threshold, take this row
        if latitudes[index] == -max_latitude:
            break

        # If the bottom of the tile is less than the threshold, take the previous row
        if latitudes[index] < -max_latitude:
            max_row -= 1
            break
    # Don't raise here if we don't find a row above the max latitude - that's fine, we'll just fall
    # through and take the last row

    return min_row, max_row


def _tile_index(
    rng, *, input_size: tuple[int, int], max_latitude: float, tile_size: int
) -> tuple[int, int]:
    """
    Generate random (y, x) indices for the top-left corner of a tile within an image

    :param input_img_size: The size of the input image
    :param tile_size: The size of each tile.
    :param max_latitude: The maximum latitude for the tiles;
                         will exclude tiles which extend above/below this latitude N/S.

    :returns: A tuple of (y, x) indices
    """
    height_range = _included_indices(input_size[0], tile_size, max_latitude)
    width_range = (0, input_size[1] - tile_size + 1)

    y_index = int(rng.integers(*height_range))
    x_index = int(rng.integers(*width_range))

    return y_index, x_index


def _tile(input_img: np.ndarray, start: tuple[int, int], size: int) -> np.ndarray:
    """
    Extract a tile at the provided location + size from a 2d array
    """
    return input_img[start[0] : start[0] + size, start[1] : start[1] + size]


def _tile_rms(tile: np.ndarray) -> float:
    """
    Calculate the RMS of a tile, ignoring NaNs

    :param tile: the input tile
    :returns: the RMS of the tile
    """
    return np.sqrt(np.nanmean(tile**2))


def extract_tiles(
    rng: np.random.Generator,
    input_img: np.ndarray,
    *,
    num_tiles: int,
    max_rms: float,
    max_latitude: float = 64.0,
    tile_size: int = 32,
    allow_nan: bool = False,
    return_indices: bool = False,
) -> np.ndarray:
    """
    Randomly extract tiles from an input image.

    Tiles are selected such that no part of any tile exceeds the provided maximum latitude;
    this assumes the input image is centred on the equator and ranges from -90 to 90 degrees latitude.

    :param rng: a seeded numpy random number generator
    :param input_img: The input image from which to extract tiles.
    :param tile_size: The size of each tile.
    :param num_tiles: The number of tiles to extract.
    :param max_rms: Maximum allowed RMS value in a tile
    :param max_latitude: The maximum latitude for the tiles;
                         will exclude tiles which extend above/below this latitude N/S.
    :param allow_nan: Whether to allow the tiles to contain NaN (probably land) pixels.
    :param return_indices: whether to also return the location of each tile

    :returns: A numpy array containing the extracted tiles.
    :returns: if `return_indices`, returns the location of each tile as well
    :raises IOError: if the input image is not 2d
    :raises IOError: if the input image is smaller than the tile size
    """
    if input_img.ndim != 2:
        raise IOError(f"Input image must be 2d; got shape {input_img.shape}")
    if input_img.shape[0] < tile_size or input_img.shape[1] < tile_size:
        raise IOError(
            f"Tile size must be smaller than image size; got {input_img.shape} but {tile_size=}"
        )

    # Choose the range of indices to pick from
    height_range = slice(0, input_img.shape[0] - tile_size + 1)
    width_range = slice(0, input_img.shape[1] - tile_size + 1)

    tiles = np.empty((num_tiles, tile_size, tile_size), dtype=input_img.dtype)
    indices_found = 0
    indices = []
    while indices_found < num_tiles:
        y, x = _tile_index(
            rng,
            input_size=input_img.shape,
            max_latitude=max_latitude,
            tile_size=tile_size,
        )

        tile = _tile(input_img, (y, x), tile_size)

        if tile.shape != (tile_size, tile_size):
            raise IOError(
                f"Extracted tile has wrong shape {tile.shape}, expected {(tile_size, tile_size)}"
            )

        if not allow_nan and np.isnan(tile).any():
            continue

        # Check the RMS of the tile
        if _tile_rms(tile) < max_rms:
            tiles[indices_found] = tile
            indices_found += 1

            if return_indices:
                indices.append((y, x))

    if return_indices:
        return tiles, indices
    return tiles
