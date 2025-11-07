"""
Input and output utilities
"""

import pathlib
from typing import Callable
from functools import cache

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from ..utils import util

GRAVITY = 9.80665


class IOError(Exception):
    """General error with I/O"""


class PowerSpectrumError(IOError):
    """Error calculating FFT or power spectrum"""


def _remove_nanmean(array: np.ndarray) -> np.ndarray:
    """
    Remove mean of an array that possibly contains NaNs
    """
    return array - np.nanmean(array)


def read_currents(path: pathlib.Path, *, remove_mean: bool = True) -> np.ndarray:
    """
    Read a .dat file holding current data and return a 720x1440 shaped numpy array giving
    the current in m/s (I think).
    Sets grid points with a value of exactly 0 to nan, since this is a flag for a grid point
    over land.

    :param path: location of the .dat file; current data is located in
                 data/projects/SING/richard_stuff/Table2/currents/ on the RDSF
    :param remove_mean: whether the mean will be removed (nan-aware)
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

    data[data == -1.9e19] = np.nan

    if remove_mean:
        data = _remove_nanmean(data)

    return np.flipud(data.reshape(shape))


@cache
def _read_clean_current_metadata(metadata_path: pathlib.Path) -> pd.DataFrame:
    """
    Read the metadata file for the clean currents .dat file

    :returns: metadata dataframe; model/name/year
    """
    # First line is the number of models/runs
    with open(metadata_path, "r") as f:
        num_runs = int(f.readline().strip())
        df = pd.read_csv(f, names=["model", "name", "year"], sep=r"\s+")

    if len(df) != num_runs:
        raise IOError(
            f"Metadata file {metadata_path} has {len(df)} rows, but first line says {num_runs}"
        )

    return df


def _coriolis_parameter(latitudes: np.ndarray) -> np.ndarray:
    """
    Calculate the coriolis parameter at each latitude
    """
    omega = 7.2921e-5
    torad = np.pi / 180.0

    return 2 * omega * np.sin(latitudes * torad)


def clipped_coriolis_param(latitudes: np.ndarray, clip_at: float) -> np.ndarray:
    """
    Calculate the coriolis parameter at each latitude, clipping it to the value at `clip_at` degrees

    Sets values at exactly 0 to + (could equally choose it to be negative, but we don't)
    """
    orig = _coriolis_parameter(latitudes)
    to_clip = np.abs(latitudes) <= clip_at

    clipped = orig.copy()
    clipped[to_clip] = np.min(np.abs(orig[~to_clip]))
    # If the original coriolis parameter was negative, make sure we keep it negative
    clipped[to_clip & (orig < 0)] *= -1

    return clipped


def _dlat_dlong(shape: tuple[int, int]) -> tuple[float, np.ndarray]:
    """
    Get the grid spacing in metres, given the shape of the grid.
    """
    torad = np.pi / 180.0
    R = 6_371_229.0

    lats, longs = util.lat_long_grid(shape)
    dlat = np.abs(lats[1] - lats[0]) * torad * R
    dlong = (
        np.abs(longs[1] - longs[0]) * torad * R * np.cos(torad * lats)[:, np.newaxis]
    )

    return dlat, dlong


def current_speed_from_mdt(mdt: np.ndarray, clip_at: float = 2.5) -> np.ndarray:
    """
    Convert geodetic MDT to currents, clipping the coriolis parameter to avoid infinities at the equator.

    By assuming geostrophic balance, we can take the gradient of the MDT to get the steady-state
    currents.
    This requires us to work out the coriolis parameter at each latitude, and to take the gradient
    of the MDT.

    :param mdt: the mean dynamic topography, in metres, covering the globe.

    :returns: the current speed in m/s

    """

    lats, _ = util.lat_long_grid(mdt.shape)

    # Find the grid spacing (in m)
    dlat, dlong = _dlat_dlong(mdt.shape)

    # Find the coriolis parameter at each latitude
    f = clipped_coriolis_param(lats, clip_at)

    # Velocities are gradients * coriolis param for geostrophic balance
    dmdt_dlat = np.gradient(mdt, axis=0) / dlat
    dmdt_dlon = np.gradient(mdt, axis=1) / dlong

    # u should be negative, but it doesnt matter for speed
    u = GRAVITY / f[:, np.newaxis] * dmdt_dlat
    v = GRAVITY / f[:, np.newaxis] * dmdt_dlon

    return np.sqrt(u**2 + v**2)


def read_clean_mdt(
    path: pathlib.Path,
    metadata_path: pathlib.Path,
    *,
    year: int,
    model: str = "ACCESS-CM2",
    name: str = "r1i1p1f1_gn",
    remove_mean: bool = True,
) -> np.ndarray:
    """
    Read clean MDT data from a .dat file.

    :param path: path to .dat file containing mean dynamic topographies as a Fortran-ordered .dat file
    :param metadata_path: path to a textfile containing the MDT metadata
    :param year: gets the MDT for this year, according to the metadata
    :param model: gets the MDT from this model, according to the metadata
    :param name: gets the with this name, according to the metadata. There's more documentation about what this
                 means somewhere, but basically don't worry about it
    :param remove_mean: whether to remove the mean (nan-aware).
    """
    metadata = _read_clean_current_metadata(metadata_path)

    # The dat file contains a header (record length), then the record, then a footer (record length)
    # We want to find the number of bytes to skip to get to the correct record, which
    # corresponds to the row number in the metadata file

    # Find the row in the metadata file
    row = metadata[
        (metadata["year"] == year)
        & (metadata["model"] == model)
        & (metadata["name"] == name)
    ]
    if len(row) == 0:
        raise ValueError(
            f"Could not find entry for {model=}, {name=}, {year=} in metadata"
        )
    if len(row) > 1:
        raise ValueError(
            f"Found multiple entries for {model=}, {name=}, {year=} in metadata"
        )

    # This tells us how many records to skip
    row_index = row.index[0]

    with open(path, "rb") as f:
        n_bytes_per_record = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Add the header + footer
        n_bytes_per_record += 8

        # Check the file is the right size, based on the metadata
        expected_size = int(n_bytes_per_record) * len(metadata)
        f.seek(0, 2)  # Seek to end of file
        actual_size = f.tell()
        if actual_size != expected_size:
            raise IOError(
                f"File size {actual_size} does not match expected {expected_size} from metadata"
            )

        offset = row_index * n_bytes_per_record

        f.seek(offset)
        header = np.fromfile(f, dtype=np.int32, count=1)[0]
        if header + 8 != n_bytes_per_record:
            raise IOError(
                f"Record length marker {header} does not match expected {n_bytes_per_record - 8}"
            )

        retval = np.fromfile(f, dtype="<f4", count=header // 4)

    retval[retval == -1.9e19] = np.nan

    if remove_mean:
        retval = _remove_nanmean(retval)

    # Make it look right
    return np.flipud(retval.reshape((720, 1440)))


def read_clean_currents(
    path: pathlib.Path,
    metadata_path: pathlib.Path,
    *,
    year: int,
    model: str = "ACCESS-CM2",
    name: str = "r1i1p1f1_gn",
    clip_at: float = 2.5,
) -> np.ndarray:
    """
    Read clean current data from a .dat file, clipping the speed to the provided value

    Read a .dat file containing clean current data,
    given the model/name/year, returning a 720x1440 numpy array giving the current
    in m/s.
    Sets land grid points to np.nan.
    Since the clean current data is stored in a large file containing multiple years and models, we need
    to choose the correct one.

    Notes on the name convention from the CMIP6 documentation can be found in docs/current_denoising/generation/ioutils.md,
    or in the original at https://docs.google.com/document/d/1h0r8RZr_f3-8egBMMh7aqLwy3snpD6_MrDz1q8n5XUk.

    :param path: location of the .dat file; clean current data is located in
                 data/projects/SING/richard_stuff/Table2/clean_currents/ on the RDSF
    :param metadata_path: location of the metadata .csv file describing the contents of the .dat file
    :param year: start of the 5-year period for which to extract data
    :param model: the climate model to use
    :param name: the model variant to use. Name follows the convention {realisation/initialisation/physics/forcing}_grid
    :param clip_at: latitude (in degrees) at which to clip the coriolis parameter

    :returns: a numpy array holding current speeds
    :raises ValueError: if the requested year/model/name is not found in the metadata
    :raises IOError: if the file is malformed, or has a different length to expected from the metadata

    """
    mdt = read_clean_mdt(path, metadata_path, year=year, model=model, name=name)

    return current_speed_from_mdt(mdt, clip_at=clip_at)


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


def tile_rms(tile: np.ndarray) -> float:
    """
    Calculate the RMS of a tile, ignoring NaNs

    :param tile: the input tile
    :returns: the RMS of the tile
    """
    return np.sqrt(np.nanmean(tile**2))


def fft_fraction(tile: np.ndarray, power_threshold: float):
    """
    Get the fraction of Fourier power that is above a threshold.

    Intended to be used with `extract_tiles` to choose tiles with a small contribution
    from low-frequency components

    :param tile: the image to calculate the power contributions for. Must be 2d and square.
    :param power_threshold: the threshold above which frequencies are considered "high",
                            as a fraction of the maximum power in the image. I.e. this is
                            a float between 0 and 1; using 0.5 will return the fraction of
                            power at freqencies greater than half the maximum.

    :returns: fraction of power in the Fourier spectrum that is above the threshold.
    :raises PowerSpectrumError: if the tile is not square or 2D.
    :raises PowerSpectrumError: if the requested power_threshold is outside the range [0, 1].
    """
    if not tile.ndim == 2 or tile.shape[0] != tile.shape[1]:
        raise PowerSpectrumError(f"tile must be 2d and square; got {tile.shape=}")
    if not 0 <= power_threshold <= 1:
        raise PowerSpectrumError(
            f"power_threshold must be as a fraction of maximum power, i.e. between 0 and 1; got {power_threshold}"
        )

    # Compute power
    f = np.fft.fft2(tile)
    fshift = np.fft.fftshift(f)
    power_spectrum = np.abs(fshift) ** 2

    # Create frequency grid
    y, x = np.indices(tile.shape)
    center_y, center_x = tile.shape[0] // 2, tile.shape[1] // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Calculate high frequency power for all windows
    high_freq_mask = r > (r.max() * 2 * power_threshold)
    total_power = power_spectrum.sum()
    high_freq_power = power_spectrum[..., high_freq_mask].sum()

    return high_freq_power / total_power


def _tile_overlaps_mask(
    index: tuple[int, int], mask: np.ndarray, tile_size: int
) -> bool:
    """
    Check if a tile starting at `index` with width/height `tile_size` overlaps any True pixels in `mask`.
    """
    return util.tile(mask, index, tile_size).any()


def extract_tiles(
    rng: np.random.Generator,
    input_img: np.ndarray,
    *,
    num_tiles: int,
    tile_criterion: Callable[[np.ndarray], bool] | None = None,
    forbidden_mask: np.ndarray | None = None,
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
    :param tile_criterion: If specified, a function that takes a tile and returns a bool, telling us whether
                           to keep the tile.
    :param forbidden_mask: an array telling us elements we don't want to include in our output masks.
                              Useful if there is a location-based criterion for selection (e.g. distance from land).
    :param max_latitude: The maximum latitude for the tiles;
                         will exclude tiles which extend above/below this latitude N/S.
                         Pass np.inf to keep all latitudes.
    :param allow_nan: Whether to allow the tiles to contain NaN (probably land) pixels.
    :param return_indices: whether to also return the location of each tile. Useful for plotting.

    :returns: A numpy array containing the extracted tiles, shaped (num_tiles, tile_size, tile_size).
    :returns: if `return_indices`, returns the location of each tile as well
    :raises IOError: if `forbidden_mask` has a different shape to `input_img`.
    :raises IOError: if the input image is not 2d
    :raises IOError: if the input image is smaller than the tile size
    """
    if input_img.ndim != 2:
        raise IOError(f"Input image must be 2d; got shape {input_img.shape}")
    if input_img.shape[0] < tile_size or input_img.shape[1] < tile_size:
        raise IOError(
            f"Tile size must be smaller than image size; got {input_img.shape} but {tile_size=}"
        )
    if forbidden_mask is not None and forbidden_mask.shape != input_img.shape:
        raise IOError(
            f"Mask shape must match img, got {input_img.shape=}; {forbidden_mask.shape=}"
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

        if forbidden_mask is not None and _tile_overlaps_mask(
            (y, x), forbidden_mask, tile_size
        ):
            continue

        tile = util.tile(input_img, (y, x), tile_size)

        if tile.shape != (tile_size, tile_size):
            raise IOError(
                f"Extracted tile has wrong shape {tile.shape}, expected {(tile_size, tile_size)}"
            )

        if not allow_nan and np.isnan(tile).any():
            continue

        # We might want to discard this tile
        if tile_criterion is not None and not tile_criterion(tile):
            continue

        # We did it! we found a tile to keep
        # Store the tile + increment the counter
        tiles[indices_found] = tile
        if return_indices:
            indices.append((y, x))
        indices_found += 1

    if return_indices:
        return tiles, indices
    return tiles


def distance_from_land(arr: np.ndarray, *, land_sentinel=np.nan):
    """
    Get the distance (in grid points) from the nearest land point.

    This is not the "real" Euclidean distance, since it doesn't account for the variation
    in size of grid point with latitude, but should be good enough for choosing the region
    of interest for picking training tiles.

    :param arr: 2d array of grid point data. N
    :param land_sentinel: value that indicates a point is land
    """
    if arr.ndim != 2:
        raise ValueError

    mask = arr != land_sentinel if not np.isnan(land_sentinel) else ~np.isnan(arr)
    return distance_transform_edt(mask)
