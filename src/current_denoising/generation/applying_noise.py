"""
Utilities for applying noise to data

"""

import numpy as np
from scipy.ndimage import gaussian_filter


class NoiseApplicationError(Exception):
    """General exception for noise application errors."""


class RescalingError(NoiseApplicationError):
    """Error with rescaling the noise strength"""


class ReweightingError(NoiseApplicationError):
    """Error with reweighting synthetic noisy tile to better match reality"""


def noise_strength_map(current_grid: np.ndarray, *, filter_size: int) -> np.ndarray:
    """
    Get the expected noise strength map from a gridded field of current
    velocities.

    Sets land values to 0.0, since we don't really care what happens here (there's no signal),
    but we likely want to avoid NaNs in the output (this might break future calculations).

    :param current_grid: gridded field showing current strength; land should be indicated
                         with np.nan.
    :param filter_size: size of Gaussian filter to apply, in grid points
    :returns: a grid of the same shape as current_grid, showing the expected noise strength
              at that grid point. Land values are set to 0.0.
    """
    land_mask = np.isnan(current_grid)

    current_grid = np.where(land_mask, 0.0, current_grid)

    filtered = gaussian_filter(current_grid, sigma=filter_size)
    filtered[land_mask] = 0.0

    return filtered


def _weighted_add(
    clean_target: np.ndarray,
    noise_quilt: np.ndarray,
    strength_map: np.ndarray,
    *,
    weighting_factor: float,
) -> np.ndarray:
    """
    Add noise to a target.

    :param clean_target: the clean target currents
    :param noise_quilt: the noise quilt to add
    :param strength_map: the noise strength map to use for weighting.
                         Might come from noise_strength_map().
    :param weighting_factor: a weighting factor applied to the noise before addition.
                             Probably randomly generated.

    :returns: the noisy target data
    """
    if np.isnan(strength_map).any():
        raise RescalingError("strength_map contains NaNs - these should be set to 0...")

    return clean_target + (noise_quilt * strength_map * weighting_factor)


def reweight_noisy_tile(
    noisy_tile: np.ndarray, clean_tile_maximum: float
) -> np.ndarray:
    """
    Reweight a noisy tile to better match the real noisy data.

    We expect the addition of a noise tile + signal to have inflated the values in the array,
    so we'll scale it by a value derived from the maximum values in the noisy tile and the
    corresponding clean tile.

    :param noisy_tile: a simulated noise tile - clean signal with noise applied. e.g. derived
                       from add_noise()
    :param clean_tile: the maximum current value in the corresponding clean tile

    :returns: the reweighted noisy tile
    :raises ReweightingError: if clean_tile_maximum is NaN or non-positive
    """
    if np.isnan(clean_tile_maximum):
        raise ReweightingError(
            f"clean_tile_maximum is NaN - cannot reweight noisy tile.\n"
            "\tMaybe you meant to use np.nanmax()?"
        )
    if clean_tile_maximum <= 0:
        raise ReweightingError(
            f"clean_tile_maximum isn't positive ({clean_tile_maximum:.3f})."
            "This shouldn't be possible."
        )

    # We want to scale such that the new maximum is halfway between the current maximum
    # and the clean maximum
    noisy_max = np.nanmax(noisy_tile)
    scale_factor = (clean_tile_maximum + noisy_max) / (2 * noisy_max)

    return noisy_tile * scale_factor


def add_noise(
    clean_target: np.ndarray,
    noise_quilt: np.ndarray,
    strength_map: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add noise to a target, choosing a random weighting factor for the noise strength
    and scaling the result to better match real noisy data.

    :param clean_target: the clean target currents
    :param noise_quilt: the noise quilt to add
    :param rng: random number generator

    :returns: the noisy target data, with random-strength noise added and weighted to match
              real noisy data.

    :raises NoiseError: if the shapes of clean_target and noise_quilt do not match
    """
    if clean_target.shape != noise_quilt.shape:
        raise NoiseApplicationError(
            "Shapes of clean target and noise quilt do not match: "
            f"{clean_target.shape} vs {noise_quilt.shape}"
        )
    if clean_target.shape != strength_map.shape:
        raise NoiseApplicationError(
            "Shapes of clean target and strength map do not match: "
            f"{clean_target.shape} vs {strength_map.shape}"
        )

    weight = rng.uniform(0.5, 2.5)
    noisy = _weighted_add(
        clean_target, noise_quilt, strength_map, weighting_factor=weight
    )

    return reweight_noisy_tile(noisy, np.nanmax(clean_target))
