"""
Training data and augmentations for the denoising model
"""

import numpy as np

from ..generation.ioutils import _tile_index
from ..generation.applying_noise import add_noise
from ..utils import util


class DataError(Exception):
    """
    General exception for data processing
    """


class BadNoiseTileError(DataError):
    """
    Noise tiles are bad shapes
    """


def get_training_pairs(
    clean_source: np.ndarray,
    noise_strength_map: np.ndarray,
    noise_tiles: np.ndarray | list[np.ndarray],
    max_latitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Get matched training pairs from a clean source gridded field, a map showing
    the global noise strength and an iterable of noise tiles.

    Rescales the noisy tile to better match the scale of the original tile than a naive
    addition - see `applying_noise.reweight_noisy_tile` for details.
    The `clean_source` array will probably be a gridded field of MDT residuals
    (i.e. the difference between the MDT and the MDT after smoothing).

    Returns one training pair per provided noise tile, as an Nx2xDxD array, if provided
    N DxD noise tiles.

    :param clean_source: global gridded field to add noise to. Patches of the right
                         size will be randomly drawn from this, restricted to being within
                         +- max latitude
    :param noise_strength_map: an array of the same shape as `clean_source` which modulates
                               the noise values as they vary in space. Useful if we expect to
                               have higher noise in some areas than others.
    :param noise_tiles: an iterable (either a list or numpy array) containing the noise tiles
                        to apply to data. This tells us how many tiles to extract and also what
                        size they should be. If a list, must contain square tiles of the same size;
                        if a numpy array, must have shape NxDxD.
    :param max_latitude: maximum latitude to extract tiles from.
    :param rng: random number generator, used for selecting stochastic noise strength and
                tile location.

    :returns: an Nx2xDxD array of training pairs, [clean, noisy]
    :raises DataError: if the strength map + source data are different shapes
    :raises BadNoiseTileError: if the noise tiles are a non-square or inconsistently sized list
    :raises BadNoiseTileError: if the noise tiles is an array of non-square indices
    """
    if clean_source.shape != noise_strength_map.shape:
        raise DataError(f"Got {clean_source.shape=} but {noise_strength_map.shape=}")

    # List preconditions
    if isinstance(noise_tiles, list):
        if len({t.shape for t in noise_tiles}) != 1:
            raise BadNoiseTileError(
                f"List of tiles contains different shapes: {set(t.shape for t in noise_tiles)}"
            )
        if any((t.ndim != 2 for t in noise_tiles)):
            raise BadNoiseTileError(
                f"Got noise tile dimensions {set(t.ndim for t in noise_tiles)}"
            )
        if any((t.shape[0] != t.shape[1] for t in noise_tiles)):
            raise BadNoiseTileError(
                f"Got non-square tiles: {set(t.shape for t in noise_tiles)}"
            )

        d = noise_tiles[0].shape[0]

    # Array preconditions. Why didn't I just enforce that it's a list
    elif isinstance(noise_tiles, np.ndarray):
        if noise_tiles.ndim != 3 or noise_tiles.shape[1] != noise_tiles.shape[2]:
            raise BadNoiseTileError(f"Got {noise_tiles.shape=}")

        _, d, _ = noise_tiles.shape

    else:
        raise ValueError(f"got {type(noise_tiles)=}; must be list or array")

    clean_tiles = []
    noisy_tiles = []

    for noise_tile in noise_tiles:
        # Choose a location with the RNG, subject to our latitude condition
        location = _tile_index(
            rng, input_size=clean_source.shape, max_latitude=max_latitude, tile_size=d
        )

        # Get the pair of tiles at this location
        clean = util.tile(clean_source, location, d)
        noisy = add_noise(
            clean, noise_tile, util.tile(noise_strength_map, location, d), rng
        )

        clean_tiles.append(clean)
        noisy_tiles.append(noisy)

    return np.stack([np.array(clean_tiles), np.array(noisy_tiles)], axis=1)
