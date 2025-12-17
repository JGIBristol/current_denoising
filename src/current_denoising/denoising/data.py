"""
Training data and augmentations for the denoising model
"""

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import albumentations

from ..generation.ioutils import extract_tiles
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


class DatasetError(DataError):
    """
    Error initialising the dataset/dataloader
    """


class NaNError(DataError):
    """
    Error with NaNs in the data
    """


class DenoisingDataset(torch.utils.data.Dataset):
    """
    Dataset for the denoising - pairs of clean/noisy tiles
    """

    def __init__(self, clean: np.ndarray, noisy: np.ndarray, augment: bool):
        """
        Initialise the data
        """
        if clean.shape != noisy.shape:
            raise DatasetError(f"{clean.shape=} but {noisy.shape=}")

        # Replace NaNs in the data with 0
        clean_nans = np.isnan(clean)
        noisy_nans = np.isnan(noisy)
        if (clean_nans != noisy_nans).any():
            raise NaNError(f"Got different NaNs")
        clean[clean_nans] = 0
        noisy[noisy_nans] = 0

        # Add channel dimensions
        self.clean = clean
        self.noisy = noisy

        # No-op if no transformations (e.g. for testing/validation)
        self.augmentations = (
            albumentations.Compose(
                [
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.VerticalFlip(p=0.5),
                    # This still gives us 25% chance for no rotation
                    albumentations.RandomRotate90(p=1),
                ],
                additional_targets={"noisy": "image"},
            )
            if augment
            else None
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx: int):
        """
        Returns clean/noisy
        """
        clean = self.clean[idx]
        noisy = self.noisy[idx]

        if self.augmentations is not None:
            augmented = self.augmentations(image=clean, noisy=noisy)
            clean = augmented["image"]
            noisy = augmented["noisy"]

        clean = torch.from_numpy(clean).float().unsqueeze(0)
        noisy = torch.from_numpy(noisy).float().unsqueeze(0)

        return clean, noisy


@dataclass
class DataConfig:
    """
    Configuration for dataloader/dataset.
    """

    train: bool
    """ Whether this dataset will be used for training"""

    batch_size: int
    """ Batch size """

    num_workers: int
    """
    Number of subprocesses to use for dataloading.

    0 loads data in the main process.
    """


def get_training_pairs(
    clean_source: np.ndarray,
    noise_strength_map: np.ndarray,
    noise_tiles: np.ndarray | list[np.ndarray],
    max_latitude: float,
    max_nan_fraction: float,
    rng: np.random.Generator,
    *,
    return_indices=False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Get matched training pairs from a clean source gridded field, a map showing
    the global noise strength and an iterable of noise tiles.

    Extracts all the tiles that meet the latitude + NaN fraction conditions from the
    `clean_source` image; if not enough noise_tiles are provided to match these, a warning
    will be issued.

    Rescales the noisy tile to better match the scale of the original tile than a naive
    addition - see `applying_noise.reweight_noisy_tile` for details.
    The `clean_source` array will probably be a gridded field of MDT residuals
    (i.e. the difference between the MDT and the MDT after smoothing).

    Returns one training pair per provided noise tile, as an Nx2xDxD array, if provided
    N DxD noise tiles.

    :param clean_source: global gridded field to add noise to. Patches of the right
                         size will be randomly drawn from this, restricted to being within
                         +- max latitude and subject to the criterion imposed by `nan_fraction`.
    :param noise_strength_map: an array of the same shape as `clean_source` which modulates
                               the noise values as they vary in space. Useful if we expect to
                               have higher noise in some areas than others.
    :param noise_tiles: an iterable (either a list or numpy array) containing the noise tiles
                        to apply to data. This tells us what size they should be.
                        If a list, must contain square tiles of the same size;
                        if a numpy array, must have shape NxDxD.
                        A warning will be issued if there are too few noise_tiles than tiles
                        extracted from `clean_source`.
    :param max_latitude: maximum latitude to extract tiles from.
    :param max_nan_fraction: the maximum amount of NaN allowed in the tile.
    :param rng: random number generator, used for selecting stochastic noise strength.
    :param return_indices: whether to also return the locations of each tile in the original grid.

    :returns: an Nx2xDxD array of training pairs, [clean, noisy]
    :raises DataError: if the strength map + source data are different shapes
    :raises BadNoiseTileError: if the noise tiles are a non-square or inconsistently sized list
    :raises BadNoiseTileError: if the noise tiles is an array of non-square indices
    """
    if clean_source.shape != noise_strength_map.shape:
        raise DataError(f"Got {clean_source.shape=} but {noise_strength_map.shape=}")

    if not 0 <= max_nan_fraction <= 1:
        raise ValueError(f"{max_nan_fraction=}")

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

    # Extract all the possible tiles from the clean data, given our latitude and nan
    # fraction constraints
    clean_tiles, clean_indices = extract_tiles(
        clean_source,
        tile_criterion=lambda tile: (np.sum(np.isnan(tile)) / tile.size)
        < max_nan_fraction,
        max_latitude=max_latitude,
        tile_size=d,
        allow_nan=True,
        return_indices=True,
    )

    # We mostly expect the user to have provided more noise than necessary
    if len(clean_tiles) < len(noise_tiles):
        noise_tiles = noise_tiles[: len(clean_tiles)]

    # But if not, we might want to throw some tiles away
    # This might mean that we miss regions of the map, especially near the bottom
    # and right of the gridded field, so warn the user of this
    elif len(clean_tiles) > len(noise_tiles):
        warnings.warn(
            f"Extracted {len(clean_tiles)} tiles from clean data, but only {len(noise_tiles)} noise tiles were provided.\n"
            "The remaining clean tiles will be discarded, which geographically biases the returned synthetic tiles."
        )

        clean_tiles = clean_tiles[: len(noise_tiles)]
        clean_indices = clean_indices[: len(noise_tiles)]

    noisy_tiles = []
    for clean_tile, index, noise_tile in zip(
        clean_tiles, clean_indices, noise_tiles, strict=True
    ):
        noisy = add_noise(
            clean_tile, noise_tile, util.tile(noise_strength_map, index, d), rng
        )
        noisy_tiles.append(noisy)

    retval = np.stack([np.array(clean_tiles), np.array(noisy_tiles)], axis=1)

    # I'm pretty sure that this shouldn't go wrong, but just in case
    # let's check that we have the right numbers of indices/tiles
    assert len(clean_indices) == len(
        clean_tiles
    ), f"Got {len(clean_indices)=} but {len(clean_tiles)=}"
    assert len(clean_indices) == len(
        noisy_tiles
    ), f"Got {len(clean_indices)=} but {len(noisy_tiles)=}"

    if return_indices:
        return retval, clean_indices
    return retval


def dataloader(clean_tiles: np.ndarray, noisy_tiles: np.ndarray, config: DataConfig):
    """
    Dataloader for the denoiser.

    :param clean_tiles, noisy_tiles: NxDxD shaped clean/noisy images
    :param config: extra information needed to trin the model

    """
    if clean_tiles.shape != noisy_tiles.shape:
        raise DatasetError(f"{clean_tiles.shape=} but {noisy_tiles.shape=}")

    # Only augment training data
    dataset = DenoisingDataset(clean_tiles, noisy_tiles, augment=config.train)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        # Only shuffle the training set
        shuffle=config.train,
        # Only drop the last incomplete batch (if one exists) during training
        # might help stablise training by making all batches the same size
        drop_last=config.train,
        # Pinning memory is faster on GPU (probably), and doesn't really
        # make a difference when we're on CPU so always do it
        pin_memory=True,
        # Only keep the worker processes around if we're using them
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
