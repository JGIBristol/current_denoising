"""
Training data and augmentations for the denoising model
"""

from dataclasses import dataclass

import numpy as np
import torch
import albumentations

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
