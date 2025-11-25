"""
Tests for denoiser data preprocessing
"""

import pytest
import numpy as np

from current_denoising.denoising import data
from current_denoising.generation import ioutils


def test_training_pairs_wrong_sized_noise_map():
    """
    Check we get the right error if the noise map + source field have different shapes
    """
    with pytest.raises(data.DataError):
        data.get_training_pairs(np.empty((4, 8)), np.empty((5, 8)), None, None, None)


def test_training_pair_bad_noise_tiles():
    """
    Check we get the right error if the noise tiles are different shapes, a numpy array
    of the wrong dimension, or a list of 3d arrays
    """
    tile2 = np.empty((2, 2))
    tile3 = np.empty((3, 3))

    # List with inconsistent shapes
    with pytest.raises(data.BadNoiseTileError):
        data.get_training_pairs(
            np.empty((4, 8)), np.empty((4, 8)), [tile2, tile3], None, None
        )

    # List with non-2d contents
    with pytest.raises(data.BadNoiseTileError):
        data.get_training_pairs(
            np.empty((4, 8)), np.empty((4, 8)), [np.empty((2, 3, 3))], None, None
        )

    # List with non-square contents
    with pytest.raises(data.BadNoiseTileError):
        data.get_training_pairs(
            np.empty((4, 8)), np.empty((4, 8)), [np.empty((3, 2))], None, None
        )

    # Array containing non-square tiles
    with pytest.raises(data.BadNoiseTileError):
        data.get_training_pairs(
            np.empty((4, 8)), np.empty((4, 8)), np.empty((3, 3, 2)), None, None
        )


def test_training_pair_negative_maxlat():
    """
    Check we get the right error if maximum latitude is not a positive number
    """
    with pytest.raises(ioutils.IOError):
        data.get_training_pairs(
            np.empty((4, 8)), np.empty((4, 8)), np.empty((3, 3, 3)), -1, None
        )


class MockRNG:
    """
    Fake RNG for testing with deterministic output

    Needs methods for returning integers (for indexing)
    and floats (for noise strength)
    """

    def __init__(self):
        self.indices = [(0, 0), (1, 1)]
        self.return_y = True
        self.index = 0

    def integers(self, *args) -> int:
        """
        Alternate between returning a y-index and an x-index
        since this is what our code expects...
        """
        if self.return_y:
            self.return_y = False
            return self.indices[self.index][0]
        else:
            self.return_y = True
            self.index += 1
            return self.indices[self.index - 1][1]
        # We shouldn't get here in any of our tests...
        raise

    def uniform(self, *args):
        return 2.0


def test_training_pairs():
    """
    Using a deterministic mock RNG, check that we get the expected tiles
    """
    # Use fake RNG for testing with predetermined output
    rng = MockRNG()

    # Our MDT map
    map = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )

    # Make a noise strength strength quilt with some different elements
    # on the diagonal
    noise_strength_map = np.ones_like(map)
    noise_strength_map[0, 0] = 2
    noise_strength_map[1, 1] = 0

    noise_tiles = [
        np.array(
            [
                [1, 0],
                [0, 1],
            ]
        ),
        np.array(
            [
                [2, 1],
                [0, 0],
            ]
        ),
    ]

    # Expect the RNG to return (0, 0) and (1, 1) which should both be valid indices
    # This gives us [[1,2],[1,2]] and [[2,3],[2,3]] as our clean inputs
    # Our strength maps are [[2,1],[1,0]] and [[0, 1],[1,1]]
    # The noise tiles are defined above, and the "stochastic" strength parameter is
    # always 2.0 with our mock RNG
    # Before scaling, this gives us expected noise tiles of [[5,2],[1,2]] and [[2,5],[2,3]]
    # However, we scale the output to have a maximum halfway between the unscaled noisy tile
    # and the clean tile, which gives us the below
    expected_output = np.array(
        [
            [
                [
                    [1, 2],
                    [1, 2],
                ],
                [
                    [3.5, 1.4],
                    [0.7, 1.4],
                ],
            ],
            [
                [
                    [2, 3],
                    [2, 3],
                ],
                [
                    [1.6, 4],
                    [1.6, 2.4],
                ],
            ],
        ]
    )

    result = data.get_training_pairs(map, noise_strength_map, noise_tiles, np.inf, rng)
    np.testing.assert_allclose(result, expected_output)


def test_dataloader():
    """
    Check we can get some tiles out
    """
    tiles = np.empty((4, 3, 3))

    config = data.DataConfig(train=True, batch_size=2, num_workers=2)
    loader = data.dataloader(tiles, tiles, config)

    clean, noisy = next(iter(loader))
    assert clean.shape == (2, 1, 3, 3)

    # Larger batch size than dataset, should get the whole thing
    # since we're not training and don't drop incomplete batches
    config = data.DataConfig(train=False, batch_size=5, num_workers=0)
    loader = data.dataloader(tiles, tiles, config)

    clean, noisy = next(iter(loader))
    assert clean.shape == (4, 1, 3, 3)


def test_dataloader_invalid_nans():
    """
    Check that if we have different NaNs in clean/noisy data, we get an error
    """
    clean = np.empty((4, 3, 3))
    noisy = clean.copy()

    clean[1, 0] = np.nan
    noisy[0, 1] = np.nan

    with pytest.raises(data.NaNError):
        data.dataloader(
            clean, noisy, data.DataConfig(train=True, batch_size=2, num_workers=2)
        )


def test_dataloader_nans():
    """
    Check that NaNs are replaced with 0
    """
    tiles = np.empty((4, 3, 3))
    tiles[:, 1, 0] = np.nan

    config = data.DataConfig(train=False, batch_size=2, num_workers=2)
    loader = data.dataloader(tiles, tiles, config)

    clean, noisy = next(iter(loader))
    assert clean.shape == (2, 1, 3, 3)
    assert (clean[:, 0, 1, 0] == 0.0).all()

    assert noisy.shape == (2, 1, 3, 3)
    assert (noisy[:, 0, 1, 0] == 0.0).all()


def test_data_augmentations():
    """
    Check that the training data is augmented (and test data isn't)
    """
    n_repeats = 100
    tiles = np.tile(np.arange(4).reshape((2, 2)), (n_repeats, 1, 1))
    train_loader = data.dataloader(
        tiles, tiles, data.DataConfig(train=True, batch_size=5, num_workers=0)
    )

    expected_train_tiles = np.array(
        [
            [
                [0, 1],  # Original
                [2, 3],
            ],
            [
                [2, 0],  # Rotated
                [3, 1],
            ],
            [
                [3, 2],  # Rotated twice
                [1, 0],
            ],
            [
                [1, 0],  # Flipped
                [3, 2],
            ],
        ]
    )

    train_tiles_found = [False, False, False, False]
    for actual, _ in train_loader:
        batch = actual.squeeze(1).numpy()

        for i, expected in enumerate(expected_train_tiles):
            matches = (batch == expected).all(axis=(1, 2))
            if matches.any():
                train_tiles_found[i] = True

    assert all(train_tiles_found)

    expected = np.array(
        [
            [0, 1],
            [2, 3],
        ]
    )
    val_loader = data.dataloader(
        tiles, tiles, data.DataConfig(train=False, batch_size=5, num_workers=0)
    )
    for actual, _ in val_loader:
        assert (actual.squeeze(1).numpy() == expected).all()
