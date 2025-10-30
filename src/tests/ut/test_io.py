"""
Tests for the generator IO operations
"""

import pytest

import numpy as np

from current_denoising.generation import ioutils


@pytest.fixture
def rng() -> np.random.Generator:
    """A random number generator"""
    return np.random.default_rng(seed=0)


@pytest.fixture
def image() -> np.ndarray:
    """An example image"""
    return np.arange(32).reshape((4, 8))


@pytest.fixture
def tall_image() -> np.ndarray:
    """
    A tall example image - 19 rows, 5 columns

    Useful for testing things like the latitude exclusion functions
    """
    return np.arange(95).reshape((19, 5))


def test_extract_tiles_invalid_image_dim():
    """
    Check we get the right error if we pass a 1D or 3D image
    """
    num_tiles = 1
    tile_size = 8
    with pytest.raises(ioutils.IOError):
        ioutils.extract_tiles(
            None, np.empty((32, 32, 3)), num_tiles=num_tiles, max_rms=np.inf
        )

    with pytest.raises(ioutils.IOError, match="Input image must be 2d"):
        ioutils.extract_tiles(None, np.empty(32), num_tiles=num_tiles, max_rms=np.inf)


def test_extract_tiles_large_tiles():
    """
    Check we get the right error if we pass a large tile size
    """
    with pytest.raises(ioutils.IOError):
        ioutils.extract_tiles(
            None, np.empty((128, 32)), num_tiles=1, max_rms=np.inf, tile_size=64
        )

    with pytest.raises(ioutils.IOError):
        ioutils.extract_tiles(
            None, np.empty((32, 128)), num_tiles=1, max_rms=np.inf, tile_size=64
        )


def test_extract_indices(rng):
    """
    Check we get all of, and only, the expected indices
    """
    expected_indices = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}

    # Generate enough indices that we expect to always get all of them
    actual_indices = {
        ioutils._tile_index(rng, input_size=(3, 4), max_latitude=np.inf, tile_size=2)
        for _ in range(20)
    }

    assert actual_indices == expected_indices


class MockRNG:
    """
    Used to mock an RNG - returns a specified sequence of integers
    """

    def __init__(self, h: int, w: int, tile: int):
        # These are the indices we'll return
        self.indices = [(0, 0), (1, 1), (2, 3), (2, 5)]
        self.return_y = True
        self.index = 0

    def integers(self, *args) -> int:
        if self.return_y:
            self.return_y = False
            return self.indices[self.index][0]
        else:
            self.return_y = True
            self.index += 1
            return self.indices[self.index - 1][1]
        raise


def test_extract_tiles(image):
    """
    Mocking the RNG to return a predefined sequence, see
    if we get the right tiles out
    """
    num_tiles = 4
    tile_size = 2

    # Mock the RNG by using this class which just returns some predefined
    # co-ordinates in order
    rng = MockRNG(*image.shape, tile_size)

    tiles = ioutils.extract_tiles(
        rng,
        image,
        num_tiles=num_tiles,
        max_rms=np.inf,
        tile_size=tile_size,
        max_latitude=np.inf,
    )

    assert len(tiles) == num_tiles

    expected = [
        image[0:2, 0:2],
        image[1:3, 1:3],
        image[2:4, 3:5],
        image[2:4, 5:7],
    ]

    np.testing.assert_array_equal(tiles[0], expected[0])
    np.testing.assert_array_equal(tiles[1], expected[1])
    np.testing.assert_array_equal(tiles[2], expected[2])
    np.testing.assert_array_equal(tiles[3], expected[3])


def test_tile_rms():
    """
    Check we calculate RMS properly
    """
    tile = np.array([[10, 11], [18, 19]])
    expected_rms = np.sqrt((10**2 + 11**2 + 18**2 + 19**2) / 4)

    assert ioutils._tile_rms(tile) == pytest.approx(expected_rms)


def test_bad_max_latitude():
    """
    Check we get the right error if we pass a nonsensical value (i.e. <0) as the max latitude
    """
    ioutils._included_indices(32, 2, 45)
    with pytest.raises(ioutils.IOError):
        ioutils._included_indices(32, 8, 0)
    with pytest.raises(ioutils.IOError):
        ioutils._included_indices(32, 8, -1)


def test_exclude_latitude_indices(tall_image):
    """
    Check that when we exclude some latitudes, we get the right range of possible indices
    for the random choice.
    """
    assert ioutils._included_indices(tall_image.shape[0], 2, 64.0) == (3, 14)

    # Check we get the right thing when the maximum latitude lies right on a row
    assert ioutils._included_indices(tall_image.shape[0], 2, 50.0) == (4, 13)

    # Check we get it right when the max latitude is exactly at the start
    assert ioutils._included_indices(tall_image.shape[0], 2, 90.0) == (0, 17)

    # Check we get it right when the max latitude is exactly at the end
    # Turns out this is the same as the above
    assert ioutils._included_indices(tall_image.shape[0], 2, 90.0) == (0, 17)


def test_exclude_latitude_include_all(tall_image):
    """
    Check we get the entire range if our latitude is >90, or if the size of the tile pushes it >90
    """
    assert ioutils._included_indices(tall_image.shape[0], 2, 90.0) == (0, 17)
    assert ioutils._included_indices(tall_image.shape[0], 3, 100) == (0, 17)


def test_exclude_latitude_too_small(tall_image):
    """
    Check the right error is raised if we exclude too many latitudes, such that the provided tile size
    can't be extracted
    """
    with pytest.raises(ioutils.IOError):
        ioutils._included_indices(tall_image.shape[0], 4, 15.0)


def test_clipped_coriolis():
    """
    Check that we correctly clip the coriolis parameter
    """
    latitudes = np.arange(-90, 100, 10)

    raw_coriolis = ioutils._coriolis_parameter(latitudes)
    expected_coriolis = raw_coriolis.copy()
    expected_coriolis[8:9] = raw_coriolis[7]
    expected_coriolis[9:11] = -raw_coriolis[7]

    assert np.any(raw_coriolis != expected_coriolis)

    assert np.array_equal(
        ioutils.clipped_coriolis_param(latitudes, clip_at=10), expected_coriolis
    )
    assert np.array_equal(
        ioutils.clipped_coriolis_param(latitudes, clip_at=15), expected_coriolis
    )


def test_remove_nanmean():
    """
    Check we get the right centred array if our array contains NaNs
    """
    np.testing.assert_array_almost_equal(
        ioutils._remove_nanmean(
            np.array(
                [
                    [0, 1],
                    [2, np.nan],
                ]
            )
        ),
        np.array(
            [
                [-1, 0],
                [1, np.nan],
            ]
        ),
    )


def test_remove_mean():
    """
    Check we get the right centred array for an array with no NaNs
    """
    np.testing.assert_array_almost_equal(
        ioutils._remove_nanmean(
            np.array(
                [
                    [0, 1],
                    [2, 1],
                ]
            )
        ),
        np.array(
            [
                [-1, 0],
                [1, 0],
            ],
        ),
    )
