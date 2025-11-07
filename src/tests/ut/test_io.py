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


@pytest.fixture
def sinusoid_image() -> np.ndarray:
    """
    An image composed of controlled sinusoids,
    so we know its frequency spectrum.
    """
    size = 32
    freq_powers = {2: 0.6, 4: 0.3, 8: 0.1}
    y, x = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size))

    for freq, power in freq_powers.items():
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)

        component = (
            np.sin(2 * np.pi * freq * x / size + phase_x)
            + np.sin(2 * np.pi * freq * y / size + phase_y)
        ) * np.sqrt(power)
        image += component

    return image


def test_extract_tiles_invalid_image_dim():
    """
    Check we get the right error if we pass a 1D or 3D image
    """
    num_tiles = 1
    tile_size = 8
    with pytest.raises(ioutils.IOError):
        ioutils.extract_tiles(None, np.empty((32, 32, 3)), num_tiles=num_tiles)

    with pytest.raises(ioutils.IOError, match="Input image must be 2d"):
        ioutils.extract_tiles(None, np.empty(32), num_tiles=num_tiles)


def test_extract_tiles_large_tiles():
    """
    Check we get the right error if we pass a large tile size
    """
    with pytest.raises(ioutils.IOError):
        ioutils.extract_tiles(None, np.empty((128, 32)), num_tiles=1, tile_size=64)

    with pytest.raises(ioutils.IOError):
        ioutils.extract_tiles(None, np.empty((32, 128)), num_tiles=1, tile_size=64)


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

    def __init__(self):
        # These are the indices we'll return
        self.indices = [(0, 0), (1, 1), (2, 3), (2, 5), (0, 0)]
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
        # We shouldn't get here in any of our tests...
        raise


def test_extract_tile_simple():
    """
    Simple check to see if we can extract a 2x2 tile from a 2x2 image
    (i.e. do we just get the whole image back)
    """

    # The mock RNG returns 0, 0 first
    img = np.arange(4).reshape((2, 2))
    np.testing.assert_array_equal(
        ioutils.extract_tiles(
            MockRNG(),
            img,
            num_tiles=1,
            tile_criterion=None,
            max_latitude=np.inf,
            tile_size=2,
            allow_nan=False,
            return_indices=False,
        ),
        [img],
    )


def test_extract_tile_simple_criterion():
    """
    Check that we can extract a tile if we provide a criterion function
    """
    # The mock RNG returns (0, 0) first, which we will reject,
    # and then returns (1, 1) which we will accept
    img = np.arange(9).reshape((3, 3))
    expected_tile = np.reshape([[4, 5], [7, 8]], (2, 2))

    np.testing.assert_array_equal(
        ioutils.extract_tiles(
            MockRNG(),
            img,
            num_tiles=1,
            tile_criterion=lambda tile: np.max(tile) == 8,
            max_latitude=np.inf,
            tile_size=2,
            allow_nan=False,
            return_indices=False,
        ),
        [expected_tile],
    )


def test_extract_tiles_no_criterion(image):
    """
    Mocking the RNG to return a predefined sequence, see
    if we get the right tiles out
    """
    num_tiles = 4
    tile_size = 2

    # Mock the RNG by using this class which just returns some predefined
    # co-ordinates in order
    rng = MockRNG()

    tiles = ioutils.extract_tiles(
        rng,
        image,
        num_tiles=num_tiles,
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


def test_extract_tiles_rms_criterion(image):
    """
    Extract tiles but only those with a certain RMS,
    which should make us skip one tile
    compared to test_extract_tiles_no_criterion
    """
    num_tiles = 4
    tile_size = 2

    # Mock the RNG by using this class which just returns some predefined
    # co-ordinates in order
    rng = MockRNG()

    tiles = ioutils.extract_tiles(
        rng,
        image,
        num_tiles=num_tiles,
        tile_criterion=lambda tile: ioutils.tile_rms(tile) < 24,
        tile_size=tile_size,
        max_latitude=np.inf,
    )

    assert len(tiles) == num_tiles

    # Now we expect to reject the last tile, and instead the MockRNG
    # will return (0, 0) as the fifth "random" number which gives us
    # the first tile back again
    expected = [
        image[0:2, 0:2],
        image[1:3, 1:3],
        image[2:4, 3:5],
        image[0:2, 0:2],
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

    assert ioutils.tile_rms(tile) == pytest.approx(expected_rms)


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


def test_fft_fraction_wrong_shape():
    """
    Check we get the right errors if the wrong shaped stuff is passed in to the FFT calculation
    """
    # This one should work
    ioutils.fft_fraction(np.ones((3, 3)), 0.5)

    # These ones shouldn't
    with pytest.raises(ioutils.PowerSpectrumError):
        ioutils.fft_fraction(np.ones((3, 3, 3)), 0.5)
    with pytest.raises(ioutils.PowerSpectrumError):
        ioutils.fft_fraction(np.ones((2, 3)), 0.5)


def test_fft_fraction_wrong_threshold():
    """
    Check we get the right error if a power threshold outside [0, 1] is requested
    """
    # This one should work
    ioutils.fft_fraction(np.ones((3, 3)), 0.5)
    ioutils.fft_fraction(np.ones((3, 3)), 0.0)
    ioutils.fft_fraction(np.ones((3, 3)), -0.0)
    ioutils.fft_fraction(np.ones((3, 3)), 1.0)

    # These ones shouldn't
    with pytest.raises(ioutils.PowerSpectrumError):
        ioutils.fft_fraction(np.ones((3, 3)), -0.1)
    with pytest.raises(ioutils.PowerSpectrumError):
        ioutils.fft_fraction(np.ones((3, 3)), 2)


def test_fft_fraction(sinusoid_image):
    """
    Check we get the right fraction of Fourier power from a test image
    """
    # r_max is the half-diagonal
    r_max = np.sqrt(sinusoid_image.shape[0] ** 2 + sinusoid_image.shape[1] ** 2)
    ioutils.fft_fraction(sinusoid_image, 0.5)

    # Based on the image we defined above, we know there will only be contributions
    # at r=2, r=4 and r=8
    # And that their relative strengths will be in the ratio
    # 6:3:1
    # Therefore 4/10 of the power will be above the r=2 bits
    np.testing.assert_almost_equal(ioutils.fft_fraction(sinusoid_image, 3 / r_max), 0.4)


def test_land_distance_nan():
    """
    Check we get the right distance from land if NaN is the sentinel
    """
    r8 = np.sqrt(8)
    r5 = np.sqrt(5)
    r2 = np.sqrt(2)
    distances = np.array(
        [
            [r8, r5, 2, r5, r8],
            [r5, r2, 1, r2, r5],
            [2, 1, 0, 1, 2],
            [r5, r2, 1, r2, r5],
            [r8, r5, 2, r5, r8],
        ]
    )

    arr = np.zeros_like(distances)
    arr[2, 2] = np.nan
    np.testing.assert_almost_equal(ioutils.distance_from_land(arr), distances)


def test_land_distance_zero_sentinel():
    """
    Check we get the right distance from land if 0 is the sentinel
    """
    r8 = np.sqrt(8)
    r5 = np.sqrt(5)
    r2 = np.sqrt(2)
    distances = np.array(
        [
            [r8, r5, 2, r5, r8],
            [r5, r2, 1, r2, r5],
            [2, 1, 0, 1, 2],
            [r5, r2, 1, r2, r5],
            [r8, r5, 2, r5, r8],
        ]
    )

    arr = np.ones_like(distances)
    arr[2, 2] = 0

    np.testing.assert_almost_equal(
        ioutils.distance_from_land(arr, land_sentinel=0), distances
    )
