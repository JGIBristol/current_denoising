"""
Test for utilities
"""

import pytest
import numpy as np

from current_denoising.utils import util


def test_grid_latlong():
    """
    Check we get the right co-ordinates for a quarter degree grid
    """
    lat, long = util.lat_long_grid((720, 1440))

    a = 90 - 1 / 8
    np.testing.assert_array_equal(lat, np.linspace(-a, a, 720))

    b = 180 - 1 / 8
    np.testing.assert_array_equal(long, np.linspace(-b, b, 1440))


def test_twelfth_degree_latlong():
    """
    Check we get the right co-ords for a twelfth degree grid
    """
    lat, long = util.lat_long_grid((2160, 4320))

    a = 90 - 1 / 24
    np.testing.assert_array_equal(lat, np.linspace(-a, a, 2160))

    b = 180 - 1 / 24
    np.testing.assert_array_equal(long, np.linspace(-b, b, 4320))


@pytest.fixture
def sample_grid() -> np.ndarray:
    """
    A sample grid for testing
    """
    grid = np.arange(648, dtype=float).reshape((18, 36))
    np.fill_diagonal(grid, np.nan)
    return grid


def test_get_tile(sample_grid: np.ndarray):
    """
    Check we can get the right tile from a grid.
    """
    tile = util.get_tile(sample_grid, (75, -165), 30)

    expected = np.array(
        [
            [np.nan, 38, 39],
            [73, np.nan, 75],
            [109, 110, np.nan],
        ]
    )
    np.testing.assert_array_equal(tile, expected)


def test_get_tile_equal(sample_grid: np.ndarray):
    """
    Check we get the right grid point if we're equidistant between two.
    """
    tile = util.get_tile(sample_grid, (70, -160), 30)

    expected = np.array(
        [
            [np.nan, 38, 39],
            [73, np.nan, 75],
            [109, 110, np.nan],
        ]
    )
    np.testing.assert_array_equal(tile, expected)


def test_get_tile_edge(sample_grid: np.ndarray):
    """
    Check we can get a tile that starts at the edge of the grid.
    """
    tile = util.get_tile(sample_grid, (90, -180), 30)
    expected = np.array(
        [
            [np.nan, 1, 2],
            [36, np.nan, 38],
            [72, 73, np.nan],
        ]
    )
    np.testing.assert_array_equal(tile, expected)

    tile = util.get_tile(sample_grid, (75, -180), 30)
    expected = np.array(
        [
            [36, np.nan, 38],
            [72, 73, np.nan],
            [108, 109, 110],
        ]
    )
    np.testing.assert_array_equal(tile, expected)

    tile = util.get_tile(sample_grid, (90, -165), 30)
    expected = np.array([[1, 2, 3], [np.nan, 38, 39], [73, np.nan, 75]])
    np.testing.assert_array_equal(tile, expected)


def test_not_whole_grid_points(sample_grid: np.ndarray):
    """
    Check the right error gets raised if the requested tile size isn't a whole number of grid points.
    """
    util.get_tile(sample_grid, (90, -180), 10)
    with pytest.raises(util.LatLongError):
        util.get_tile(sample_grid, (90, -180), 15)


def test_get_tile_out_of_bounds(sample_grid: np.ndarray):
    """
    Check we get an error if we try to get a tile that doesn't fit in the grid.
    """
    size = 30
    with pytest.raises(util.LatLongError):
        util.get_tile(sample_grid, (100, 0), size)
    with pytest.raises(util.LatLongError):
        util.get_tile(sample_grid, (-100, 0), size)
    with pytest.raises(util.LatLongError):
        util.get_tile(sample_grid, (200, 0), size)
    with pytest.raises(util.LatLongError):
        util.get_tile(sample_grid, (-200, 0), size)

    util.get_tile(sample_grid, (0, 90), size)
    util.get_tile(sample_grid, (0, -90), size)
    with pytest.raises(util.LatLongError):
        util.get_tile(sample_grid, (0, 190), size)
    with pytest.raises(util.LatLongError):
        util.get_tile(sample_grid, (0, -190), size)


@pytest.fixture
def image() -> np.ndarray:
    """An example image"""
    return np.arange(32).reshape((4, 8))


def test_tile(image):
    """
    Test extracting a tile from an array
    """
    expected_tile = np.array([[10, 11], [18, 19]])

    assert np.array_equal(util.tile(image, (1, 2), 2), expected_tile)


def test_cos_latitude():
    """
    Check we get the right cosine latitudes from a grid's size
    """
    # Try it for a quarter degree grid
    start_lat = np.deg2rad(-90 + 1 / 8)
    n_points = 720

    latitudes = np.linspace(start_lat, -start_lat, n_points, endpoint=True)

    np.testing.assert_array_almost_equal(
        np.cos(latitudes), util.cos_latitudes(n_points)
    )


def test_sliding_window():
    """
    Mainline test case for sliding window function
    """
    arr = np.arange(25).reshape((5, 5))
    expected = np.array(
        [
            [6, 7, 8, 9, np.nan],
            [11, 12, 13, 14, np.nan],
            [16, 17, 18, 19, np.nan],
            [21, 22, 23, 24, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    np.testing.assert_array_equal(
        expected, util.apply_to_sliding_window(arr, np.max, 2)
    )


def test_window_too_large():
    """
    Window is larger than the grid
    """
    arr = np.arange(15).reshape((3, 5))
    with pytest.raises(util.UtilError):
        util.apply_to_sliding_window(arr, np.max, 6)
    with pytest.raises(util.UtilError):
        util.apply_to_sliding_window(arr, np.max, 4)


def test_arg_in_signature():
    """
    Check the function argument helper works
    """

    def tmp(x):
        return x

    assert util._arg_in_signature(tmp, "x")
    assert not util._arg_in_signature(tmp, "y")


def test_no_axis_arg_in_fcn():
    """
    Check we get the right error if the function we provided doesn't take an
    axis argument
    """
    arr = np.arange(15).reshape((3, 5))

    util.apply_to_sliding_window(arr, np.max, 2)

    with pytest.raises(util.UtilError):
        util.apply_to_sliding_window(arr, lambda x: np.max(x), 2)
