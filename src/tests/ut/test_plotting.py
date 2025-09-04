"""
Tests for plotting utilities
"""

import numpy as np

from current_denoising.plotting import maps


def test_grid_latlong():
    """
    Check we get the right co-ordinates for a quarter degree grid
    """
    lat, long = maps.lat_long_grid((720, 1440))

    a = 90 - 1 / 8
    np.testing.assert_array_equal(lat, np.linspace(-a, a, 720))

    b = 180 - 1 / 8
    np.testing.assert_array_equal(long, np.linspace(-b, b, 1440))


def test_twelfth_degree_latlong():
    """
    Check we get the right co-ords for a twelfth degree grid
    """
    lat, long = maps.lat_long_grid((2160, 4320))

    a = 90 - 1 / 24
    np.testing.assert_array_equal(lat, np.linspace(-a, a, 2160))

    b = 180 - 1 / 24
    np.testing.assert_array_equal(long, np.linspace(-b, b, 4320))
