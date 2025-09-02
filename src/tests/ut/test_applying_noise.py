"""
Tests for the noise application utils
"""

import numpy as np

from current_denoising.generation import applying_noise


def test_noise_strength_land_mask():
    """
    Check that the land is in the right place in our noise strength map
    """
    map = np.arange(100, dtype=np.float32).reshape(10, 10)
    # Set some values to indicate that they're land
    map[(map % 3).astype(bool)] = np.nan

    assert (
        np.isnan(map) == np.isnan(applying_noise.noise_strength_map(map, filter_size=5))
    ).all()

    assert (
        np.isnan(map)
        == np.isnan(applying_noise.noise_strength_map(map, filter_size=50))
    ).all()
