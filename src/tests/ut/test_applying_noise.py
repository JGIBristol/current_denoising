"""
Tests for the noise application utils
"""

import pytest
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
        np.isnan(map) == ((applying_noise.noise_strength_map(map, filter_size=5) == 0))
    ).all()

    assert (
        np.isnan(map) == ((applying_noise.noise_strength_map(map, filter_size=50) == 0))
    ).all()


def test_weighted_add_no_weights():
    """
    Check that we add noise correctly when the weighting factor is 1
    """
    clean = np.ones((5, 5))
    noise = np.arange(25, dtype=np.float32).reshape(5, 5)
    strength_map = np.ones_like(noise)

    noisy = applying_noise._weighted_add(clean, noise, strength_map, weighting_factor=2)

    np.testing.assert_allclose(noisy, clean + noise * 2)


def test_weighted_add():
    """
    Check we can add noise correctly with a weighting factor
    """
    clean = np.ones((5, 5))
    noise = np.arange(25, dtype=np.float32).reshape(5, 5)
    strength_map = np.array(
        [
            [1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2],
        ]
    )

    weight = 2
    expected = noise * strength_map * 2 + clean

    noisy = applying_noise._weighted_add(
        clean, noise, strength_map, weighting_factor=weight
    )

    np.testing.assert_allclose(noisy, expected)


def test_weighted_add_nan_weight():
    """
    Check we get the right error if there are NaNs in the strength map
    """
    clean = np.ones((3, 3))
    noise = np.ones_like(clean)
    strength = np.ones_like(noise)
    weight = 2
    applying_noise._weighted_add(clean, noise, strength, weighting_factor=weight)

    strength[1, 1] = np.nan
    with pytest.raises(applying_noise.RescalingError):
        applying_noise._weighted_add(clean, noise, strength, weighting_factor=weight)


def test_weighted_add_nans():
    """
    Check that we add noise correctly with NaNs in the input
    """
    clean = np.ones((5, 5))
    clean[0, 0] = np.nan
    clean[1, 1] = np.nan
    noise = np.arange(25, dtype=np.float32).reshape(5, 5)
    strength = np.ones_like(noise)

    noisy = applying_noise._weighted_add(clean, noise, strength, weighting_factor=2)

    expected = clean + noise * 2
    expected[0, 0] = np.nan
    expected[1, 1] = np.nan

    np.testing.assert_allclose(noisy, expected)


def test_reweight_noise():
    """
    Check that the reweighting gives us the right values
    """
    before = np.arange(9).reshape(3, 3).astype(np.float32)
    clean_max = 4.0

    np.testing.assert_allclose(
        0.75 * before, applying_noise.reweight_noisy_tile(before, clean_max)
    )


def test_reweight_nans():
    """
    Check the reweighting works when there are NaNs in the input
    """
    before = np.arange(9).reshape(3, 3).astype(np.float32)
    before[1, 1] = np.nan
    clean_max = 4.0

    np.testing.assert_allclose(
        0.75 * before, applying_noise.reweight_noisy_tile(before, clean_max)
    )


def test_reweight_nan_max():
    """
    Check that we raise an error if the clean tile maximum is NaN
    """
    with pytest.raises(applying_noise.NoiseApplicationError):
        applying_noise.reweight_noisy_tile(np.empty((3, 3)), np.nan)


def test_reweight_zero_max():
    """
    Check that we raise an error if the clean tile maximum is zero
    """
    with pytest.raises(applying_noise.NoiseApplicationError):
        applying_noise.reweight_noisy_tile(np.empty((3, 3)), 0.0)


def test_add_noise_wrong_shape():
    """
    Check that we raise an error if the shapes of the clean target and noise quilt or strength map
    don't match
    """
    with pytest.raises(applying_noise.NoiseApplicationError):
        applying_noise.add_noise(
            np.empty((5, 5)), np.empty((3, 3)), np.empty((5, 5)), rng=None
        )

    with pytest.raises(applying_noise.NoiseApplicationError):
        applying_noise.add_noise(
            np.empty((5, 5)), np.empty((5, 5)), np.empty((3, 3)), rng=None
        )


class MockRNG:
    """Fake RNG class for testing"""

    def __init__(self, *, seed: int):
        """Initialize with a fixed seed value"""
        self.seed = seed

    def uniform(self, *args) -> int:
        """Returns the seed to mock a random number"""
        return self.seed


def test_add_noise():
    """
    Check that we add noise correctly with a random weighting factor
    and a simple strength map
    """

    clean = np.ones((5, 5))
    noise = np.arange(25, dtype=np.float32).reshape(5, 5)
    strength_map = np.ones_like(noise)
    strength_map[:, 2:] += 1

    # Simulate some land
    clean[0, 0] = np.nan

    # The mock RNG just returns the seed
    seed = 2
    expected = (49 / 97) * np.array(
        [
            [1.0, 3.0, 9.0, 13.0, 17.0],
            [11.0, 13.0, 29.0, 33.0, 37.0],
            [21.0, 23.0, 49.0, 53.0, 57.0],
            [31.0, 33.0, 69.0, 73.0, 77.0],
            [41.0, 43.0, 89.0, 93.0, 97.0],
        ]
    )
    expected[0, 0] = np.nan

    noisy = applying_noise.add_noise(clean, noise, strength_map, MockRNG(seed=seed))

    np.testing.assert_allclose(noisy, expected)
    assert np.isnan(noisy[0, 0])
