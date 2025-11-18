"""
Test for the denoising model + utility functions
"""

from current_denoising.denoising import model


def test_channels():
    """
    Check we get the right number of channels
    """
    assert model._channels(4, 8) == [8, 16, 32, 64]
