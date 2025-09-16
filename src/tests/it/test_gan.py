"""
Test for bits of the GAN
"""

import pytest

from current_denoising.generation import dcgan


@pytest.fixture
def generator():
    """uninitalised generator"""
    config = {"img_size": 32, "latent_dim": 1, "channels": 1}
    return dcgan.Generator(config)


def test_invalid_device(generator):
    """
    Check we raise an error if the device is invalid
    """
    with pytest.raises(dcgan.GenerationError):
        dcgan.generate_tiles(generator, n_tiles=4, device="invalid")


def test_inference(generator):
    """
    Check we get the right shaped output
    """
    tiles = dcgan.generate_tiles(generator, n_tiles=4, device="cpu")

    assert tiles.shape == (4, 32, 32)
