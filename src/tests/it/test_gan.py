"""
Test for bits of the GAN
"""

import pytest

import torch
from current_denoising.generation import dcgan


@pytest.fixture
def generator():
    """uninitalised generator"""
    config = {"img_size": 32, "latent_dim": 1}
    return dcgan.Generator(config)


@pytest.fixture
def discriminator():
    """uninitalised discriminator"""
    config = {"img_size": 32, "n_blocks": 4}
    return dcgan.Discriminator(config)


def test_invalid_input_size():
    """
    Check we get an error if the input size is invalid
    """
    with pytest.raises(dcgan.ModelError):
        config = {"img_size": 28, "latent_dim": 1}
        dcgan.Generator(config)
    with pytest.raises(dcgan.ModelError):
        config = {"img_size": 36, "latent_dim": 1}
        dcgan.Generator(config)


def test_invalid_gen_config():
    """
    Check we get an error if the input size is invalid
    """
    with pytest.raises(dcgan.ModelError):
        config = {"latent_dim": 1}
        dcgan.Generator(config)
    with pytest.raises(dcgan.ModelError):
        config = {"img_size": 32}
        dcgan.Generator(config)


def test_invalid_discriminator_config():
    """
    Check we get an error if the config doesn't contain the right stuff
    """
    with pytest.raises(dcgan.ModelError):
        dcgan.Discriminator({"n_blocks": 4})
    with pytest.raises(dcgan.ModelError):
        dcgan.Discriminator({"img_size": 32})


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


def test_discriminator(discriminator):
    """
    Check we can evaluate with the discriminator
    """
    x = torch.randn((4, 1, 32, 32))
    out = discriminator(x)
    assert out.shape == (4, 1)
