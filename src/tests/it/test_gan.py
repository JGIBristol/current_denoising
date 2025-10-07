"""
Test for bits of the GAN
"""

import pytest

import torch
from current_denoising.generation import dcgan


@pytest.fixture
def generator():
    """uninitalised generator"""
    return dcgan.Generator(img_size=32, latent_dim=1)


@pytest.fixture
def discriminator():
    """uninitalised discriminator"""
    return dcgan.Discriminator(img_size=32, n_blocks=4)


def test_invalid_input_size():
    """
    Check we get an error if the input size is invalid
    """
    with pytest.raises(dcgan.ModelError):
        dcgan.Generator(img_size=28, latent_dim=1)
    with pytest.raises(dcgan.ModelError):
        config = {"img_size": 36, "latent_dim": 1}
        dcgan.Generator(img_size=36, latent_dim=1)


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


def test_grad_penalty_sizes(discriminator):
    """
    Check we get the right error if we pass tensors of the wrong shape to the gradient penalty fcn
    """
    x = torch.ones((4, 1, 32, 32), requires_grad=True)
    y = torch.ones((3, 1, 32, 32), requires_grad=True)
    with pytest.raises(dcgan.TrainingError):
        dcgan._gradient_penalty(discriminator, x, y, 0)


class FakeDiscriminator(torch.nn.Module):
    """
    Fake discriminator that just does a sum
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1).sum(dim=1, keepdim=True)


def test_grad_penalty_values():
    """
    Check we get the right value from the gradient penalty function
    """
    # With these inputs, every element in the interpolated tensor is (1-alpha)
    alpha = 0.5
    real = torch.ones((4, 1, 32, 32), requires_grad=True)
    fake = torch.zeros((4, 1, 32, 32), requires_grad=True)

    # Since the critic is just a sum, it will return 32*32*(1-alpha) for each input
    disc = FakeDiscriminator()

    # Since the critic is a sum, the gradient will be 1 for each input element
    # So the gradient is a tensor of 1s
    # This means the norm is sqrt(32*32)
    # And the penalty is (32-1)^2
    expected_gp, expected_norm = torch.tensor(
        31**2, dtype=torch.float32
    ), torch.tensor(32, dtype=torch.float32)

    gp, grad_norm = dcgan._gradient_penalty(disc, real, fake, alpha)

    assert torch.isclose(gp, expected_gp)
    assert torch.isclose(grad_norm, expected_norm)
