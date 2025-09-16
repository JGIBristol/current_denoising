"""
Test for bits of the GAN
"""

from current_denoising.generation import dcgan


def test_inference():
    """
    Check we get the right shaped output
    """
    config = {"img_size": 32, "latent_dim": 1, "channels": 1}
    generator = dcgan.Generator(config)

    tiles = dcgan.generate_tiles(generator, n_tiles=4, device="cpu")

    assert tiles.shape == (4, 32, 32)
