"""
Test for the denoising model + utility functions
"""

import numpy as np

from current_denoising.denoising import model, data, train


def test_channels():
    """
    Check we get the right number of channels
    """
    assert model._channels(4, 8) == [8, 16, 32, 64]


def test_train_denoiser():
    """
    Check we can train a simple model
    """
    tile = np.arange(12).reshape((3, 2, 2))
    train_loader = data.dataloader(
        tile, tile, data.DataConfig(train=True, batch_size=2, num_workers=0)
    )
    val_loader = data.dataloader(
        tile, tile, data.DataConfig(train=False, batch_size=2, num_workers=0)
    )

    n_epochs = 1
    net = model.get_attention_unet(2, 0)
    net, train_loss, val_loss = train.train_model(
        net, "cpu", n_epochs=1, train_data=train_loader, val_data=val_loader
    )

    assert train_loss.shape == (1, 1)  # We threw away the second incomplete batch
    assert val_loss.shape == (1, 2)
