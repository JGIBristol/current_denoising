"""
Training utilities (training loop, loss etc.) for
training the denoising model

"""

import torch
import numpy as np
from monai.networks.nets import attentionunet


def train_model(
    model: attentionunet,
    *,
    n_epochs: int,
    train_data: torch.utils.data.DataLoader,
    val_data: torch.utils.data.DataLoader,
) -> tuple[attentionunet, np.ndarray, np.ndarray]:
    """
    Train the model on the provided data, using a hard-coded loss
    and optimiser.

    Uses the Adam optimiser and MSE loss for pixel-wise comparison.
    TODO NaNs in the data (which indicate land) should be replaced with
    0s.

    :param model: the model to train
    :param n_epochs: how many times the model should see the full set of training data
    :param train_data: dataloader holding training data (i.e. data which is
                       augmented + where incomplete batches are dropped).
    :param val_data: dataloader holding validation data (which looks as "real"
                     as possible).

    :returns: the trained model
    :returns: training losses as an [n_epochs, n_batches] shaped numpy array
    :returns: validation losses as an [n_epochs, n_batches] shaped numpy array
    """


def new_model():
    """
    Train a new model on the provided clean + noisy tiles.
    """
    raise NotImplementedError(
        "Once I've trained a model somewhere I might want to implement this;"
        "just copy + paste the function calls that I've chained together here"
    )
