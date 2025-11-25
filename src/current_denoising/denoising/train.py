"""
Training utilities (training loop, loss etc.) for
training the denoising model

"""

from tqdm.notebook import trange
import torch
import numpy as np
from monai.networks.nets import attentionunet


def _train_step(
    model: attentionunet,
    train_data: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    loss: torch.nn.Module,
    device: str,
) -> tuple[attentionunet, np.ndarray]:
    """
    Find the loss given the data.

    Returns 1d array of losses per batch.
    """
    model.train()

    losses = np.empty(len(train_data))
    for i, (noisy_batch, clean_batch) in enumerate(train_data):
        noisy_batch, clean_batch = noisy_batch.to(
            device, non_blocking=True
        ), clean_batch.to(device, non_blocking=True)

        optimiser.zero_grad()

        batch_loss = loss(model(noisy_batch), clean_batch)

        batch_loss.backward()
        optimiser.step()

        losses[i] = batch_loss.item()

    return model, losses


def _val_step(
    model: attentionunet,
    val_data: torch.utils.data.DataLoader,
    loss: torch.nn.Module,
    device: str,
) -> tuple[attentionunet, np.ndarray]:
    """
    Find the loss given the data.

    Returns 1d array of losses per batch.
    """
    model.eval()

    losses = np.empty(len(val_data))
    for i, (noisy_batch, clean_batch) in enumerate(val_data):
        noisy_batch, clean_batch = noisy_batch.to(
            device, non_blocking=True
        ), clean_batch.to(device, non_blocking=True)

        with torch.no_grad():
            losses[i] = loss(model(noisy_batch), clean_batch).item()

    return model, losses


def train_model(
    model: attentionunet,
    device: str,
    *,
    n_epochs: int,
    train_data: torch.utils.data.DataLoader,
    val_data: torch.utils.data.DataLoader,
    show_progress: bool = True,
) -> tuple[attentionunet, np.ndarray, np.ndarray]:
    """
    Train the model on the provided data, using a hard-coded loss
    and optimiser.

    Uses the Adam optimiser and MSE loss for pixel-wise comparison.
    NaNs in the data (which indicate land) will be replaced with 0s.

    :param model: the model to train
    :param device: either "cpu" or "cuda", probably
    :param n_epochs: how many times the model should see the full set of training data
    :param train_data: dataloader holding training data (i.e. data which is
                       augmented + where incomplete batches are dropped).
    :param val_data: dataloader holding validation data (which looks as "real"
                     as possible).
    :param show_progress: show a (jupyter-notebook compatible) progress bar

    :returns: the trained model
    :returns: training losses as an [n_epochs, n_batches] shaped numpy array
    :returns: validation losses as an [n_epochs, n_batches] shaped numpy array
    """
    optim = torch.optim.Adam(model.parameters())
    loss = torch.nn.MSELoss()

    model.to(device)

    train_losses = np.empty((n_epochs, len(train_data)))
    val_losses = np.empty((n_epochs, len(val_data)))

    pbar = trange(n_epochs)
    for i in pbar:
        model, _loss = _train_step(model, train_data, optim, loss, device)
        train_losses[i] = _loss

        model, _loss = _val_step(model, val_data, loss, device)
        val_losses[i] = _loss

        pbar.set_description(f"Val loss: {_loss.mean():.3f}")

    return model, train_losses, val_losses


def new_model():
    """
    Train a new model on the provided clean + noisy tiles.
    """
    raise NotImplementedError(
        "Once I've trained a model somewhere I might want to implement this;"
        "just copy + paste the function calls that I've chained together here"
    )
