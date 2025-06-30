"""
Plots to validate images - for example:
 - the GAN's generated images
 - the real data
 - the denoised signal
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _n_axes(batch: torch.Tensor) -> tuple[int, int]:
    """Find the number of rows and cols in our plot"""
    nrow = int(np.sqrt(batch.shape[0]))
    if nrow == np.sqrt(batch.shape[0]):
        ncol = nrow
    else:
        ncol = nrow + 1

    return nrow, ncol


def show(batch: torch.Tensor, **kwargs) -> plt.Figure:
    """
    Plot a batch of images

    This is useful for visualising the output of a GAN, or the denoised signal.

    Kwargs are passed to imshow
    """
    n_rows, ncols = _n_axes(batch)

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol * 3, nrow * 3))
    for i, axis in enumerate(axes.flat):
        axis.imshow(batch[i].cpu().detach().numpy().transpose(1, 2, 0), **kwargs)
        axis.axis("off")

    return fig


def hist(batch: torch.Tensor, **hist_kw) -> plt.Figure:
    """
    Plot a histogram of the pixel values in the batch of images

    This might correspond to velocities - we might be interested in this to see if
    our GAN is correctly generating the noise distribution.
    """
    fig, axis = plt.subplots()
    axis.hist(batch.cpu().detach().numpy().flatten(), **hist_kw)

    return fig


def fft(batch) -> plt.Figure:
    """
    Plot the FFT of the batch of images

    This is useful for visualising the frequency content of the images.
    """
    n_rows, n_cols = _n_axes(batch)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows * 3, n_cols * 3))
    for i, axis in enumerate(axes.flat):
        fft_img = np.abs(
            np.fft.fft2(batch[i].cpu().detach().numpy().transpose(1, 2, 0).squeeze())
        )
        axis.imshow(fft_img, cmap="gray", norm=LogNorm())
        axis.axis("off")

    return fig
