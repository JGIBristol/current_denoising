"""
Training losses and so on
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_losses(
    loss1: list[list[float]],
    loss2: list[list[float]],
    labels=("Train", "Validation"),
    axis=None,
) -> matplotlib.figure.Figure:
    """
    Plot the training and validation losses against epoch

    :param train_losses: list of lists of floats, the training losses for each epoch

    """
    assert len(loss1) == len(loss2)

    label1, label2 = labels

    epochs = np.arange(len(loss1))

    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()

    axis.plot(epochs, np.mean(loss1, axis=1), label=label1, color="C0")
    axis.plot(epochs, np.mean(loss2, axis=1), label=label2, color="C1")

    # Find quartiles - the mean might be outside this, which would be interesting wouldn't it
    loss1_upper = np.percentile(loss1, 75, axis=1)
    loss1_lower = np.percentile(loss1, 25, axis=1)

    loss2_upper = np.percentile(loss2, 75, axis=1)
    loss2_lower = np.percentile(loss2, 25, axis=1)

    axis.fill_between(epochs, loss2_lower, loss2_upper, alpha=0.5, color="C1")
    axis.fill_between(epochs, loss1_lower, loss1_upper, alpha=0.5, color="C0")

    axis.set_title("Loss")
    axis.set_xlabel("Epoch")
    axis.legend()

    return fig
