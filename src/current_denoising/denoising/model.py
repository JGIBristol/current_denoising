"""
Denoising model(s)

"""

from monai.networks.nets import attentionunet


def _channels(n_layers: int, init_channels: int) -> list[int]:
    """
    Get the sequence of channels used by the model from the number of layers,


    Returns a list starting with the number of channels after the first convolution,
    upsamples to 8 then successively doubles.

    :param n_layers: how many layers in our model.
    :param init_channels: how many channels after the first convolution.

    :returns: a list of the number of channels that can be passed to the monai constructor.
    """
    return [init_channels * 2**i for i in range(n_layers)]


def get_attention_unet(n_layers: int, dropout: float):
    """
    An Attention-UNet model for translating one
    2d image to another.

    Hard-coded to:
     - take a greyscale image and produce greyscale output
     - start by convolving to an 8 channel image
     - half the size of the feature map in each layer
     - double the number of channels in each layer
     - Use 3x3 kernels
    """
    # Find how many channels and strides we need based on the number of layers
    return attentionunet.AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=_channels(n_layers, init_channels=8),
        strides=[2 for _ in range(n_layers - 1)],
        kernel_size=3,
        up_kernel_size=3,
        dropout=dropout,
    )
