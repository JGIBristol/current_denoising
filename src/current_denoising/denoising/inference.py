"""
Use the model
"""

import torch
import numpy as np
from monai.networks.nets import attentionunet


def denoise(net: attentionunet, target: np.ndarray) -> np.ndarray:
    """
    Denoise the provided noisy tiles

    NaN values will be replaced with 0s before denoising.

    :param net: a trained denoiser model.
    :param target: the noisy image as a DxD shaped numpy array

    :returns: the denoised image
    """
    net.eval()
    device = next(net.parameters()).device

    target = np.asarray(target, dtype=np.float32)
    nan_mask = np.isnan(target)
    tensor = torch.from_numpy(np.nan_to_num(target, nan=0.0)).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = net(tensor).cpu().squeeze().numpy()

    return np.where(nan_mask, np.nan, output)
