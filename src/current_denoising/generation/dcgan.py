"""
Deep convolutional GAN (DCGAN) implementation
"""

import pathlib
import warnings
from functools import cache
from typing import NamedTuple

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from ..plotting import training


class DataError(Exception):
    """General error with data"""


class ModelError(Exception):
    """Error setting up a model"""


class GenerationError(Exception):
    """Error during generation"""


class TrainingError(Exception):
    """Error during training"""


class InceptionPool3(torch.nn.Module):
    def __init__(self, extractor, mean, std):
        super().__init__()
        self.extractor = extractor

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in [0,1]; shape N x C x H x W (C=1 or 3)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        y = self.extractor(x)["feat"]  # N x 2048 x 1 x 1
        return torch.flatten(y, 1)  # N x 2048


class TileLoader(torch.utils.data.Dataset):
    """
    Dataloader for GAN training
    """

    def __init__(self, tiles: np.ndarray):
        """images should be a numpy array of shape (N, H, W)"""
        self._originals = tiles

        if np.any(np.isnan(tiles)):
            raise DataError(
                "Tiles contain NaNs - GAN training tiles should be far from land"
            )

        # Rescale to [-1, 1] for the GAN
        # We already know there are no NaNs in the data, so its safe to use
        # min and max here instead of nan-aware versions
        min_ = np.min(tiles)
        max_ = np.max(tiles)
        self.images = 2 * (tiles - min_) / (max_ - min_) - 1

    def __len__(self):
        """Total number of training patches"""
        return len(self.images)

    def __getitem__(self, idx: int):
        """Get a training patch"""
        return torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)


class GANHyperParams(NamedTuple):
    """
    Hyperparameters for GAN training
    """

    n_epochs: int
    g_lr: float
    d_lr: float
    n_critic: int
    lambda_gp: float
    generator_latent_size: int
    generator_latent_channels: int
    n_discriminator_blocks: int


class GANTrainingMetrics:
    """
    Holds the metrics collected during GAN training, and provides plotting utilities.

    Useful for illustrating the performance of the GAN and for reporting, diagnosis,
    optimisation etc. See the attributues for a list of the metrics collected.
    """

    def __init__(self, n_batches: int, n_epochs: int):
        """
        Initialise empty arrays

        """
        self.gen_losses = np.zeros((n_epochs, n_batches))
        """Generator losses per epoch; shape (n_epochs, n_batches)"""

        self.critic_losses = np.zeros((n_epochs, n_batches))
        """Critic losses per epoch; shape (n_epochs, n_batches)"""

        self.wasserstein_dists = np.zeros((n_epochs, n_batches))
        """Wasserstein distances per epoch; shape (n_epochs, n_batches)"""

        self.gradient_penalties = np.zeros((n_epochs, n_batches))
        """Gradient penalties per epoch; shape (n_epochs, n_batches)"""

        self.generator_param_gradients = np.zeros(n_epochs)
        """
        Average generator gradient norms per epoch; shape (n_epochs,)

        This is the gradient of the generator's parameters (after the update step).
        Tells us the strength of the generator's update signal.
        """

        self.critic_param_gradients = np.zeros(n_epochs)
        """
        Average critic gradient norms per epoch; shape (n_epochs,).

        This is the gradient of the critic's parameters (after the update step).
        Tells us the strength of the critic's update signal.
        """

        self.critic_interp_grad_norms = np.zeros((n_epochs, n_batches))
        """
        Average gradient norms for the interpolated samples used in the gradient penalty
        shape (n_epochs, n_batches)

        This is a data gradient, not a parameter gradient; tells us how much the critic's
        output changes with a small perturbation of the input images.
        We want this to be 1 here - if the norm is much larger than 1, the generator
        will not be able to learn from the critic's output (since small changes in the input
        massively change its decision); if it's much smaller than 1, the critic is not sensitive
        enough to the input, and again the generator will struggle to learn because it can make
        big changes without being punished.
        If you want to know why this should be around 1 (and not some other number),
        read about Kantorovich-Rubinstein duality and 1-Lipschitz functions, but I don't think
        I really understand this...

        """

    def plot_scores(self) -> plt.Figure:
        """
        Plot the model losses across training epochs.

        Creates a new figure containing one axis for plotting the generator and criticlosses.
        Does not display, save or apply tight_layout.

        :return: a new matplotlib figure
        """
        fig, axis = plt.subplots(1, 1, figsize=(15, 5), sharex=True)

        training.plot_losses(
            self.gen_losses,
            self.critic_losses,
            labels=("Generator Loss", "Discriminator Loss"),
            axis=axis,
        )
        axis.set_title("Losses")

        return fig

    def plot_gp_wd_ratio(self, lambda_gp: float) -> plt.Figure:
        """
        Plot the contribution from the gradient penalty and the Wasserstein distance
        in the discriminator loss.

        Creates a new figure containing two axes; one for the critic loss terms,
        and one for their ratio.

        :param lambda_gp: the lambda_gp hyperparameter used during training; used to scale the
                          gradient penalty during the critic update.
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 5))

        scaled_gps = lambda_gp * self.gradient_penalties

        x = np.arange(len(scaled_gps))
        axes[0].plot(x, scaled_gps.mean(axis=1), label="Gradient Penalty")
        axes[0].fill_between(
            x, scaled_gps.min(axis=1), scaled_gps.max(axis=1), alpha=0.2
        )
        axes[0].plot(
            x, self.wasserstein_dists.mean(axis=1), label="Wasserstein Distance"
        )
        axes[0].fill_between(
            x,
            self.wasserstein_dists.min(axis=1),
            self.wasserstein_dists.max(axis=1),
            alpha=0.2,
        )

        axes[1].plot(
            scaled_gps.mean(axis=1) / self.wasserstein_dists.mean(axis=1),
            label=r"$\lambda \times$GP / WD",
            color="C2",
        )
        axes[1].axhline(
            0.1,
            color="k",
            linestyle="--",
        )
        axes[1].axhline(
            0.6,
            color="k",
            linestyle="--",
        )
        axes[1].set_ylim(-0.5, 2)

        axes[1].set_title("Ratio; high -> GP dominates, low -> WD dominates")
        axes[1].legend()
        axes[0].legend()
        fig.tight_layout()

        return fig

    def plot_param_gradients(self, g_lr: float, d_lr) -> plt.Figure:
        """
        Plot the gradients of the models wrt their parameters, and their ratio
        (scaled by the ratios of their learning rates).

        :param g_lr: generator learning rate
        :param d_lr: critic learning rate

        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 5))
        axes[0].plot(
            self.generator_param_gradients, label="Generator Gradients", color="C0"
        )
        axes[0].plot(
            self.critic_param_gradients, label="Discriminator Gradients", color="C1"
        )
        axes[1].plot(
            (g_lr * self.generator_param_gradients)
            / (d_lr * self.critic_param_gradients),
            color="C2",
            label="ratio",
        )
        axes[1].set_title("G / D gradients (scaled by LR)")
        axes[1].axhline(0.8, color="k", linestyle="dashed")
        axes[1].axhline(1.2, color="k", linestyle="dashed")
        axes[1].set_ylim(0, 2)
        for axis in axes:
            axis.legend()

        fig.tight_layout()

        return fig

    def plot_critic_grad_norms(self) -> plt.Figure:
        """
        Plot norm of critic gradients wrt the input images.

        This tells us how sensitive the critic is to perturbations of
        the input tensor; the norm should be around 1.

        :return: a figure
        """
        fig, axis = plt.subplots()
        grad_norms = self.critic_interp_grad_norms
        x = np.arange(len(grad_norms))
        axis.plot(x, np.mean(grad_norms, axis=1), color="C0")
        axis.fill_between(
            x,
            np.min(grad_norms, axis=1),
            np.max(grad_norms, axis=1),
            alpha=0.2,
            color="C0",
        )
        axis.axhline(0.9, color="k", linestyle="dashed")
        axis.axhline(1.1, color="k", linestyle="dashed")
        axis.set_ylim(0, 2)
        axis.set_title(f"Gradient Norm")
        fig.tight_layout()

        return fig


class Generator(torch.nn.Module):
    """
    Generator for the noise patches.

    Generates an image from a random input vector; should be trained on
    deep-ocean patches of satellite data, such that the slowly-varying
    signal is not learned, but the small-scale noise is.

    Assumes the input images have a spatial size of 1 and many channels;
    learns to scale the input up from 1x1 to the desired output size

    """

    @staticmethod
    def _n_upsamples(img_size: int, latent_size: int):
        """
        Find the number of upsampling steps needed to go from the latent size to the output size
        """
        n_upsamples = int(np.log2(img_size // latent_size))
        if latent_size * (2**n_upsamples) != img_size:
            raise ModelError(
                f"Must be able to upsample from latent size of {latent_size} to {img_size} via powers of 2\n"
                f"got {latent_size}x2^{n_upsamples} = {latent_size * (2**n_upsamples)} != {img_size}"
            )
        return n_upsamples

    def __init__(self, img_size: int, latent_channels: int, latent_size: int):
        """
        defines the architecture

        :param img_size: size of the generated image (assumed square).
                         Must be a power of 2.
        :param latent_channels: number of channels in the input noise map
        :param latent_size: size of the input noise map, assumed square.
        """
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        # Other things that I might want to make configurable later
        # This is the number of channels after the first convolution
        self.base_channels = 128

        # Our generator starts by projecting a high-channel, 1x1 image into a small
        # low-res feature map
        num_upsamples = self._n_upsamples(img_size, self.latent_size)

        super().__init__()

        # Initial conv to process the noise map
        self.l1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.latent_channels,
                self.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm2d(self.base_channels),
            torch.nn.ReLU(inplace=True),
        )

        # Build up convolutional blocks
        blocks = []

        in_channels = self.base_channels
        for k in range(num_upsamples - 1):
            out_channels = self.base_channels // (2 ** (k + 1))
            blocks += [
                torch.nn.Upsample(scale_factor=2, mode="nearest"),
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Final block - upsample to the right size + channels
        # No batch norm here before tanh
        blocks += [
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh(),
        ]

        self.conv_blocks = torch.nn.Sequential(*blocks)

    def forward(self, z):
        """Generate an image from noise vector z"""
        out = self.l1(z)
        img = self.conv_blocks(out)
        return img

    def gen_imgs(self, batch_size: int, noise_size: int | None = None) -> torch.Tensor:
        """
        Generate a batch of images, using the device that this model is on.

        Will break or do something weird if this model is on multiple devices, but that won't happen...

        :param batch_size: how many images
        :param noise_size: size of input noise map.
                           If not specified, defaults to the size that the model was trained on.
        """
        latent_size = noise_size if noise_size is not None else self.latent_size
        z_d = torch.randn(
            batch_size,
            self.latent_channels,
            latent_size,
            latent_size,
            device=_get_device(self),
            dtype=torch.float32,
        )
        return self(z_d)


class Discriminator(torch.nn.Module):
    """
    Classifies a single-channel image as real or fake

    Not technically a discriminator, since this is a WGAN - does not provide
    normalised outputs, so it's a "critic" rather than a "discriminator".
    """

    def __init__(self, img_size: int, n_blocks: int):
        """
        define the arch

        :param img_size: size of the generated image (assumed square).
        :param n_blocks: number of convolutional blocks
        """
        super().__init__()

        self.n_channels = 1

        def discriminator_block(in_filters: int, out_filters: int):
            block = [
                spectral_norm(
                    torch.nn.Conv2d(
                        in_filters, out_filters, kernel_size=3, stride=2, padding=1
                    )
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ]
            return block

        blocks = []
        in_ch, out_ch = self.n_channels, 16
        for _ in range(n_blocks):
            blocks += discriminator_block(in_ch, out_ch)

            in_ch = out_ch
            out_ch = out_ch * 2

        self.model = torch.nn.Sequential(*blocks)

        # The height and width of downsampled image
        ds_size = img_size // 2**n_blocks

        # We doubled the number of out_ch one time too many, so use
        # in_ch here since it is half the out_ch from the last block
        # No need to add any activation since we using a WGAN - we don't
        # need to squash the output to [0, 1]
        self.adv_layer = torch.nn.Linear(in_ch * ds_size**2, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images as real or fake"""
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def _gradient_penalty(
    critic: torch.nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    WGAN gradient penalty; instead of clipping weights, we penalize the model if |gradient| is not 1.

    We do this by interpolating between real and fake samples (add a random amount of real and fake
    together), and compute the gradient of the discriminator's output with respect
    to these interpolated samples.

    We then penalize the model if the norm of this gradient is not 1, using a mean-squared error.

    :param critic: the discriminator/critic model
    :param real_samples: batch of real samples
    :param fake_samples: batch of fake samples
    :param alpha: weight for real vs fake samples

    """
    if real_samples.shape != fake_samples.shape:
        raise TrainingError(
            f"real_samples and fake_samples must have the same shape; got {real_samples.shape} and {fake_samples.shape}"
        )

    mixture = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = critic(mixture)
    fake = torch.ones(d_interpolates.size(), device=real_samples.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=mixture,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty, grad_norm.mean()


def _to_rgb(images: torch.Tensor) -> torch.Tensor:
    return images.repeat(1, 3, 1, 1)


def _gen_imgs(generator: Generator, batch_size: int, noise_size: int) -> torch.Tensor:
    """
    Generate a batch of images from the generator.

    Uses the device that the generator is on (cannot have parameters)
    on multiple devices).

    :param generator: the generator model
    :param batch_size: number of images to generate
    :param noise_size: spatial size of the input noise map

    :return: a (batch_size, 1, H, W) tensor of generated images
    """
    warnings.warn("use the class method instead", category=DeprecationWarning)
    return generator.gen_imgs(batch_size, noise_size)


@cache
def _get_device(model: torch.nn.Module) -> str:
    """
    Get (a) device that a model is currently on.
    Will fail if the model has parameters on multiple devices.
    """
    devices = set(x.device for x in model.parameters())

    try:
        # Unpack to get the device - there should only be one
        (device,) = devices
        return device

    except ValueError:
        if len(devices) != 1:
            devices_str = ", ".join(str(x) for x in devices)
            errmsg = f"Expected one device: found {devices_str}" f"For model\n{model}"
            raise ModelError(errmsg)


def _random_orientation_augment(batch: torch.Tensor) -> torch.Tensor:
    """Apply random 0/90/180/270 rotation and optional horizontal flip per-image."""
    out = batch.clone()

    # Choose how many rotations to make for each image, and whether to flip
    k = torch.randint(0, 4, (out.size(0),), device=out.device)
    flip = torch.rand(out.size(0), device=out.device) < 0.5

    for rot in range(4):
        # These are the tiles that we will rotate
        if not (mask := k == rot).any():
            continue

        # Deal with rotations
        if mask.any():
            out_subset = torch.rot90(out[mask], k=rot, dims=(2, 3))

        # Deal with flips
        if (subset_flip := flip[mask]).any():
            out_subset[subset_flip] = torch.flip(out_subset[subset_flip], dims=(3,))

        out[mask] = out_subset

    return out


def _train_critic(
    generator: Generator,
    discriminator: Discriminator,
    optimiser: torch.optim.Optimizer,
    hyperparams: GANHyperParams,
    batch_size: int,
    latent_size: int,
    real_imgs: torch.Tensor,
    alphas: torch.Tensor,
) -> tuple[float, float, float, float]:
    """
    Train the critic (mislabed "Discriminator" here) on one set of real and fake images.

    :param generator: the generator model
    :param discriminator: the critic
    :param optimiser: optimiser for the critic
    :param hyperparams: hyperparam object
    :param real_imgs: a batch of real images
    :param alpha: floats used to interpolate between real/fake images when finding the
                  gradient penalty term for the loss.

    :return: estimate of the mean Wasserstein distance between critic's outputs for real + fake images.
    :return: gradient penalty term for the loss
    :return: gradient norm for the critic (wrt the data, not the parameters)
    :reuturn: the loss
    """
    if not len(alphas) == hyperparams.n_critic:
        raise TrainingError(f"{hyperparams.n_critic=} but {len(alphas)=}")

    w_accum = 0.0
    gp_accum = 0.0
    interp_grad_accum = 0.0
    for i in range(hyperparams.n_critic):
        optimiser.zero_grad()

        # Generate some fake images for discriminator training
        # Detach since we don't want to propagate through the generator
        # input size of 1 during training
        gen_imgs_d = generator.gen_imgs(batch_size, latent_size).detach()

        # Perform data augmentation
        gen_imgs_d = _random_orientation_augment(gen_imgs_d)
        augmented_real_imgs = _random_orientation_augment(real_imgs)

        # detatch the generator output to avoid backpropagating through it
        # we don't want to update the generator during discriminator training
        real_loss = discriminator(augmented_real_imgs)
        fake_loss = discriminator(gen_imgs_d)

        # Gradient penalty
        gp, grad_norm = _gradient_penalty(
            discriminator, augmented_real_imgs, gen_imgs_d, alphas[i]
        )
        d_obj = fake_loss.mean() - real_loss.mean()
        d_loss = d_obj + hyperparams.lambda_gp * gp

        d_loss.backward()
        optimiser.step()

        w_accum -= d_obj.detach().item()
        gp_accum += gp.detach().item()
        interp_grad_accum += grad_norm.detach().item()

    return (
        w_accum / hyperparams.n_critic,
        gp_accum / hyperparams.n_critic,
        interp_grad_accum / hyperparams.n_critic,
        d_loss.item(),
    )


def _train_generator(
    generator: Generator,
    discriminator: Discriminator,
    optimiser: torch.optim.Optimizer,
    batch_size: int,
    noise_size: int,
) -> float:
    """
    Train the generator.

    :return: the generator loss
    """
    optimiser.zero_grad()

    # Generate a batch of images (with augmentation)
    # Now we do want to update the generator, so don't detatch
    gen_imgs = generator.gen_imgs(batch_size, noise_size)
    gen_imgs = _random_orientation_augment(gen_imgs)

    g_loss = -discriminator(gen_imgs).mean()

    g_loss.backward()
    optimiser.step()
    return g_loss.item()


def _grad_norm(model: torch.nn.Module) -> float:
    """
    Parameter L2 gradient norm
    """
    return torch.norm(
        torch.stack(
            [
                p.grad.detach().data.norm(2)
                for p in model.parameters()
                if p.grad is not None
            ]
        ),
        2,
    ).item()


def train(
    generator: Generator,
    discriminator: Discriminator,
    dataloader: torch.utils.data.DataLoader,
    hyperparams: GANHyperParams,
    *,
    output_dir: pathlib.Path | None = None,
) -> tuple[Generator, Discriminator, GANTrainingMetrics]:
    """
    Train the provided generator and discriminator.

    :param generator: a generator model
    :param discriminator: classifies images as real or fake
    :param dataloader: dataloader for the training data
    :param output_dir: directory to save training plots to. If None, no plots will be saved.

    :return: trained generator
    :return: trained discriminator
    :return: GAN training metrics
    :raises TrainingError: if the Generator + discriminator are on different devices
    """
    device = _get_device(generator)
    if device != _get_device(discriminator):
        raise TrainingError(
            f"Got different devices for generator and discriminator:\n\t{device} and {_get_device(discriminator)}"
        )

    if not dataloader.drop_last:
        raise TrainingError(
            "Cannot train with dataloader.drop_last=True; we need to drop the last"
            "possibly incomplete batch to ensure that the critic and generator train"
            "at the same rate. We could work around this, but I don't want to"
        )

    # Set up optimisers
    betas = (0.0, 0.9)
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=hyperparams.g_lr, betas=betas
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=hyperparams.d_lr,
        betas=betas,
    )

    n_batches = len(dataloader)
    training_metrics = GANTrainingMetrics(
        n_batches=n_batches, n_epochs=hyperparams.n_epochs
    )

    # Pre-generate weights for interpolation between real and fake samples when we do the
    # gradient penalty
    batch_size = dataloader.batch_size
    alphas = torch.rand(
        (hyperparams.n_epochs, hyperparams.n_critic, batch_size, 1, 1, 1),
        device=device,
        requires_grad=False,
    )

    for epoch in tqdm(range(hyperparams.n_epochs)):
        for batch, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]

            # Train Discriminator first
            w, gp, grad_norm, loss = _train_critic(
                generator,
                discriminator,
                optimizer_d,
                hyperparams,
                batch_size,
                hyperparams.generator_latent_size,
                imgs,
                alphas[epoch],  # Get a sub-tensor of alphas
            )
            training_metrics.wasserstein_dists[epoch, batch] = w
            training_metrics.gradient_penalties[epoch, batch] = gp
            training_metrics.critic_interp_grad_norms[epoch, batch] = grad_norm
            training_metrics.critic_losses[epoch, batch] = loss

            # Now train the generator
            loss = _train_generator(
                generator,
                discriminator,
                optimizer_g,
                batch_size,
                hyperparams.generator_latent_size,
            )
            training_metrics.gen_losses[epoch, batch] = loss

        # Find the parameter gradients (i.e. how big the update step was this epoch)
        training_metrics.generator_param_gradients[epoch] = _grad_norm(generator)
        training_metrics.critic_param_gradients[epoch] = _grad_norm(discriminator)

    return generator, discriminator, training_metrics


def train_new_gan(
    dataloader: torch.utils.data.DataLoader,
    hyperparams: GANHyperParams,
    device: str,
    *,
    img_size: int,
    output_dir: pathlib.Path | None = None,
) -> tuple[Generator, Discriminator, GANTrainingMetrics]:
    """
    Train a new GAN

    :param dataloader: dataloader for the training data
    :param device: device to use for training - i.e. "cpu" or "cuda".
    :param img_size: size of the images
    :param output_dir: directory to save training plots to. If None, no plots will be saved.

    :return: trained generator
    :return: trained discriminator
    :return: GAN training metrics
    """
    generator = Generator(
        img_size,
        hyperparams.generator_latent_channels,
        hyperparams.generator_latent_size,
    )
    discriminator = Discriminator(img_size, hyperparams.n_discriminator_blocks)

    generator.to(device)
    discriminator.to(device)

    return train(
        generator,
        discriminator,
        dataloader,
        hyperparams,
        output_dir=output_dir,
    )


def generate_tiles(
    generator: Generator, *, n_tiles: int, noise_size: int, device: str
) -> np.ndarray:
    """
    Generate tiles using the generator, scaled to between 0 and 1.

    Returns the tiles as a numpy array of shape (n_tiles, tile_size, tile_size)
    on the CPU (even if generation happens on a GPU).

    :param generator: trained generator model
    :param n_tiles: number of tiles to generate
    :param noise_size: spatial size of input noise map
    :param device: device to run the generator on

    :return: numpy array of shape (n_tiles, tile_size, tile_size)
    """
    if device not in ("cpu", "cuda"):
        raise GenerationError("device must be 'cpu' or 'cuda'")

    generator.eval()

    with torch.no_grad():
        z = torch.randn(n_tiles, generator.latent_channels, noise_size, noise_size).to(
            device
        )
        gen_tile = ((generator(z) + 1) / 2).squeeze(1).cpu().numpy()

    return gen_tile
