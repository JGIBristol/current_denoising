"""
Deep convolutional GAN (DCGAN) implementation
"""

import pathlib
from typing import NamedTuple

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision.models import resnet18, ResNet18_Weights
from torcheval.metrics import FrechetInceptionDistance

from ..plotting import training


class DataError(Exception):
    """General error with data"""


class ModelError(Exception):
    """Error setting up a model"""


class GenerationError(Exception):
    """Error during generation"""


class TrainingError(Exception):
    """Error during training"""


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
    lr: float
    d_g_lr_ratio: float
    n_critic: int
    lambda_gp: float
    n_fid_batches: int


class GANTrainingMetrics:
    """
    Holds the metrics collected during GAN training, and provides plotting utilities.

    Useful for illustrating the performance of the GAN and for reporting, diagnosis,
    optimisation etc. See the attributues for a list of the metrics collected.
    """

    def __init__(self, n_batches: int, n_epochs: int):
        """
        Initialise empty arrays

        Sets most arrays to 0s, but the FID scores to NaN since some will be
        missing - we won't compute the FID every epoch.
        """
        self.gen_losses = np.zeros((n_epochs, n_batches))
        """Generator losses per epoch; shape (n_epochs, n_batches)"""

        self.critic_losses = np.zeros((n_epochs, n_batches))
        """Critic losses per epoch; shape (n_epochs, n_batches)"""

        self.fid_scores = np.ones(n_epochs) * np.nan
        """
        FID scores per epoch.
        We only plot every `plot_interval` epochs, so many of these will remain NaN.
        """

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
        Plot the model losses and FID score across training epochs.

        Creates a new figure containing two axes; one for the generator and critic
        losses, one for the FID score. Does not display, save or apply tight_layout.

        :param plot_interval: interval (in epochs) at which the FID was computed.
        :return: a new matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

        training.plot_losses(
            self.gen_losses,
            self.critic_losses,
            labels=("Generator Loss", "Discriminator Loss"),
            axis=axes[0],
        )
        axes[0].set_title("Losses")

        # The FID scores will be full of NaNs since we don't compute it every epoch
        indices = np.arange(len(self.fid_scores))
        keep = ~np.isnan(self.fid_scores)
        axes[1].plot(indices[keep], self.fid_scores[keep], color="C1")
        axes[1].set_title("FID Score")

        return fig

    def plot_param_gradients(self, lambda_gp: float) -> plt.Figure:
        """
        Plot the generator and critic parameter gradients across training epochs
        and their ratio.

        Creates a new figure containing two axes; one for the generator and critic
        parameter gradients, and one for their ratio.
        We expect the generator gradients to be smaller than the critic gradients,
        lying between 0.1 and 0.6 ish.

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
        axes[1].set_ylim(-3, 3)

        axes[1].set_title("Ratio; high -> GP dominates, low -> WD dominates")
        axes[0].legend()
        fig.tight_layout()

        return fig

    def plot_param_gradients(self, d_g_lr_ratio: float) -> plt.Figure:
        """
        Plot the gradients of the models wrt their parameters, and their ratio
        (scaled by the d_g_lr_ratio hyperparameter).

        :param d_g_lr_ratio: the d_g_lr_ratio hyperparameter used during training;
                             this is the factor by which the critic's learning rate
                             is less than the generator

        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 5))
        axes[0].plot(
            self.generator_param_gradients, label="Generator Gradients", color="C0"
        )
        axes[0].plot(
            self.critic_param_gradients, label="Discriminator Gradients", color="C1"
        )
        axes[1].plot(
            d_g_lr_ratio * self.generator_param_gradients / self.critic_param_gradients,
            color="C2",
            label="ratio",
        )
        axes[1].set_title("d_g_lr_ratio * G / D gradients")
        axes[1].axhline(0.8, color="k", linestyle="dashed")
        axes[1].axhline(1.2, color="k", linestyle="dashed")
        axes[1].set_ylim(0, 2)
        for axis in axes:
            axis.legend()

        fig.tight_layout()

        return fig


class Generator(torch.nn.Module):
    """
    Generator for the noise patches.

    Generates an image from a random input vector; should be trained on
    deep-ocean patches of satellite data, such that the slowly-varying
    signal is not learned, but the small-scale noise is.

    Assumes the input images have 1 channel.

    """

    def __init__(self, config: dict):
        """
        defines the architecture

        :param config: configuration dictionary
                       Should contain:
                          - img_size: size of the generated image (assumed square).
                                      Must be a power of 2
                          - latent_dim: dimension of the input noise vector
        """
        if "img_size" not in config or "latent_dim" not in config:
            raise ModelError(
                f"config must contain 'img_size' and 'latent_dim'; got {config}"
            )

        img_size = config["img_size"]
        self.latent_dim = config["latent_dim"]

        # Other things that I might want to make configurable later
        self.init_size = 4
        self.base_channels = 128

        # Our generator starts by projecting the latent space into a
        # small, low-res feature map
        num_upsamples = int(np.log2(config["img_size"]) - np.log2(self.init_size))
        if (2**num_upsamples * self.init_size) != img_size:
            raise ModelError(f"img_size must be a power of 2; got {img_size}")

        super().__init__()

        # Project the latent space into a small feature map
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.base_channels * self.init_size**2)
        )

        # Build up convolutional blocks
        blocks = [
            torch.nn.BatchNorm2d(self.base_channels),
            torch.nn.ReLU(inplace=True),
        ]

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
        out = out.view(out.shape[0], self.base_channels, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    """
    Classifies a single-channel image as real or fake

    Not technically a discriminator, since this is a WGAN - does not provide
    normalised outputs, so it's a "critic" rather than a "discriminator".
    """

    def __init__(self, config: dict):
        """
        define the arch

        :param config: configuration - should contain:
                            - img_size: size of the generated image (assumed square).
                            - n_blocks: number of convolutional blocks
        """
        if "img_size" not in config or "n_blocks" not in config:
            raise ModelError(
                f"Discriminator config must contain 'img_size' and 'n_blocks'; got {config}"
            )

        super().__init__()

        self.n_channels = 1

        def discriminator_block(in_filters: int, out_filters: int):
            block = [
                torch.nn.Conv2d(
                    in_filters, out_filters, kernel_size=3, stride=2, padding=1
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ]
            return block

        blocks = []
        in_ch, out_ch = self.n_channels, 16
        for _ in range(config["n_blocks"]):
            blocks += discriminator_block(in_ch, out_ch)

            in_ch = out_ch
            out_ch = out_ch * 2

        self.model = torch.nn.Sequential(*blocks)

        # The height and width of downsampled image
        ds_size = config["img_size"] // 2 ** config["n_blocks"]

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


def _gen_imgs(generator: Generator, batch_size: int, device: str) -> torch.Tensor:
    """
    Generate a batch of images from the generator

    :param generator: the generator model
    :param batch_size: number of images to generate
    :param device: device to use

    :return: a (batch_size, 1, H, W) tensor of generated images
    """
    z_d = torch.randn(
        batch_size, generator.latent_dim, device=device, dtype=torch.float32
    )
    return generator(z_d)


def train(
    generator: Generator,
    discriminator: Discriminator,
    dataloader: torch.utils.data.DataLoader,
    hyperparams: GANHyperParams,
    device: str,
    *,
    fid_interval: int = 10,
    output_dir: pathlib.Path | None = None,
) -> tuple[Generator, Discriminator, GANTrainingMetrics]:
    """
    Train a new GAN

    :param generator: a generator model
    :param discriminator: classifies images as real or fake
    :param dataloader: dataloader for the training data
    :param device: device to use for training - i.e. "cpu" or "cuda".
    :param fid_interval: interval (in epochs) at which the FID will be computed.
                          Also sets the interval at which generated images will be saved,
                          if output_dir is not None.
    :param output_dir: directory to save training plots to. If None, no plots will be saved.

    :return: trained generator
    :return: trained discriminator
    :return: GAN training metrics
    """
    generator.to(device)
    discriminator.to(device)

    # Set up optimisers
    betas = (0.0, 0.9)
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=hyperparams.lr, betas=betas
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=hyperparams.lr / hyperparams.d_g_lr_ratio,
        betas=betas,
    )

    # We'll be tracking training using the FID score, so set that up here too
    fid_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    fid_model.to(device)
    fid_metric = FrechetInceptionDistance(
        feature_dim=1000, model=fid_model, device=device
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

            # Ground truth labels
            # TODO should be better - use better API than Variable/FloatTensor
            real_labels = torch.tensor(
                (imgs.shape[0], 1), dtype=torch.float32, device=device
            ).fill_(1.0)
            fake_labels = torch.tensor(
                (imgs.shape[0], 1), dtype=torch.float32, device=device
            ).fill_(0.0)

            # Train Discriminator first
            # TODO function for this
            w_accum = 0.0
            gp_accum = 0.0
            interp_grad_accum = 0.0
            for i in range(hyperparams.n_critic):
                optimizer_d.zero_grad()

                # Generate some fake images for discriminator training
                # Detach since we don't want to propagate through the generator
                gen_imgs_d = _gen_imgs(generator, batch_size, device).detach()

                # detatch the generator output to avoid backpropagating through it
                # we don't want to update the generator during discriminator training
                real_loss = discriminator(imgs)
                fake_loss = discriminator(gen_imgs_d)

                # Gradient penalty
                gp, grad_norm = _gradient_penalty(
                    discriminator, imgs.data, gen_imgs_d.data, alphas[epoch, i]
                )
                d_obj = fake_loss.mean() - real_loss.mean()
                d_loss = d_obj + hyperparams.lambda_gp * gp

                d_loss.backward()
                optimizer_d.step()

                w_accum -= d_obj.detach().item()
                gp_accum += gp.detach().item()
                interp_grad_accum += grad_norm.detach().item()

            current_w = w_accum / hyperparams.n_critic
            current_gp = gp_accum / hyperparams.n_critic
            current_grad_norm = interp_grad_accum / hyperparams.n_critic

            #  Train Generator
            # TODO wrapper
            optimizer_g.zero_grad()

            # Generate a batch of images
            # Now we do want to update the generator, so don't detatch
            gen_imgs_g = _gen_imgs(generator, batch_size, device)
            g_loss = -discriminator(gen_imgs_g).mean()

            g_loss.backward()
            optimizer_g.step()

            training_metrics.gen_losses[epoch, batch] = g_loss.item()
            training_metrics.critic_losses[epoch, batch] = d_loss.item()
            training_metrics.wasserstein_dists[epoch, batch] = current_w
            training_metrics.gradient_penalties[epoch, batch] = current_gp
            training_metrics.critic_interp_grad_norms[epoch, batch] = current_grad_norm

        # Find the parameter gradients (i.e. how big the update step was this epoch)
        # TODO helper...
        g_grad_norm = torch.norm(
            torch.stack(
                [
                    p.grad.detach().data.norm(2)
                    for p in generator.parameters()
                    if p.grad is not None
                ]
            ),
            2,
        ).item()
        d_grad_norm = torch.norm(
            torch.stack(
                [
                    p.grad.detach().data.norm(2)
                    for p in discriminator.parameters()
                    if p.grad is not None
                ]
            ),
            2,
        ).item()
        training_metrics.generator_param_gradients[epoch] = g_grad_norm
        training_metrics.critic_param_gradients[epoch] = d_grad_norm

        # Evaluate FID and save some images, if desired
        if not epoch % fid_interval:
            # Just use the last batch we generated
            # Rescale from [-1, 1] to [0, 1]
            gen_imgs_g = (gen_imgs_g + 1) / 2

            # FID
            fid_metric.reset()
            generator.eval()
            with torch.no_grad():
                for _ in range(hyperparams.n_fid_batches):
                    # Generate some images, scale them and convert to RGB
                    # TODO i think this is wrong
                    gen_imgs_g = _gen_imgs(generator, batch_size, device)
                    gen_imgs_g = (gen_imgs_g + 1) / 2
                    gen_imgs_g = _to_rgb(gen_imgs_g).cuda().float()

                    # Get some real images
                    # TODO i think this is wrong
                    real_imgs = next(iter(dataloader)).cuda()
                    real_imgs = (real_imgs + 1) / 2
                    real_imgs = _to_rgb(real_imgs).cuda().float()

                    # Evaluate the FID
                    fid_metric.update(gen_imgs_g, is_real=False)
                    fid_metric.update(real_imgs, is_real=True)

            # Save the most recent batch of fake images
            if output_dir is not None:
                out_dir = output_dir / f"epoch_{epoch}"
                out_dir.mkdir(parents=True, exist_ok=True)
                vutils.save_image(
                    gen_imgs_g.data,
                    out_dir / "fake_images.png",
                    normalize=False,
                    nrow=int(np.sqrt(batch_size)),
                )

            training_metrics.fid_scores[epoch] = fid_metric.compute().item()
            generator.train()

    return generator, discriminator, training_metrics


def generate_tiles(
    generator: torch.nn.Module, *, n_tiles: int, device: str
) -> np.ndarray:
    """
    Generate tiles using the generator, scaled to between 0 and 1.

    Returns the tiles as a numpy array of shape (n_tiles, tile_size, tile_size)
    on the CPU (even if generation happens on a GPU).

    :param generator: trained generator model
    :param n_tiles: number of tiles to generate
    :param device: device to run the generator on

    :return: numpy array of shape (n_tiles, tile_size, tile_size)
    """
    if device not in ("cpu", "cuda"):
        raise GenerationError("device must be 'cpu' or 'cuda'")

    generator.eval()
    # TODO make this use the generation helper
    latent_dim = generator.l1[0].in_features

    with torch.no_grad():
        z = torch.randn(n_tiles, latent_dim).to(device)
        gen_tile = ((generator(z) + 1) / 2).squeeze(1).cpu().numpy()

    return gen_tile
