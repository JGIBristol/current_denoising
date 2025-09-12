"""
Deep convolutional GAN (DCGAN) implementation
"""

import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
from torchvision.models import resnet18, ResNet18_Weights
from torcheval.metrics import FrechetInceptionDistance


class DataError(Exception):
    """General error with data"""


class TileLoader(torch.utils.data.Dataset):
    """
    Used to load tiles for GAN training
    """

    def __init__(self, tiles: np.ndarray):
        """images should be a numpy array of shape (N, H, W)"""
        self._originals = tiles

        if np.any(np.isnan(tiles)):
            raise DataError(
                "Tiles contain NaNs - GAN training tiles should be far from land"
            )

        # Rescale to [-1, 1] for the GAN
        min_ = np.min(tiles)
        max_ = np.max(tiles)
        self.images = 2 * (tiles - min_) / (max_ - min_) - 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = torch.FloatTensor(self.images[idx]).unsqueeze(0)
        return image


class Generator(torch.nn.Module):
    """
    Generates noise
    """

    def __init__(self, config: dict):
        """define the arch"""
        super().__init__()

        self.init_size = config["img_size"] // 4
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(config["latent_dim"], 128 * self.init_size**2)
        )

        self.conv_blocks = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, config["channels"], 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        """Generate an image from noise vector z"""
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    """
    Classifies noise as real or fake
    """

    def __init__(self, config: dict):
        """define the arch"""
        super().__init__()

        def discriminator_block(in_filters, out_filters):
            block = [
                torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ]
            return block

        self.model = torch.nn.Sequential(
            *discriminator_block(config["channels"], 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = config["img_size"] // 2**4
        self.adv_layer = torch.nn.Linear(128 * ds_size**2, 1)

    def forward(self, img):
        """Classify an image as real or fake"""
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def _gradient_penalty(D, real_samples, fake_samples) -> torch.Tensor:
    """
    WGAN gradient penalty

    Instead of clipping weights, we penalize the model if |gradient| is not 1.
    We do this by interpolating between real and fake samples,
    and computing the gradient of the discriminator's output with respect
    to these interpolated samples.

    We then penalize the model if the norm of this gradient is not 1.

    :

    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )

    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=real_samples.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty, grad_norm.mean()


def _to_rgb(images: torch.Tensor) -> torch.Tensor:
    return images.repeat(1, 3, 1, 1)


def train(generator: Generator, discriminator: Discriminator, config: dict) -> tuple[
    Generator,
    Discriminator,
    list[list[float]],
    list[list[float]],
    list[float],
    list[list[float]],
    list[list[float]],
    list[float],
    list[float],
    list[list[float]],
]:
    """
    Train the GAN

    :param generator: generates images
    :param discriminator: classifies images as real or fake
    :param config: configuration dictionary - see `gan.ipynb` for what it should contain

    :return: trained generator
    :return: trained discriminator
    :return: list of generator losses per epoch
    :return: list of discriminator losses per epoch
    :return: fid scores
    :return: list of wasserstein distances per epoch
    :return: list of gradient penalties per epoch
    :return: list of gradient penalties per epoch
    :return: list of average generator gradients
    :return: list of average discriminator gradients
    :return: list of lists of grad norms
    """
    generator.cuda()
    discriminator.cuda()

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=config["learning_rate"], betas=(0.0, 0.9)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["learning_rate"] / config["d_g_lr_ratio"],
        betas=(0.0, 0.9),
    )

    # We'll be tracking training using the FID score
    fid_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    fid_model.to("cuda")
    fid_metric = FrechetInceptionDistance(
        feature_dim=1000, model=fid_model, device="cuda"
    )
    fid_scores = []

    gen_losses = []
    disc_losses = []
    w_dists = []
    gps = []
    grad_norms = []
    g_grads = []
    d_grads = []
    for i in tqdm(range(config["n_epochs"])):
        gen_losses.append([])
        disc_losses.append([])
        w_dists.append([])
        gps.append([])
        grad_norms.append([])

        for imgs in config["dataloader"]:
            batch_size = imgs.shape[0]

            # Ground truth labels
            real_labels = Variable(
                torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False
            )
            fake_labels = Variable(
                torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False
            )
            real_imgs = Variable(imgs.type(torch.cuda.FloatTensor))

            # Train Discriminator first
            w_accum = 0.0
            gp_accum = 0.0
            interp_grad_accum = 0.0
            for j, _ in enumerate(range(config["n_critic"])):
                optimizer_d.zero_grad()

                # Generate some fake images for discriminator training
                z_d = Variable(
                    torch.cuda.FloatTensor(
                        np.random.normal(0, 1, (batch_size, config["latent_dim"]))
                    )
                )
                gen_imgs_d = generator(z_d)

                # detatch the generator output to avoid backpropagating through it
                # we don't want to update the generator during discriminator training
                real_loss = discriminator(real_imgs)
                fake_loss = discriminator(gen_imgs_d.detach())

                gp, grad_norm = _gradient_penalty(
                    discriminator, real_imgs.data, gen_imgs_d.data
                )
                d_obj = fake_loss.mean() - real_loss.mean()
                d_loss = d_obj + config["lambda_gp"] * gp

                d_loss.backward()
                optimizer_d.step()

                w_accum -= d_obj.detach().item()
                gp_accum += gp.detach().item()
                interp_grad_accum += grad_norm.detach().item()

            current_w = w_accum / config["n_critic"]
            current_gp = gp_accum / config["n_critic"]
            current_grad_norm = interp_grad_accum / config["n_critic"]

            #  Train Generator
            optimizer_g.zero_grad()

            # Generate a batch of images
            z_g = Variable(
                torch.cuda.FloatTensor(
                    np.random.normal(0, 1, (batch_size, config["latent_dim"]))
                )
            )
            gen_imgs_g = generator(z_g)

            # Now we do want to update the generator, so don't detatch
            g_loss = -discriminator(gen_imgs_g).mean()

            g_loss.backward()
            optimizer_g.step()

            gen_losses[-1].append(g_loss.item())
            disc_losses[-1].append(d_loss.item())
            w_dists[-1].append(current_w)
            gps[-1].append(current_gp)
            grad_norms[-1].append(current_grad_norm)

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
        g_grads.append(g_grad_norm)
        d_grads.append(d_grad_norm)

        if not i % config["plot_interval"]:
            # Save the most recent batch of fake images
            out_dir = config["output_dir"] / f"epoch_{i}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Rescale from [-1, 1] to [0, 1]
            gen_imgs_g = (gen_imgs_g + 1) / 2
            vutils.save_image(
                gen_imgs_g.data,
                out_dir / "fake_images.png",
                normalize=False,
                nrow=int(np.sqrt(config["batch_size"])),
            )

            # FID
            fid_metric.reset()
            generator.eval()
            with torch.no_grad():
                for _ in range(16):
                    # Generate some images
                    z = Variable(
                        torch.cuda.FloatTensor(
                            np.random.normal(
                                0, 1, (config["batch_size"], config["latent_dim"])
                            )
                        )
                    )
                    gen_imgs_g = generator(z)
                    gen_imgs_g = (gen_imgs_g + 1) / 2
                    gen_imgs_g = _to_rgb(gen_imgs_g).cuda().float()

                    # Get some real images
                    real_imgs = next(iter(config["dataloader"])).cuda()
                    real_imgs = (real_imgs + 1) / 2
                    real_imgs = _to_rgb(real_imgs).cuda().float()

                    # Evaluate the FID
                    fid_metric.update(gen_imgs_g, is_real=False)
                    fid_metric.update(real_imgs, is_real=True)

            fid_scores.append(fid_metric.compute().item())
            generator.train()

    return (
        generator,
        discriminator,
        gen_losses,
        disc_losses,
        fid_scores,
        w_dists,
        gps,
        g_grads,
        d_grads,
        grad_norms,
    )
