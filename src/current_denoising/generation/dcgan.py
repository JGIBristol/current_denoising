"""
Deep convolutional GAN (DCGAN) implementation
"""

import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils


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
                torch.nn.Dropout2d(0.25),
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def train(
    generator: Generator, discriminator: Discriminator, config: dict
) -> tuple[Generator, Discriminator, list[list[float]], list[list[float]]]:
    """
    Train
    """
    generator.cuda()
    discriminator.cuda()
    config["loss"].cuda()

    optimizer_g = torch.optim.RMSprop(
        generator.parameters(), lr=config["learning_rate"]
    )
    optimizer_d = torch.optim.RMSprop(
        discriminator.parameters(), lr=config["learning_rate"]
    )

    gen_losses = []
    disc_losses = []
    for i in tqdm(range(config["n_epochs"])):
        gen_losses.append([])
        disc_losses.append([])

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
            for _ in range(config["n_critic"]):
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
                d_loss = (
                    fake_loss.mean()
                    - real_loss.mean()
                    + config["lambda_gp"]
                    * _gradient_penalty(discriminator, real_imgs.data, gen_imgs_d.data)
                )

                d_loss.backward()
                optimizer_d.step()

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

        if not i % 20:
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

    return generator, discriminator, gen_losses, disc_losses
