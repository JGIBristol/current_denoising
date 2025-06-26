"""
Deep convolutional GAN (DCGAN) implementation
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable


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

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = torch.nn.Sequential(
            *discriminator_block(config["channels"], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = config["img_size"] // 2**4
        self.adv_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * ds_size**2, 1), torch.nn.Sigmoid()
        )

    def forward(self, img):
        """Classify an image as real or fake"""
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def train(
    generator: Generator, discriminator: Discriminator, config: dict
) -> tuple[Generator, Discriminator]:
    """
    Train
    """
    generator.cuda()
    discriminator.cuda()
    config["loss"].cuda()

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=config["lr"], betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config["lr"], betas=(0.5, 0.999)
    )

    for _ in tqdm(range(config["n_epochs"])):
        for imgs in config["dataloader"]:

            # Adversarial ground truths
            valid = Variable(
                torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False
            )
            fake = Variable(
                torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False
            )

            # Configure input
            real_imgs = Variable(imgs.type(torch.cuda.FloatTensor))

            #  Train Generator
            optimizer_g.zero_grad()

            # Sample noise as generator input
            z = Variable(
                torch.cuda.FloatTensor(
                    np.random.normal(0, 1, (imgs.shape[0], config["latent_dim"]))
                )
            )

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = config["loss"](discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_g.step()

            #  Train Discriminator
            optimizer_d.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = config["loss"](discriminator(real_imgs), valid)
            fake_loss = config["loss"](discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()
