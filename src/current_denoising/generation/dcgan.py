"""
Deep convolutional GAN (DCGAN) implementation
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, config: dict):
        super(Generator, self).__init__()

        self.init_size = config["img_size"] // 4
        self.l1 = nn.Sequential(
            nn.Linear(config["latent_dim"], 128 * self.init_size**2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, config["channels"], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, config: dict):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(config["channels"], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = config["img_size"] // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
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

    for epoch in tqdm(range(config["n_epochs"])):
        for imgs in config["dataloader"]:

            # Adversarial ground truths
            valid = Variable(
                torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False
            )
            fake = Variable(
                torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False
            )

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            #  Train Generator
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
            )

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
