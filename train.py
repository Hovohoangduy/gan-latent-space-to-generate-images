import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_processing import Datasets
from gan_model import Generator, Discriminator
from generate_latent_space import generate_latent_points
from config import load_config

config = load_config('config.yaml')

# ========================
# Helpers
# ========================
def save_generated_images(images, epoch, n=10):
    images = (images + 1) / 2.0
    save_image(images[:n * n], f'saved/generated_plot_{epoch:03d}.png', nrow=n)

def train():
    os.makedirs("saved", exist_ok=True)
    os.makedirs("saved/models", exist_ok=True)

    dataset = Datasets(config['DATASET_PATH'])
    dataloader = DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True)

    generator = Generator(config['LATENT_DIM']).to(config['DEVICE'])
    discriminator = Discriminator().to(config['DEVICE'])

    opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config['EPOCHS']):
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(config['DEVICE'])
            if torch.isnan(real_imgs).any():
                continue

            batch_size = real_imgs.size(0)

            real_labels = torch.ones((batch_size, 1), device=config['DEVICE'])
            fake_labels = torch.zeros((batch_size, 1), device=config['DEVICE'])

            # -------------------
            # Train Discriminator
            # -------------------
            latent = generate_latent_points(config['LATENT_DIM'], batch_size)
            fake_imgs = generator(latent).detach()

            real_preds = discriminator(real_imgs)
            fake_preds = discriminator(fake_imgs)

            d_loss_real = criterion(real_preds, real_labels)
            d_loss_fake = criterion(fake_preds, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # -------------------
            # Train Generator
            # -------------------
            latent = generate_latent_points(config['LATENT_DIM'], batch_size)
            fake_imgs = generator(latent)
            preds = discriminator(fake_imgs)

            g_loss = criterion(preds, real_labels)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        print(f"Epoch [{epoch+1}/{config['EPOCHS']}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"saved/models/generator_{epoch+1:03d}.pth")
            save_generated_images(fake_imgs, epoch + 1)

if __name__ == '__main__':
    train()