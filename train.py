import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

# ========================
# Parameters
# ========================
IMAGE_SIZE = 128
LATENT_DIM = 100
BATCH_SIZE = 28
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Custom Dataset
# ========================
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, n=10000):
        self.image_paths = os.listdir(root_dir)
        self.image_paths = random.sample(self.image_paths, min(n, len(self.image_paths)))
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Corrupted image.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.asarray(image).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
            image = torch.from_numpy(image).permute(2, 0, 1)
            return image
        except:
            # Return a black image instead of crashing
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

# ========================
# Discriminator Model
# ========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(8 * 8 * 128, 1)
            # No sigmoid here
        )

    def forward(self, x):
        return self.model(x)

# ========================
# Generator Model
# ========================
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ========================
# Helpers
# ========================
def save_generated_images(images, epoch, n=10):
    images = (images + 1) / 2.0
    save_image(images[:n * n], f'saved/generated_plot_{epoch:03d}.png', nrow=n)

def generate_latent_points(latent_dim, n_samples):
    return torch.randn(n_samples, latent_dim, device=DEVICE)

def train():
    os.makedirs("saved", exist_ok=True)
    os.makedirs("saved/models", exist_ok=True)

    dataset = UTKFaceDataset("data/part1")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    generator = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(DEVICE)
            if torch.isnan(real_imgs).any():
                continue

            batch_size = real_imgs.size(0)

            real_labels = torch.ones((batch_size, 1), device=DEVICE)
            fake_labels = torch.zeros((batch_size, 1), device=DEVICE)

            # -------------------
            # Train Discriminator
            # -------------------
            latent = generate_latent_points(LATENT_DIM, batch_size)
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
            latent = generate_latent_points(LATENT_DIM, batch_size)
            fake_imgs = generator(latent)
            preds = discriminator(fake_imgs)

            g_loss = criterion(preds, real_labels)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"saved/models/generator_{epoch+1:03d}.pth")
            save_generated_images(fake_imgs, epoch + 1)

if __name__ == '__main__':
    train()