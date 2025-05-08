import torch.nn as nn

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
