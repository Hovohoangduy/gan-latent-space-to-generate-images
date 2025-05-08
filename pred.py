import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy import linspace, mean, expand_dims
from torchvision import transforms
from torch.autograd import Variable

# Define a simple generator model for illustration (you would replace this with your own model)
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 128 * 128)  # Assuming 128x128 images with 3 channels
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(x.size(0), 3, 128, 128)  # Reshape to 3x128x128
        return self.tanh(x)  # Output images in the range [-1, 1]

# Function to generate random latent points
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    return torch.from_numpy(z_input).float()

# Function to plot generated images
def plot_generated(examples, n):
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i].permute(1, 2, 0).cpu().detach().numpy())
    plt.show()

# Function for linear interpolation
def interpolate_points(p1, p2, n_steps=10):
    ratios = linspace(0, 1, num=n_steps)
    vectors = []
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return torch.stack(vectors)

# Load model (here, we're assuming the model is defined, and weights are loaded separately)
latent_dim = 100
model = Generator(latent_dim)
model.load_state_dict(torch.load('saved_data_during_training/models/generator_model_128x128_100.pth'))
model.eval()

# Generate latent vectors
latent_points = generate_latent_points(latent_dim, 25)
latent_points = Variable(latent_points)

# Generate images using the model
with torch.no_grad():
    generated_images = model(latent_points)
    # Scale images from [-1, 1] to [0, 1] for plotting
    generated_images = (generated_images + 1) / 2.0

# Plot the generated images
plot_generated(generated_images, 5)

# Interpolation between two latent points
pts = generate_latent_points(latent_dim, 2)
pts = Variable(pts)

# Interpolate points in latent space
interpolated = interpolate_points(pts[0], pts[1])

# Generate images using the interpolated latent points
with torch.no_grad():
    interpolated_images = model(interpolated)
    interpolated_images = (interpolated_images + 1) / 2.0

# Plot the interpolated images
plot_generated(interpolated_images, len(interpolated))

# Arithmetic with latent vectors
# Example: using specific latent points
# Assume you have identified indices for features of interest
feature1_ix = [3, 39, 40]
feature2_ix = [4, 7, 8]
feature3_ix = [9, 10, 11, 31]

# Retrieve latent vectors corresponding to specific features
latent_points = generate_latent_points(latent_dim, 100)

# Function to average latent points for a specific feature
def average_points(points, ix):
    zero_ix = [i - 1 for i in ix]
    vectors = points[zero_ix]
    avg_vector = mean(vectors, axis=0)
    return torch.from_numpy(avg_vector).float()

# Average vectors for each feature
feature1 = average_points(latent_points, feature1_ix)
feature2 = average_points(latent_points, feature2_ix)
feature3 = average_points(latent_points, feature3_ix)

# Vector arithmetic: feature1 - feature2 + feature3
result_vector = feature1 - feature2 + feature3

# Generate image using the new calculated vector
result_vector = result_vector.unsqueeze(0)  # Add batch dimension
with torch.no_grad():
    result_image = model(result_vector)
    result_image = (result_image + 1) / 2.0

# Plot the result
plt.imshow(result_image[0].permute(1, 2, 0).cpu().detach().numpy())
plt.show()