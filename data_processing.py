import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import random
from config import load_config

config = load_config('config.yaml')
# ========================
# Custom Dataset
# ========================
class Datasets(Dataset):
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
            image = Image.fromarray(image).resize((config['IMAGE_SIZE'], config['IMAGE_SIZE']))
            image = np.asarray(image).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
            image = torch.from_numpy(image).permute(2, 0, 1)
            return image
        except:
            # Return a black image instead of crashing
            return torch.zeros(3, config['IMAGE_SIZE'], config['IMAGE_SIZE'])