import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    CLASS_MAP = {
        1: 0,  # front
        2: 1,  # back
        3: 2,  # sleeves
        4: 3   # hood
    }

    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.image_names = sorted(os.listdir(self.image_dir))
        self.label_names = sorted(os.listdir(self.label_dir))
        self.transform = transform
        self.resize = T.Resize((64, 64), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        label_path = os.path.join(self.label_dir, self.label_names[idx])

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.transform:
            image = self.transform(image)
        
        label = self.resize(label)
        label = np.array(label, dtype=np.uint8)

        # Remap classes
        remapped = np.full_like(label, 255, dtype=np.uint8)
        for original, mapped in self.CLASS_MAP.items():
            remapped[label == original] = mapped

        return image, torch.from_numpy(remapped).long()