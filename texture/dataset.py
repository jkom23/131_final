import csv
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config


def load_labels(csv_path: str) -> List[Tuple[str, str]]:
    """Read (crop_filename, texture_class) pairs from the labels CSV."""
    pairs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["crop_filename"].strip()
            label = row["texture_class"].strip()
            if label in config.TEXTURE_CLASSES:
                pairs.append((fname, label))
    return pairs


class RegionCropDataset(Dataset):
    def __init__(self, crops_dir: str, labels_csv: str, augment: bool = True):
        self.crops_dir = crops_dir
        self.samples = load_labels(labels_csv)
        self.class_to_idx = {c: i for i, c in enumerate(config.TEXTURE_CLASSES)}
        sz = config.TEXTURE_INPUT_SIZE

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((sz, sz)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fname, label = self.samples[idx]
        path = os.path.join(self.crops_dir, fname)
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Crop not found: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb), self.class_to_idx[label]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for imbalanced classes."""
        counts = np.zeros(len(config.TEXTURE_CLASSES), dtype=np.float32)
        for _, label in self.samples:
            counts[self.class_to_idx[label]] += 1
        counts = np.where(counts == 0, 1, counts)
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32)
