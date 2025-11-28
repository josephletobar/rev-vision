import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ArrowXYDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

        # Match files by stem
        self.items = []
        for f in os.listdir(label_dir):
            if f.endswith(".npy"):
                stem = f.replace(".npy", "")
                img_path = os.path.join(img_dir, stem + ".png")
                label_path = os.path.join(label_dir, f)

                if os.path.exists(img_path):
                    self.items.append((img_path, label_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label_path = self.items[idx]

        # image: read, convert, normalize
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # 3,H,W

        # labels: load 6 floats
        coords = np.load(label_path).astype(np.float32)   # shape (6,)
        coords = torch.from_numpy(coords)

        return img, coords