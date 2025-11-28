from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
import random


def augment_image(img):
    # img is HWC RGB float32 in [0,1]
    out = img.copy()

    # --- brightness jitter ---
    if random.random() < 0.6:
        factor = 1.0 + (random.uniform(-0.20, 0.20))  # ±20%
        out = np.clip(out * factor, 0, 1)

    # --- contrast jitter ---
    if random.random() < 0.6:
        mean = out.mean()
        factor = 1.0 + random.uniform(-0.15, 0.15)
        out = np.clip((out - mean) * factor + mean, 0, 1)

    # --- slight gaussian blur ---
    if random.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    # --- tiny gaussian noise ---
    if random.random() < 0.6:
        noise = np.random.normal(0, 0.01, out.shape)  # std=1%
        out = np.clip(out + noise, 0, 1)

    return out



class ArrowDataset(Dataset):
    def __init__(self, img_dir, hm_dir, augment=False):
        self.img_dir = img_dir
        self.hm_dir = hm_dir
        self.augment = augment

        self.items = []
        for f in os.listdir(self.hm_dir):
            if f.endswith(".npy"):
                stem = f.replace(".npy", "")
                img_path = os.path.join(self.img_dir, stem + ".png")
                hm_path  = os.path.join(self.hm_dir, f)

                if os.path.exists(img_path):
                    self.items.append((img_path, hm_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, hm_path = self.items[idx]

        # load image (H,W,3)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # --- apply augmentations only to image ---
        if self.augment:
            img = augment_image(img)

        # convert image HWC -> CHW
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        # load heatmap
        hm = np.load(hm_path).astype(np.float32)
        hm = torch.from_numpy(np.transpose(hm, (2, 0, 1))).float()

        return img, hm



# debug
if __name__ == "__main__":
    IMG_DIR = "data/arrow_data/images_out"
    HM_DIR  = "data/arrow_data/heatmaps"

    ds = ArrowDataset(IMG_DIR, HM_DIR, augment=True)
    print("Dataset size:", len(ds))

    if len(ds) > 0:
        img, hm = ds[0]
        print("Returned image shape:", img.shape)
        print("Returned heatmap shape:", hm.shape)