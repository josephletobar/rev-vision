from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch

class ArrowDataset(Dataset):
    def __init__(self, img_dir, hm_dir):
        self.img_dir = img_dir
        self.hm_dir = hm_dir

        # match PNG images with NPY heatmaps by filename stem
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

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        # load heatmap (H,W,3)
        hm = np.load(hm_path).astype(np.float32)

        # HWC → CHW
        hm = np.transpose(hm, (2, 0, 1))
        hm = torch.from_numpy(hm).float()

        # print("DEBUG IMG SHAPE:", img.shape, "   HM SHAPE:", hm.shape)

        return img, hm


# optional manual debug runner
if __name__ == "__main__":
    IMG_DIR = "data/arrow_data/images_out"
    HM_DIR  = "data/arrow_data/heatmaps"

    ds = ArrowDataset(IMG_DIR, HM_DIR)
    print("Dataset size:", len(ds))

    if len(ds) > 0:
        img, hm = ds[0]
        print("Returned image shape:", img.shape)
        print("Returned heatmap shape:", hm.shape)