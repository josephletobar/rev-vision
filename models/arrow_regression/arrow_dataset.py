from torch.utils.data import Dataset
import os
import numpy as np
import cv2

# PyTorch dataset that loads each resized training image and its corresponding arrow heatmap

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

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        hm = np.load(hm_path).astype(np.float32)

        return img, hm
    
if __name__ == "__main__":
    IMG_DIR = "data/arrow_data/images"
    HM_DIR  = "data/arrow_data/heatmaps"

    ds = ArrowDataset(IMG_DIR, HM_DIR)
    print("Dataset size:", len(ds))

    if len(ds) > 0:
        img, hm = ds[0]
        print("Image shape:", img.shape)
        print("Heatmap shape:", hm.shape)

        # visualize
        disp = (img * 255).astype(np.uint8).copy()
        hm_vis = (hm.max(axis=-1) * 255).astype(np.uint8)

        cv2.imshow("image", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        cv2.imshow("heatmap max", hm_vis)
        cv2.waitKey(0)