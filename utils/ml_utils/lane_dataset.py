import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(960, 720)):
        self.image_dir = image_dir 
        self.mask_dir = mask_dir

        # get all filenames and make sure a matching mask file exists
        self.image_names = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png") and os.path.exists(os.path.join(mask_dir, f))
        ])

        # apply transformations
        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(), # convert to pytorch tensor
        ])
        
        print(f"Loaded {len(self.image_names)} image-mask pairs.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)
        mask = (mask > 0.5).float()  # binarize mask (convert all pixels to either 0 or 1)

        return image, mask 

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = LaneDataset("data/images/", "data/lane_masks/")
    print(f"Loaded {len(dataset)} image–mask pairs:\n")

    for name in dataset.image_names:
        print(f"{name} <--> {name}")