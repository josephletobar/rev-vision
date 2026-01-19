import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import config

import torch.nn as nn
import torchvision.models as models

class LaneSegmentationModel(nn.Module):
    def __init__(self, n_classes=1, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            self.encoder_layers = list(self.encoder.children())
            
            # Remove fully connected layer and avgpool
            self.encoder = nn.Sequential(*self.encoder_layers[:-2])
            encoder_channels = 512
        else:
            raise NotImplementedError("Only resnet18 backbone implemented for now")
        
        # Simple decoder: upsample + conv
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


# Load model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"[Lane Segmentation] CUDA active: {torch.cuda.get_device_name(0)}")
else:
    print("[Lane Segmentation] CPU only")

def load_model(weights):
    model = LaneSegmentationModel(n_classes=1).to(device)
    model.load_state_dict(torch.load(config.LANE_MODEL, map_location=device))
    model.eval()
    return model

# Transform 
transform = T.Compose([
    T.Resize((960, 720)),
    T.ToTensor(),
])

# Predict function
def deeplab_predict(input_data):
    if input_data is None:
        print(f"[deeplab_predict] None input_data in module {__name__}")
        return None, None

    model = load_model(config.LANE_MODEL)

    if isinstance(input_data, str):  # filepath
        image = Image.open(input_data).convert("RGB")
    elif isinstance(input_data, np.ndarray):  # numpy array (BGR from OpenCV)
        image = Image.fromarray(input_data[..., ::-1])  # convert BGR to RGB
    else:
        raise TypeError("Input must be a filepath or a numpy array")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(): 
        output = model(input_tensor) # run the transformed image through the model 
                                            # DeepLab returns dict; use ['out']. UNet returns tensor directly.
        
        # Convert model output to a binary NumPy mask:
        # 1) squeeze: remove extra batch/channel dims [1,1,H,W] → [H,W]
        # 2) > 0.5: threshold probabilities at 0.5; boolean mask
        # 3) float(): convert booleans to 0.0 / 1.0
        # 4) cpu(): move from GPU to CPU
        # 5) numpy(): convert to NumPy array (float32, values in {0.0, 1.0})
        pred_mask = (output.squeeze() > 0.5).float().cpu().numpy() 

    return image, pred_mask

# Run on sample images 
if __name__ == "__main__":
    test_dir = "data/images/"
    weights = "lane_deeplab_model"
    for filename in sorted(os.listdir(test_dir))[:50]:  # preview x predictions
        if not filename.endswith(".png"): continue
        img_path = os.path.join(test_dir, filename)

        if img_path is None:
            print(f"None image in module {__name__}")
            continue

        img, pred = deeplab_predict(img_path, weights)

        # Set up a plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred, cmap="gray")

        plt.suptitle(filename)
        plt.show()
