import torch.nn as nn
import torchvision.models as models
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import torch.optim as optim

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
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LaneSegmentationModel(n_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()  # single-channel mask
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# --- load model ---
model = LaneSegmentationModel(n_classes=1).to(device)
model.load_state_dict(torch.load("weights/lane_segmentation_model.pt", map_location=device))
model.eval()

# video processing
cap = cv2.VideoCapture("test_videos/bowling.mp4")

def preprocess_frame(frame, size=(720, 480)):
    # frame: BGR uint8 (H, W, 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (size[1], size[0]))  # (W, H)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(frame).unsqueeze(0)  # BCHW
    return tensor


def deeplab_predict(frame):
    img_tensor = preprocess_frame(frame).to(device)

    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_mask = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()  # float [0,1]

    # resize mask in float space (important)
    pred_mask = cv2.resize(
        pred_mask,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # convert to uint8 only for visualization
    mask_vis = (pred_mask * 255).astype(np.uint8)

    

    return mask_vis


