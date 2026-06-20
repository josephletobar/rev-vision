import torch.nn as nn
import torchvision.models as models
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import torch.optim as optim
import config

try:
    from torchvision.models import ResNet18_Weights
except ImportError:
    ResNet18_Weights = None

class LaneSegmentationModel(nn.Module):
    def __init__(self, n_classes=1, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            if ResNet18_Weights is None:
                self.encoder = models.resnet18(pretrained=pretrained)
            else:
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                self.encoder = models.resnet18(weights=weights)
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



# criterion = nn.BCEWithLogitsLoss()  # single-channel mask
# optimizer = optim.Adam(model.parameters(), lr=1e-3)


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


def deeplab_predict(model, device,  frame):
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


# TODO: No masks touching ball point?
def sam_predict(frame, lane_model, found_ball_point):
    h, w = frame.shape[:2]
    x, y = map(int, found_ball_point)

    if x < 0 or x >= w or y < 0 or y >= h:
        return np.zeros((h, w), dtype=np.uint8)

    with torch.no_grad():
        lane_model.set_image(frame)
        lane_results = lane_model(text=["bowling lane"])

    result = lane_results[0]
    if result.masks is None or result.masks.data is None:
        print("[SAM3 lane] no masks")
        return np.zeros((h, w), dtype=np.uint8)

    masks = result.masks.data  # [N, mask_h, mask_w]
    print(f"[SAM3 lane] masks={len(masks)} point=({x}, {y})")
    kept_mask = None

    for mask_tensor in masks:
        mask = mask_tensor.detach().cpu().numpy().astype(np.uint8)

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if mask[y, x] > 0:
            kept_mask = mask
            break

    if kept_mask is None:
        print("[SAM3 lane] no mask touched ball point")
        return np.zeros((h, w), dtype=np.uint8)

    return (kept_mask * 255).astype(np.uint8)
