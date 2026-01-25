import torch.nn as nn
import torchvision.models as models
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import torch.optim as optim

from archive.lane_segmentation.deeplab_predict import load_model


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


while(cap.isOpened()):
    # read the frame
    ret, frame = cap.read()
    if not ret: break
    if frame is None: continue

    img_np = frame

    img_tensor = preprocess_frame(frame).to(device)

    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_mask = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()

    # --- plot ---
    # pred_mask is (H, W) float in [0,1]
    mask_vis = (pred_mask * 255).astype(np.uint8)

    # ensure mask matches frame size
    mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]))

    # colorize mask (red)
    mask_color = np.zeros_like(frame)
    mask_color[:, :, 2] = mask_vis  # red channel

    # overlay
    overlay = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)

    cv2.imshow("Lane Segmentation", overlay)



    # -- quad fitter -- 
    SCALE = 10

    H_net, W_net = pred_mask.shape
    H_orig, W_orig = frame.shape[:2]

    from quadrilateral_fitter import QuadrilateralFitter

    # pred_mask: (H, W) float or uint8
    mask = (pred_mask > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)

    polygon = cnt.squeeze()  # (N, 2)

    polygon = polygon.copy()
    polygon[:, 0] *= SCALE # scale x axis (quad fitter struggles with narrow shapes)

    fitter = QuadrilateralFitter(polygon)
    quad = np.array(fitter.fit(), dtype=np.float32)
    # fitter.plot()

    quad[:, 0] /= SCALE

    quad = quad.astype(np.int32)

    overlay = img_np.copy()

    sx = W_orig / W_net
    sy = H_orig / H_net

    quad_up = quad.astype(np.float32)
    quad_up[:, 0] *= sx
    quad_up[:, 1] *= sy
    quad_up = quad_up.astype(np.int32)

    cv2.fillPoly(frame, [quad_up], (0, 255, 0))    

    # cv2.fillPoly(overlay, [quad], (0, 255, 0))

    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)

    blended = cv2.resize(blended, None, fx=0.5, fy=0.5)
    cv2.imshow("Lane Quad", blended)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
