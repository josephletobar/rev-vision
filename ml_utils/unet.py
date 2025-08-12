import torch
import torch.nn as nn

# A block of two convolutional layers with ReLU activations,
# used to extract features while keeping spatial dimensions the same.
# ReLU (Rectified Linear Unit) sets all negative values to zero,
# helping the model learn non-linear patterns efficiently.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)
    
# U-Net architecture for image segmentation.
# Consists of a downsampling path (encoder), a bottleneck, and an upsampling path (decoder).
# - The encoder extracts features and reduces spatial resolution.
# - The bottleneck captures high-level features.
# - The decoder upsamples and merges features from the encoder (skip connections) to reconstruct fine details.
# The final output is a 1-channel mask with values in [0, 1], suitable for binary segmentation.   
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Downsampling
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = DoubleConv(256, 512)

        # Upsampling
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DoubleConv(256, 128)
        self.up0 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec0 = DoubleConv(128, 64)

        # Final output mask
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Down
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))

        # Up
        d2 = self.dec2(torch.cat([self.up2(m), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1))

        return torch.sigmoid(self.out(d0))  # 0–1 mask