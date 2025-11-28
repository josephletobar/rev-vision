import torch
import torch.nn as nn


# basic conv block
class Conv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ArrowRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- shallow encoder ----
        self.c1 = Conv(3, 32)
        self.c2 = Conv(32, 32)
        self.pool1 = nn.MaxPool2d(2)        # NEW: gives bigger receptive field

        self.c3 = Conv(32, 64)
        self.c4 = Conv(64, 64)
        self.pool2 = nn.MaxPool2d(2)        # NEW: now receptive field is large enough

        # ---- shallow decoder ----
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c5  = Conv(32, 32)

        self.up2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.c6  = Conv(32, 32)

        # final heatmaps
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.c2(x1)

        x2 = self.pool1(x1)
        x2 = self.c3(x2)
        x2 = self.c4(x2)

        x3 = self.pool2(x2)

        u1 = self.up1(x3)
        u1 = self.c5(u1)

        u2 = self.up2(u1)
        u2 = self.c6(u2)

        out = self.out(u2)
        return out


# sanity test
if __name__ == "__main__":
    model = ArrowRegressor()
    dummy = torch.randn(1, 3, 640, 384)
    out = model(dummy)
    print("Output:", out.shape)