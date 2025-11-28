import torch
import torch.nn as nn
from torchvision import models


class ArrowRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # replace final FC (512 → 6)
        self.backbone.fc = nn.Linear(512, 6)

    def forward(self, x):
        return self.backbone(x)