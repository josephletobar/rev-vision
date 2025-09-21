import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_deeplab_model(stock_pretrained=True, device="cpu"):
    # stock_pretrained=True -> use torchvision’s default VOC weights
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if stock_pretrained else None
    model = deeplabv3_resnet50(weights=weights, weights_backbone=None if not stock_pretrained else None)

    # Replace the final classifier conv: 21 -> 1 channel
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)

    # shrink aux head if present
    if model.aux_classifier is not None:
        aux_in = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(aux_in, 1, kernel_size=1)

    return model.to(device)