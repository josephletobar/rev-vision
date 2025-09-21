import cv2
import numpy as np
from ml_utils.deeplab_predict import deeplab_predict

def post_processing(mask, frame):
    # Mask size threshhold
    min_fraction = 0.01
    mask_fraction = np.count_nonzero(mask) / mask.size
    if mask_fraction < min_fraction:
        return frame # skip overlay if too small

    # Convert mask from single-channel float array to 8-bit 3-channel format
    mask_uint8 = (mask * 255).astype(np.uint8)  # convert from 0/1 float to 0-255 uint8
    mask_color = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels of color

    # Multiply mask_color by green tint
    colored_mask = np.zeros_like(mask_color)
    colored_mask[:, :, 1] = mask_color[:, :, 1]  # green channel only

    # Resize mask to match frame size (resize colored_mask, not mask_color)
    colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]))

    # Smooth Borders
    mask_blurred = cv2.GaussianBlur(colored_mask, (191, 191), 0) # Gaussian blur with 121x121 kernel
    _, mask_smoothed = cv2.threshold(mask_blurred, 120, 255, cv2.THRESH_BINARY) # Convert to binary mask: pixels>120 set to 255 (white), 
                                                                               # others to 0 (black) using binary thresdhold
