import argparse
import matplotlib.pylab as plt
import cv2
import numpy as np
from ml_utils.unet.unet_predict import unet_predict
from ml_utils.deeplab_predict import deeplab_predict
from cv_utils.mask_post_processing import OverlayProcessor

overlay = OverlayProcessor()

# Parse arguments
parser = argparse.ArgumentParser(description="Lane Assist Video Processing")
parser.add_argument("--video", type=str, required=True, help="Path to video file")
args = parser.parse_args()

# Load weights
weights = "ml_utils/weights/lane_deeplab_model.pth"

# For video processing
cap = cv2.VideoCapture(args.video)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame wasn't read
    
    _, pred_mask = deeplab_predict(frame, weights) # run model on current frame to get its prediction mask
    result = overlay.apply(pred_mask, frame) 

    cv2.imshow("Video", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key

cap.release()
cv2.destroyAllWindows()