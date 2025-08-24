import argparse
import matplotlib.pylab as plt
import cv2
import numpy as np
from ml_utils.deeplab_predict import deeplab_predict
from cv_utils.mask_processing import OverlayProcessor, ExtractProcessor
from cv_utils.birds_eye_view import BirdsEyeTransformer
from cv_utils.ball_detection import detect_ball

overlay = OverlayProcessor()
extract = ExtractProcessor()
perspective = BirdsEyeTransformer()

# Parse arguments
parser = argparse.ArgumentParser(description="Lane Assist Video Processing")
parser.add_argument("--video", type=str, required=True, help="Path to video file")
args = parser.parse_args()

# Load weights
weights = "ml_utils/weights/lane_deeplab_model_2.pth"

# For video processing
cap = cv2.VideoCapture(args.video)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame wasn't read
    
    # Run model on current frame to get its prediction mask
    _, pred_mask = deeplab_predict(frame, weights) 
    preview = overlay.apply(pred_mask, frame) 

    extraction = extract.apply(pred_mask, frame) # extract the mask from the frame
    warp = perspective.warp(frame, extraction) # get the birds eye view of the mask
    detect_ball(extraction, preview)

    cv2.imshow("Lane Overlay", preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key

    # cv2.imshow("Test", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break  # Exit on 'q' key

cap.release()
cv2.destroyAllWindows()