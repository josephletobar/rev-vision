import argparse
import matplotlib.pylab as plt
import cv2
import numpy as np
from ml_utils.unet_predict import predict
from cv_utils.mask_post_processing import post_processing

# Parse arguments
parser = argparse.ArgumentParser(description="Lane Assist Video Processing")
parser.add_argument("--video", type=str, required=True, help="Path to video file")
args = parser.parse_args()

# Load weights
weights = "road_deeplab_model2"

# For video processing
cap = cv2.VideoCapture(args.video)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame wasn't read
    
    _, pred_mask = predict(frame) # run model on current frame to get its prediction mask
    result = post_processing(pred_mask, frame) 

    cv2.imshow("Video", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key

cap.release()
cv2.destroyAllWindows()