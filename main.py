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
parser.add_argument("--output", type=str, help="Path to save output video (optional)")
args = parser.parse_args()

# Load weights
weights = "ml_utils/weights/lane_deeplab_model_2.pth"

# For video processing
cap = cv2.VideoCapture(args.video)

out = None
if args.output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

try:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break  # Exit if frame wasn't read
        
        # Run model on current frame to get its prediction mask
        _, pred_mask = deeplab_predict(frame, weights) 
        preview = overlay.apply(pred_mask, frame) 

        extraction = extract.apply(pred_mask, frame) # extract the mask from the frame
        if extraction is not None:
            detect_ball(extraction, preview) # detect extraction on the extraction

            warp = perspective.warp(frame, extraction, alpha=0.3) # get a perspective transform
            # detect_ball(warp, warp) # detect ball on the warp



        cv2.imshow("Lane Overlay", preview)
        if out:
            out.write(preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit on 'q' key

        # Test display
        cv2.imshow("Test", warp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit on 'q' key

finally:
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()