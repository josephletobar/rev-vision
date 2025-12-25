import argparse
import matplotlib.pylab as plt
import cv2
import csv
import numpy as np
import subprocess
from models.lane_segmentation.deeplab_predict import deeplab_predict
from vision.detect_ball_yolo import find_ball, draw_path
from vision.geometric_validation import validate
from vision.mask_processing import OverlayProcessor, ExtractProcessor, extraction_validator
from vision.transformers.perspective_transformer import BirdsEyeTransformer
from vision.transformers.geometric_helper import GeometricTransformer
from vision.lane_visual import post_visual
from vision.trajectory import Trajectory
from utils.config import DEBUG_PIPELINE

def create_display(name, display, out=False):
    cv2.imshow(name, display)
    if out:
        out.write(display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return  # Exit on 'q' key  

def main():

    # create instances
    overlay = OverlayProcessor()
    extract = ExtractProcessor()
    perspective = BirdsEyeTransformer()
    ball_trajectory = Trajectory()

    # set CSV at the start of each run
    TRACKING_OUTPUT = "outputs/points.csv"
    with open(TRACKING_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])  # header

    # parse arguments
    parser = argparse.ArgumentParser(description="Lane Assist Video Processing")
    parser.add_argument("--input", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, help="Path to save output video (optional)")
    args = parser.parse_args()

    # load weights
    weights = "data/weights/lane_deeplab_model_2.pth" 

    # video processing
    cap = cv2.VideoCapture(args.input)

    # set arguments
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    try:
        while(cap.isOpened()):
            # read the frame
            ret, frame = cap.read()
            if not ret: break
            if frame is None: continue
            display = frame

            # find the ball
            found_ball_point = find_ball(frame.copy(), display)
            if not found_ball_point: continue # no need for further processing
                    
            # predict lane
            _, pred_mask = deeplab_predict(frame, weights) 
            display = overlay.apply(pred_mask, display) 

            # extract the lane
            extraction = extract.apply(pred_mask, frame) 
            if extraction is None: continue
            if not validate(extraction): continue # validate the mask  

            # display segmented lane and ball
            create_display("Lane Display", display)

            # see birds-eye view
            full_warp, M_full = perspective.transform(frame, extraction, alpha=1.3) 
            if full_warp is None or M_full is None: continue

            # detections = geometric._lane_markers(full_warp)

            # convert detected point to perspective transformed point
            pt = np.array(found_ball_point, dtype=np.float32).reshape(1, 1, 2)
            pt_warped = cv2.perspectiveTransform(pt, M_full)
            x_w, y_w = pt_warped[0, 0]

            # draw path
            draw_path(int(x_w), int(y_w), ball_trajectory, full_warp, "outputs/points.csv")                    

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Cleaning up gracefully...")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Done")

# python3 main.py --input test_videos/bowling.mp4
if __name__ == "__main__":
    main()
