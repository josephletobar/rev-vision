import argparse
import matplotlib.pylab as plt
import cv2
import csv
import numpy as np
import subprocess
from utils.ml_utils.deeplab_predict import deeplab_predict
from utils.cv_utils.mask_processing import OverlayProcessor, ExtractProcessor
from utils.cv_utils.transformers.perspective_transformer import BirdsEyeTransformer
from utils.cv_utils.ball_detection import detect_ball
from utils.cv_utils.lane_visual import visual
from utils.cv_utils.trajectory import Trajectory
from utils.config import DEBUG_PIPELINE

def main():
    overlay = OverlayProcessor()
    extract = ExtractProcessor()
    perspective = BirdsEyeTransformer()
    filter = Trajectory(buffer_size=5, threshold=120)

    # set CSV at the start of each run
    TRACKING_OUTPUT = "outputs/points.csv"
    with open(TRACKING_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])  # header

    # Parse arguments
    parser = argparse.ArgumentParser(description="Lane Assist Video Processing")
    parser.add_argument("--input", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, help="Path to save output video (optional)")
    args = parser.parse_args()

    # Load weights
    weights = "utils/ml_utils/weights/lane_deeplab_model_2.pth"

    # For video processing
    cap = cv2.VideoCapture(args.input)

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
                # print("Frame read failed — skipping")
                # continue
                break

            if frame is None:
                print(f"[main] None frame read from capture in module {__name__}")
                return
            
            # Run model on current frame to get its prediction mask
            _, pred_mask = deeplab_predict(frame, weights) 
            preview = overlay.apply(pred_mask, frame) 

            extraction = extract.apply(pred_mask, frame) # extract the mask from the frame
            if extraction is not None:
                detect_ball(extraction, preview) # detect extraction on the extraction

                try: 
                    if frame is None or extraction is None:
                        print(f"[main] None frame or extraction before perspective transform in module {__name__}")
                        return

                    warp = perspective.transform(frame, extraction, alpha=0.3) # get a perspective transform
                    if warp is None:
                        if DEBUG_PIPELINE: print("Skipping frame: no valid lane mask")
                        continue

                    H, W = warp.shape[:2]  # Height and width in pixels
                    if DEBUG_PIPELINE: print((H, W))
                    detect_ball(warp, warp, track=True, output_path=TRACKING_OUTPUT, trajectory_filter=filter) # detect ball on the warp, track this one
                except RuntimeError as e:
                    print(e)
                    continue

            cv2.imshow("Lane Overlay", preview)
            if out:
                out.write(preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit on 'q' key

            if DEBUG_PIPELINE:
                try:
                    # detect_ball(warp, warp)
                    cv2.imshow("Test", warp)
                    # if out:
                    #     out.write(warp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break  # Exit on 'q' key
                except RuntimeError as e:
                    print(e)
    
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Cleaning up gracefully...")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    # Run plotting script
    if not DEBUG_PIPELINE: # just show it if not debugging
        visual(TRACKING_OUTPUT)

# python3 main.py --input test_videos/bowling.mp4
if __name__ == "__main__":
    main()
