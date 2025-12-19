import argparse
import matplotlib.pylab as plt
import cv2
import csv
import numpy as np
import subprocess
from models.lane_segmentation.deeplab_predict import deeplab_predict
from vision.mask_processing import OverlayProcessor, ExtractProcessor
from vision.transformers.perspective_transformer import BirdsEyeTransformer
from vision.transformers.geometric_helper import GeometricTransformer
from vision.ball_detection import detect_ball
from vision.lane_visual import visual
from vision.trajectory import Trajectory
from utils.config import DEBUG_PIPELINE

def main():
    overlay = OverlayProcessor()
    extract = ExtractProcessor()
    perspective = BirdsEyeTransformer()
    geometric = GeometricTransformer()
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
    weights = "data/weights/lane_deeplab_model_2.pth" 

    # For video processing
    cap = cv2.VideoCapture(args.input)

    writer2 = None
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
                result = detect_ball(extraction, preview) # detect ball on the extraction

                cv2.imshow("Lane Overlay", preview)
                if out:
                    out.write(preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Exit on 'q' key

                # if result == False: continue
                
                try: 
                    if frame is None or extraction is None:
                        print(f"[main] None frame or extraction before perspective transform in module {__name__}")
                        return

                    full_warp, M_full = perspective.transform(frame, extraction, alpha=1.4) # get a perspective transform
                    partial_warp, M_part = perspective.transform(frame, extraction, alpha=0.3) # get a perspective transform

                    if full_warp is None or partial_warp is None:
                        if DEBUG_PIPELINE: print("Skipping frame: no valid lane mask")
                        continue
                        
                    # transform matrix relationship between full and partial
                    M_rel = M_full @ np.linalg.inv(M_part)

                    partial_warp, detections = geometric.partial_transform(partial_warp)

                    full_warp = geometric.full_transform(full_warp, M_rel, detections)

                    detect_ball(partial_warp, full_warp, track=True, output_path=TRACKING_OUTPUT, 
                                trajectory_filter=filter, M_rel=M_rel) # detect ball on the partial warp, track this one

                    # warp_copy = warp.copy()

                except RuntimeError as e:
                    print(e)
                    continue
       
            # TEST
            cv2.imshow("Partial Warp Processing", partial_warp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit on 'q' key
 

            if DEBUG_PIPELINE:
                try:
                    cv2.imshow("Debug", full_warp)
                    height, width = full_warp.shape[:2]         
                    full_warp = cv2.resize(full_warp, (width, height))
                    if writer2 is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer2 = cv2.VideoWriter(
                            "outputs/segmented_mask8.mp4",
                            fourcc,
                            30.0,
                            (width, height)
                        )
                    writer2.write(full_warp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        writer2.release()
                        cv2.destroyAllWindows()
                        break
                except:
                    pass

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Cleaning up gracefully...")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    # Run plotting script
    visual(TRACKING_OUTPUT)

# python3 main.py --input test_videos/bowling.mp4
if __name__ == "__main__":
    main()
