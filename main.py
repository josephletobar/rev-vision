import argparse
import matplotlib.pylab as plt
import cv2
import csv
import numpy as np
import subprocess
from models.lane_segmentation.deeplab_predict import deeplab_predict
from vision.detect_ball_yolo import find_ball, draw_path
from vision.mask_processing import OverlayProcessor, ExtractProcessor, extraction_validator
from vision.transformers.perspective_transformer import BirdsEyeTransformer
from vision.transformers.geometric_helper import GeometricTransformer
from vision.lane_visual import visual
from vision.trajectory import Trajectory
from utils.config import DEBUG_PIPELINE

def main():
    overlay = OverlayProcessor()
    extract = ExtractProcessor()
    perspective = BirdsEyeTransformer()
    filter = Trajectory(buffer_size=5, threshold=120)
    geometric = GeometricTransformer()

    ball_trajectory = Trajectory()

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
            display = frame

            found_ball_point = find_ball(frame, display)

            if not found_ball_point: # no need for further processing
                # cv2.imshow("Lane Display", display)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     writer2.release()
                #     cv2.destroyAllWindows()
                #     break
                continue
                    
            # Run model on current frame to get its prediction mask
            _, pred_mask = deeplab_predict(frame, weights) 
            display = overlay.apply(pred_mask, display) 

            cv2.imshow("Lane Display", display)
            if out:
                out.write(display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit on 'q' key

            extraction = extract.apply(pred_mask, frame) # extract the mask from the frame
            if extraction is not None:
            
                full_warp, M_full = perspective.transform(frame, extraction, alpha=1.3) # get a perspective transform


                if full_warp is None:
                    if DEBUG_PIPELINE: print("Skipping frame: no valid lane mask")
                    continue

                detections = geometric._lane_markers(full_warp)

                # convert detected point to perspective transformed
                pt = np.array(found_ball_point, dtype=np.float32).reshape(1, 1, 2)
                pt_warped = cv2.perspectiveTransform(pt, M_full)
                x_w, y_w = pt_warped[0, 0]

                draw_path(int(x_w), int(y_w), ball_trajectory, full_warp, "outputs/points.csv")
                                
            if DEBUG_PIPELINE:
                # TEST

                cv2.imshow("Debug", full_warp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Exit on 'q' key

                #     cv2.imshow("Debug 2", full_warp)
                #     height, width = full_warp.shape[:2]         
                #     full_warp = cv2.resize(full_warp, (width, height))
                #     if writer2 is None:
                #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                #         writer2 = cv2.VideoWriter(
                #             "outputs/segmented_mask8.mp4",
                #             fourcc,
                #             30.0,
                #             (width, height)
                #         )
                #     writer2.write(full_warp)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         writer2.release()
                #         cv2.destroyAllWindows()
                #         break
           

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Cleaning up gracefully...")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


# python3 main.py --input test_videos/bowling.mp4
if __name__ == "__main__":
    main()
