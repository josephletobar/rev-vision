import argparse
import matplotlib.pylab as plt
import cv2
import csv
import numpy as np
import subprocess
import socket
import json
from models.lane_segmentation.deeplab_predict import deeplab_predict
from vision.detect_ball_yolo import find_ball, draw_path_smooth, save_points_csv
from vision.geometric_validation import validate
from vision.mask_processing import OverlayProcessor, ExtractProcessor, extraction_validator, extend_mask_up
from vision.transformers.perspective_transformer import BirdsEyeTransformer
from archive.geometric_helper import GeometricTransformer
from vision.lane_visual import post_visual
from vision.trajectory import Trajectory
from config import DEBUG_PIPELINE, STEP, VIDEO_FPS

def create_display(name, display, out=False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
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

    # set socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 5000))
    sock.listen(1)
    conn, _ = sock.accept()

    # set CSV at the start of each run
    TRACKING_OUTPUT = "outputs/points.csv"
    with open(TRACKING_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "time_stamp"])  # header

    # parse arguments
    parser = argparse.ArgumentParser(description="Lane Assist Video Processing")
    parser.add_argument("--input", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, help="Path to save outputs (optional)")
    args = parser.parse_args()


    # video processing
    cap = cv2.VideoCapture(args.input)

    # set arguments
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(f"{args.output}/replay.mp4", fourcc, fps, (width, height))

    try:
        frame_idx = 0
        while(cap.isOpened()):
            # read the frame
            ret, frame = cap.read()
            if not ret: break
            if frame is None: continue

            frame_idx += 1
            if frame_idx != 1 and frame_idx % STEP != 0: continue

            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            display = frame.copy()
                    
            # predict lane
            _, pred_mask = deeplab_predict(frame.copy()) 
            if pred_mask is None: continue
            pred_mask = extend_mask_up(pred_mask.copy(), px=4) # extend for better visibility
            display = overlay.apply(pred_mask.copy(), display) # overlay mask on frame

            # extended_mask = extend_mask_up(pred_mask.copy(), px=50) # for later processing, visible pins
            # create_display("Extended Mask", extended_mask)

            # extract the lane
            extraction = extract.apply(pred_mask.copy(), frame.copy()) 
            if extraction is None: continue
            if not validate(extraction.copy()): continue # validate the mask  
            # create_display("Lane Extraction", extraction.copy())

            # find the ball
            found_ball_point = find_ball(extraction.copy(), display)
            if not found_ball_point: 
                create_display("Lane Display", display, out=out) # show only segmented lane
                continue # no need for further processing

            # display segmented lane and ball
            create_display("Lane Display", display, out=out)
            # see birds-eye view
            full_warp, M_full = perspective.transform(frame.copy(), extraction.copy(), alpha=1.5) 
            if full_warp is None or M_full is None: continue
            create_display("Birds Eye View", full_warp)

            # convert detected point to perspective transformed point
            pt = np.array(found_ball_point, dtype=np.float32).reshape(1, 1, 2)
            pt_warped = cv2.perspectiveTransform(pt, M_full)
            x_w, y_w = pt_warped[0, 0]
            y_w = y_w-100 # constant to increase y

            x_smooth, y_smooth = draw_path_smooth(int(x_w), int(y_w), ball_trajectory, full_warp)  

            save_points_csv("outputs", int(x_smooth), int(y_smooth), t_sec) 
            save_points_csv(args.output, int(x_smooth), int(y_smooth), t_sec)
            
            # send the data 
            msg = [float(x_smooth), float(y_smooth)]
            try:
                conn.sendall((json.dumps(msg) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Cleaning up gracefully...")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        try:
            conn.close()
            sock.close()
        except Exception:
            pass

        print("Done")

# python3 main.py --input test_videos/bowling.mp4
if __name__ == "__main__":
    main()
