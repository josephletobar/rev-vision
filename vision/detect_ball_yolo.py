from ultralytics import YOLO
import cv2
import csv
import os
from vision.trajectory import Trajectory
import torch
import config

import logging
if not config.DEBUG_PIPELINE:
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
ball_model = YOLO(config.BALL_MODEL).to(device)

if torch.cuda.is_available():
    print(f"[YOLO] CUDA active: {torch.cuda.get_device_name(0)}")
else:
    print("[YOLO] CPU only")


class ExponentialMovingAvg():
    def __init__(self, alpha=0.7):
        self.prev_point = None
        self.alpha = alpha
    def update(self, curr_point: tuple):
        if self.prev_point is None:
            self.prev_point = curr_point
            return curr_point
        
        smoothed_x = (self.alpha * self.prev_point[0]) + (1-self.alpha)*curr_point[0]
        smoothed_y = (self.alpha * self.prev_point[1]) + (1-self.alpha)*curr_point[1]
        smoothed_point = (int(smoothed_x), int(smoothed_y))

        self.prev_point = smoothed_point

        return smoothed_point

def find_ball(frame, display):

    with torch.no_grad(): ball_results = ball_model(frame, conf=0.4, imgsz=640) # run inference
            
    if ball_results[0].boxes is None or len(ball_results[0].boxes) == 0:

        return None
        
    ball_x1, ball_y1, ball_x2, ball_y2 = map(int, ball_results[0].boxes.xyxy[0].tolist())
    ball_cx = (ball_x1 + ball_x2) // 2
    ball_cy = (ball_y1 + ball_y2) // 2
    ball_radius = (ball_x2 - ball_x1) // 2

    cv2.circle(
        display,
        (ball_cx, ball_cy),
        ball_radius,
        (255, 0, 0),
        7
    )

    return int(ball_cx), int(ball_cy)

ema = ExponentialMovingAvg()

def draw_path_smooth(ball_cx, ball_cy, trajectory, display):

    midpoint = (ball_cx, ball_cy)
    cv2.circle(display, midpoint, 3, (0, 0, 255), -1) # draw a circle around the midpoint

    smoothed_point = ema.update(midpoint)

    trajectory.push(smoothed_point) # appends new points into the buffer

    pts = trajectory.all()
    # Draw all the previous points
    for i in range(1, len(pts)):
        cv2.line(display, pts[i-1], pts[i], (0, 0, 255), 5)
    
    return smoothed_point

def save_points_csv(write_dir, ball_cx, ball_cy, t_sec):
    write_path = f"{write_dir}/points.csv"
    file_exists = os.path.exists(write_path)

    with open(write_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["x", "y", "time_stamp"])

        writer.writerow([ball_cx, ball_cy, t_sec])
        f.flush()
        os.fsync(f.fileno())
