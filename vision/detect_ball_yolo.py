from ultralytics import YOLO
import cv2
import csv
import os
from vision.trajectory import Trajectory
import torch
from utils import config

import logging
if not config.DEBUG_PIPELINE:
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

ball_model = YOLO(f"data/weights/best_ball.pt")


class ExponentialMovingAvg():
    def __init__(self, alpha=0.75):
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

def draw_path(ball_cx, ball_cy, trajectory, display, t_sec, write_path=None):

    midpoint = (ball_cx, ball_cy)
    cv2.circle(display, midpoint, 3, (0, 0, 255), -1) # draw a circle around the midpoint

    smoothed_point = ema.update(midpoint)

    trajectory.push(smoothed_point) # appends new points into the buffer

    pts = trajectory.all()
    # Draw all the previous points
    for i in range(1, len(pts)):
        cv2.line(display, pts[i-1], pts[i], (0, 0, 255), 5)

    if write_path:
        with open(write_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ball_cx, ball_cy-25, t_sec]) # - constant to y position to adjust for bottom of lane cut off (temporary)
            f.flush()
            os.fsync(f.fileno())
