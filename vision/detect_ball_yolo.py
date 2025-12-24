from ultralytics import YOLO
import cv2
import csv
import os
from vision.trajectory import Trajectory

ball_model = YOLO(f"data/weights/best_ball.pt")

def find_ball(frame, display):

    ball_results = ball_model(frame, conf=0.4, imgsz=640) # run inference
            
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

def draw_path(ball_cx, ball_cy, trajectory, display, write_path=None):

    cv2.circle(display, (ball_cx, ball_cy), 3, (0, 0, 255), -1)
    trajectory.push((ball_cx, ball_cy)) # appends new points into the buffer

    pts = trajectory.all()
    # Draw all the previous points
    for i in range(1, len(pts)):
        cv2.line(display, pts[i-1], pts[i], (0, 0, 255), 5)

    if write_path:
        with open(write_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ball_cx, ball_cy])
            f.flush()
            os.fsync(f.fileno())
