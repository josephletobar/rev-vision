import cv2
import os
import sys
from pathlib import Path

# Extract frames from a given video for training

def extract_frames(video_path, output_dir, step=30):

    frame_idx = 0
    saved = 0
    for file in os.listdir(video_path):
        
        full_path = f"{video_path}/{file}"
        if not full_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        print(f"Loading {full_path}")

        cap = cv2.VideoCapture(full_path)
        os.makedirs(output_dir, exist_ok=True)
        
        while cap.isOpened():
            ret, frame = cap.read() 
            if not ret:
                break
            if frame_idx % step == 0:
                filename = f"frame_{saved:04d}.png"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                saved += 1
            frame_idx += 1

        cap.release()

ROOT = Path().resolve().parent
extract_frames(f"data/raw_frames/end_vids", "data/extracted_frames", step=2)