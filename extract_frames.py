import cv2
import os

def extract_frames(video_path, output_dir, step=30):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    saved = 0

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

extract_frames("bowling.mp4", "data/images")