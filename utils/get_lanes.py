from archive.geometric_validation import validate
from vision.lane_segmentation import deeplab_predict
import os
import numpy as np
import cv2
from vision.lane_segmentation import LaneSegmentationModel, deeplab_predict
import torch
from vision.mask_processing import PostProcessor
import config

PATH = "test_videos"
STEP = 3

# deeplab model setup    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaneSegmentationModel(n_classes=1).to(device)
model.load_state_dict(torch.load(config.LANE_MODEL, map_location=device))
model.eval()

post_process = PostProcessor()

OUTPUT_DIR = "data/extracted_lanes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():

    img_idx = 0
    for video in os.listdir(PATH):
        file_path = os.path.join(PATH, video)
        print(file_path)

        if not (video.lower().endswith(".mov") or video.lower().endswith(".mp4")):
            continue

        cap = cv2.VideoCapture(file_path)

        frame_idx = 0
        while(cap.isOpened()):
            try:
                # read the frame
                ret, frame = cap.read()

                if not ret: break
                if frame is None: continue

                frame_idx += 1
                if frame_idx != 1 and frame_idx % STEP != 0: continue
                        
                # predict lane
                pred_mask = deeplab_predict(model, device, frame.copy())
                if pred_mask is None: continue

                extraction, mask_boundaries = post_process.apply(pred_mask.copy(), frame.copy()) 
                if extraction is None or mask_boundaries is None: continue
                left_angle, right_angle = mask_boundaries
                if left_angle is None or right_angle is None: continue

                out_path = os.path.join(OUTPUT_DIR, f"{img_idx:06d}.png")
                cv2.imwrite(out_path, extraction)
                img_idx += 1
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

main()