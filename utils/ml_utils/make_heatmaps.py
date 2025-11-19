import os
import json
import numpy as np
import cv2

# directories
IMG_DIR = "data/images2"
ANN_DIR = "data/arrow_data/arrow_annotations"
OUT_IMG_DIR = "data/arrow_data/images"
OUT_HM_DIR = "data/arrow_data/heatmaps"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_HM_DIR, exist_ok=True)

# scaling factor for ALL images
SCALE = 0.25       # training resolution scaling
SIGMA = 6          # gaussian size


def gaussian_heatmap(h, w, cx, cy, sigma):
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))


for file in os.listdir(ANN_DIR):
    if not file.endswith(".json"):
        continue

    # load annotation json
    json_path = os.path.join(ANN_DIR, file)
    with open(json_path, "r") as f:
        data = json.load(f)

    img_filename = os.path.basename(data["imagePath"]) # the image the annotation came from
    img_path = os.path.join(IMG_DIR, img_filename)

    # not all annotations will have a matching image
    if not os.path.exists(img_path):
        print("Missing image:", img_path) 
        continue

    # load original image
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # compute scaled dimensions
    new_w = int(W * SCALE)
    new_h = int(H * SCALE)

    # resize image
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT_IMG_DIR, img_filename), img_resized)
    print("Saved:", img_filename)

    # allocate heatmaps
    hm_left   = np.zeros((new_h, new_w), dtype=np.float32)
    hm_center = np.zeros((new_h, new_w), dtype=np.float32)
    hm_right  = np.zeros((new_h, new_w), dtype=np.float32)

    # create heatmaps
    for shape in data["shapes"]:
        label = shape["label"]
        x, y = shape["points"][0]

        # scale annotation coordinates
        sx = x * SCALE
        sy = y * SCALE

        if label == "left":
            hm_left += gaussian_heatmap(new_h, new_w, sx, sy, SIGMA)
        elif label == "middle":
            hm_center += gaussian_heatmap(new_h, new_w, sx, sy, SIGMA)
        elif label == "right":
            hm_right += gaussian_heatmap(new_h, new_w, sx, sy, SIGMA)

    # stack into (H,W,3)
    heatmap = np.stack([hm_left, hm_center, hm_right], axis=-1)

    # normalize per-channel
    heatmap /= heatmap.max(axis=(0,1), keepdims=True) + 1e-8

    # save
    out_name = file.replace(".json", ".npy")
    np.save(os.path.join(OUT_HM_DIR, out_name), heatmap)

    print("Saved:", out_name)