import os
import torch
import cv2
import numpy as np
from arrow_model import ArrowRegressor

MODEL_PATH = "models/arrow_regression/weights/best_arrow_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
model = ArrowRegressor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# Peak finder
# -----------------------------
def get_peak(hm):
    # hm: (H, W)
    idx = np.argmax(hm)
    h, w = hm.shape
    y = idx // w
    x = idx % w
    return int(x), int(y)


# -----------------------------
# Inference function
# -----------------------------
def infer_arrow_points(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Image not found: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))   # CHW
    x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        out = model(x)[0].cpu().numpy()   # (3, H, W)

    left_hm, mid_hm, right_hm = out

    # find peaks
    left_pt   = get_peak(left_hm)
    middle_pt = get_peak(mid_hm)
    right_pt  = get_peak(right_hm)

    return img_bgr, (left_pt, middle_pt, right_pt), out


# -----------------------------
# Draw CLEAR points (dot + crosshair + label)
# -----------------------------
def draw_point(img, pt, color, label):
    x, y = pt
    # filled dot
    cv2.circle(img, (x, y), 6, color, -1)
    # crosshair lines
    cv2.line(img, (x - 10, y), (x + 10, y), color, 1)
    cv2.line(img, (x, y - 10), (x, y + 10), color, 1)
    # label text
    cv2.putText(img, label, (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



# -----------------------------
# Main test execution
# -----------------------------
if __name__ == "__main__":
    IMG_PATH = "data/arrow_data/images_out/frame_0004.png"

    img_bgr, (left_pt, mid_pt, right_pt), heatmaps = infer_arrow_points(IMG_PATH)

    print("Left:  ", left_pt)
    print("Mid:   ", mid_pt)
    print("Right: ", right_pt)

    # ---- CLEAN POINT VISUALIZATION ----
    vis_points = img_bgr.copy()
    draw_point(vis_points, left_pt,   (0, 0, 255),   "L")  # red
    draw_point(vis_points, mid_pt,    (0, 255, 0),   "M")  # green
    draw_point(vis_points, right_pt,  (255, 0, 0),   "R")  # blue

    cv2.imshow("arrows_pred_clean", vis_points)

    # ---- HEATMAP VISUALIZATION (same as before) ----
    names = ["left", "middle", "right"]
    for i, name in enumerate(names):
        hm = heatmaps[i]  # (H, W)

        # normalize to 0–255 for display
        hm_norm = hm - hm.min()
        hm_norm /= (hm_norm.max() + 1e-6)
        hm_vis = (hm_norm * 255).astype(np.uint8)

        hm_vis = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        cv2.imshow(f"heatmap_{name}", hm_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()