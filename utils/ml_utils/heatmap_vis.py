import numpy as np
import cv2
import os

IMG_DIR = "data/arrow_data/images"
HM_DIR  = "data/arrow_data/heatmaps"

alpha = 0.6   # heatmap opacity

for hm_file in os.listdir(HM_DIR):
    if not hm_file.endswith(".npy"):
        continue

    heatmap = np.load(os.path.join(HM_DIR, hm_file))
    img_file = hm_file.replace(".npy", ".png")
    img_path = os.path.join(IMG_DIR, img_file)

    if not os.path.exists(img_path):
        print("Missing:", img_file)
        continue

    img = cv2.imread(img_path)

    # combine left + center + right (max per pixel)
    combined = heatmap.max(axis=2)
    combined = (combined * 255).astype(np.uint8)

    # apply color map
    colored = cv2.applyColorMap(combined, cv2.COLORMAP_JET)

    # overlay
    overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)

    cv2.imshow("Heatmap Overlay", overlay)
    key = cv2.waitKey(0)

    if key == 27:   # ESC to quit
        break

cv2.destroyAllWindows()