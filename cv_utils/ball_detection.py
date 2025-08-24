import cv2
import numpy as np

def detect_ball(img, preview):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)

    # --- Blob detector params ---
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False  # don’t care about brightness
    params.filterByArea = True
    params.minArea = 100          # adjust to ball size
    params.maxArea = 20000

    params.filterByCircularity = True
    params.minCircularity = 0.6   # 1.0 = perfect circle

    params.filterByInertia = True
    params.minInertiaRatio = 0.3  # helps reject elongated blobs

    detector = cv2.SimpleBlobDetector_create(params)

    # --- Detect blobs ---
    keypoints = detector.detect(img_blur)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        cv2.circle(preview, (x, y), r, (255, 0, 0), 2)