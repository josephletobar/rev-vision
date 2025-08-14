import cv2
import numpy as np

def detect_ball(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    if circles is not None:
        print("found")
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    print("none found")