import numpy as np
import cv2

# global variables for line tracking
prev_left = None
prev_right = None
        

def draw_lines(img):

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    canny_image = cv2.Canny(gray, 50, 100)

    # Line detection
    lines = cv2.HoughLinesP(
        canny_image,
        rho=1,                     # distance resolution in pixels
        theta=np.pi / 180,         # angular resolution in radians
        threshold=40,              # minimum number of votes
        minLineLength=100,          # minimum length of line
        maxLineGap=50            # maximum allowed gap
    )

    img = np.copy(img)
    blank_image = np.zeros(img.shape, dtype=np.uint8)

    if lines is not None:
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 2:
                continue

            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), 15) # draw white line

            else:
                right_lines.append((x1, y1, x2, y2))
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 15)  # draw white line
 

    combined = cv2.addWeighted(img, 1.0, blank_image, 1.0, 0)
    return combined