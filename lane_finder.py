import numpy as np
import cv2
from sklearn.cluster import KMeans

# global variables for line tracking
prev_left = None
prev_right = None
alpha = 0.8  # smoothing factor (higher = smoother)
        
def remove_line_outliers(lines, threshold=2.0):
    lines = np.array(lines)
    if len(lines) == 0:
        return lines

    mean = np.mean(lines, axis=0)
    dists = np.linalg.norm(lines - mean, axis=1) # Compute Euclidean distance from each line to the mean line
    std = np.std(dists) # Compute standard deviation of those distances
    limit = np.mean(dists) + threshold * std # Set distance cutoff: lines farther than this are outliers

    return lines[dists < limit]

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

            if x2 == x1: # skip if horizontal
                continue

            slope = (y2 - y1) / (x2 - x1)   # Slope of the line
            if abs(slope) < 2:
                continue

            # Draw good lines
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))

                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), 15) # draw white line
            else:
                right_lines.append((x1, y1, x2, y2))

                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 15)  # draw white line
            
        # average of left lines
        if left_lines:
            # left_lines = remove_line_outliers(left_lines)
            x1s = [x1 for x1, y1, x2, y2 in left_lines]
            y1s = [y1 for x1, y1, x2, y2 in left_lines]
            x2s = [x2 for x1, y1, x2, y2 in left_lines]
            y2s = [y2 for x1, y1, x2, y2 in left_lines]

            avg_left = (
                int(np.mean(x1s)),
                int(np.mean(y1s)),
                int(np.mean(x2s)),
                int(np.mean(y2s))
            )

            cv2.line(blank_image, (avg_left[0], avg_left[1]), (avg_left[2], avg_left[3]), (255, 255, 255), 15)

            # if avg_left:
            #     # Smooth left line
            #     if prev_left is None:
            #         smoothed_left = np.array(avg_left, dtype=np.float32)
            #     else:
            #         smoothed_left = 0.9 * prev_left + 0.1 * np.array(avg_left, dtype=np.float32)

            #     prev_left = smoothed_left  # store smoothed value for next frame

            #     # Draw smoothed line
            #     x1, y1, x2, y2 = smoothed_left.astype(int)
            #     cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 15)

        
        # average of right lines
        if right_lines:
            x1s = [x1 for x1, y1, x2, y2 in right_lines]
            y1s = [y1 for x1, y1, x2, y2 in right_lines]
            x2s = [x2 for x1, y1, x2, y2 in right_lines]
            y2s = [y2 for x1, y1, x2, y2 in right_lines]

            avg_right = (
                int(np.mean(x1s)),
                int(np.mean(y1s)),
                int(np.mean(x2s)),
                int(np.mean(y2s))
            )

        cv2.line(blank_image, (avg_right[0], avg_right[1]), (avg_right[2], avg_right[3]), (255, 255, 255), 15)


    combined = cv2.addWeighted(img, 1.0, blank_image, 1.0, 0)
    return combined