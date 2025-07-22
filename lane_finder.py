import numpy as np
import cv2

# global variables for line tracking
prev_left = None
prev_right = None

def draw_lines(img):

    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Edge detection
    canny_image = cv2.Canny(hsv, 50, 100)

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
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 15) # draw white line

            else:
                right_lines.append((x1, y1, x2, y2))
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 255, 255), 15)  # draw white line

        # # === Select best pair ===
        # min_gap = float('inf')
        # best_pair = None

        # for red in left_lines:
        #     for blue in right_lines:
        #         rx = (red[0] + red[2]) / 2
        #         bx = (blue[0] + blue[2]) / 2
        #         gap = abs(rx - bx)

        #         if gap < min_gap:
        #             min_gap = gap
        #             best_pair = (red, blue)

        # # === Draw best pair ===
        # if best_pair:
        #     lx1, ly1, lx2, ly2 = best_pair[0]
        #     rx1, ry1, rx2, ry2 = best_pair[1]

        #     curr_left = np.array(best_pair[0], dtype=np.float32)
        #     curr_right = np.array(best_pair[1], dtype=np.float32)

        #     # Apply right smoothing
        #     global prev_right
        #     if prev_right is None: # If no previous, set the smoothed to the current
        #         final_right = curr_right
        #     else: # if there is a previous
        #         if np.linalg.norm(prev_right - curr_right) > 300: # make sure there isnt a big jump
        #             # big difference, discard
        #             final_right = prev_right
        #         else:
        #             # normal difference, use
        #             final_right = 0.9 * prev_right + 0.1 * curr_right

        #     prev_right = final_right  # always update with final resul

        #     rx1, ry1, rx2, ry2 = final_right

        #     # Apply left smoothing
        #     global prev_left
        #     if prev_left is None: # If no previous, set the smoothed to the current
        #         final_left = curr_left
        #     else: # if there is a previous
        #         if np.linalg.norm(prev_left - curr_left) > 300: # make sure there isnt a big jump
        #             # big difference, discard
        #             final_left = prev_left
        #         else:
        #             # normal difference, use
        #             final_left = 0.9 * prev_left + 0.1 * curr_left

        #     prev_left = final_left  # always update with final result

        #     lx1, ly1, lx2, ly2 = final_left

        #     # Draw the lines
        #     cv2.line(blank_image, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 0, 255), 15)  # red
        #     cv2.line(blank_image, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 0, 0), 15)  # blue

    combined = cv2.addWeighted(img, 1.0, blank_image, 1.0, 0)
    return combined