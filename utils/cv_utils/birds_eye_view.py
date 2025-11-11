import cv2
import numpy as np
from utils.utils import LANE_W, LANE_H

class BirdsEyeTransformer:
    def __init__(self, out_size=(LANE_W, LANE_H)):
        self.out_size = out_size

    def _prepare_mask(self, mask: np.ndarray):
        """Ensure mask is grayscale and return its nonzero pixel coordinates."""
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # return pixel coords where the mask pixel value is > 0
        ys, xs = np.where(mask > 127)
        return mask, ys, xs
    
    def _average_lines(self, lines, frame_size):
        slopes = []
        intercepts = []
        for x1, y1, x2, y2 in lines:
            if x2 != x1:  # avoid divis_debugion by zero
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                slopes.append(slope)
                intercepts.append(intercept)
        if not slopes:
            return None  # no valid lines

        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)

        # Pick two y-values to define the averaged line
        y1 = frame_size  # bottom of frame
        y2 = 0  # some height up

        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)

        return (x1, y1, x2, y2)

    def _stabilize_rotation(self, mask: np.ndarray, vis_debug=None):

        mask, ys, xs = self._prepare_mask(mask)
        if len(xs) == 0:
            return None

        # find lane outer lines for stabilization
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=100,
            maxLineGap=10
        )
        if lines is None:
            return None

        try: 
            cx = int(xs.mean())
            cy = int(ys.mean())
            mask_center = (cx, cy)

            left_lines = []
            right_lines = []

            # detect a right / left line
            for x1, y1, x2, y2 in lines[:, 0]:
                if x2 == x1:  # avoid vertical lines
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                if abs(slope) < 0.6:
                    continue  # skip horizontal or almost flat lines

                if slope > 0:  
                    right_lines.append((x1, y1, x2, y2))
                else:
                    left_lines.append((x1, y1, x2, y2))
                
            # draw averaged lines
            avg_right = self._average_lines(right_lines, mask.shape[0])
            avg_left = self._average_lines(left_lines, mask.shape[0])

            if vis_debug is not None:
                cv2.circle(vis_debug, mask_center, 10, (255,0,0), 10)

                # all detected left lines
                if len(left_lines) > 0:
                    for x1, y1, x2, y2 in left_lines:
                        cv2.line(vis_debug, (x1, y1), (x2, y2), (255, 0, 0), 10)
                # averaged left line
                if avg_left is not None:
                    cv2.line(vis_debug, avg_left[:2], avg_left[2:], (180, 180, 180), 4) 

                # all detected right lines
                if len(right_lines) > 0:
                    for x1, y1, x2, y2 in right_lines:
                        cv2.line(vis_debug, (x1, y1), (x2, y2), (0, 0, 255), 10)
                # averaged right line
                if avg_right is not None:
                   cv2.line(vis_debug, avg_right[:2], avg_right[2:], (180, 180, 180), 4) 


        except Exception as e:
            print(e)

    def _get_mask_corners(self, mask: np.ndarray, vis_debug=None):

        mask, ys, xs = self._prepare_mask(mask)
        if len(xs) == 0:
            return None
        
        offset = int(0.05 * (ys.max() - ys.min())) # offset to go x% up/down into the mask to ignore curve

        # top row of the mask
        ytop = ys.min() + offset
        xs_top = xs[ys == ytop]
        TL = (xs_top.min(), ytop)
        TR = (xs_top.max(), ytop)

        # bottom row of the mask
        ybot = ys.max() - offset*5
        xs_bot = xs[ys == ybot]
        BL = (xs_bot.min(), ybot)
        BR = (xs_bot.max(), ybot)

        # shows the found corner points
        if vis_debug is not None:
            vis_debug[ys, xs] = (255,255,255) # mark all selected pixels white
            # Print corners
            print(f"TL: ({int(TL[0])}, {int(TL[1])}), "
            f"TR: ({int(TR[0])}, {int(TR[1])}), "
            f"BL: ({int(BL[0])}, {int(BL[1])}), "
            f"BR: ({int(BR[0])}, {int(BR[1])})")
            # Show corners
            cv2.circle(vis_debug, TL, 17, (0,0,255), -1) # Red
            cv2.circle(vis_debug, TR, 17, (0,255,0), -1) # Green
            cv2.line(vis_debug, TL, TR, (255, 0, 0), 3) # Connect the top
            cv2.circle(vis_debug, BL, 17, (255,0,0), -1) # Blue
            cv2.circle(vis_debug, BR, 17, (0,255,255), -1) # Yellow
            cv2.line(vis_debug, BL, BR, (255, 0, 0), 3) # Connect the bottom


        return TL, TR, BR, BL

    def warp(self, frame, mask, alpha=1):
        """alpha=1 keeps the full warp; smaller values relax the top edge toward its midpoint."""

        DEBUG = True # set True / False as needed
        if DEBUG:
            vis_debug = mask.copy()

        stabilized = self._stabilize_rotation(mask, vis_debug if DEBUG else None)

        corners = self._get_mask_corners(mask, vis_debug if DEBUG else None)
        if corners is None:
            return None

        src = np.float32(corners)  # TL, TR, BR, BL
        Wout, Hout = self.out_size

        dst = np.float32([
            [0, 0],
            [Wout - 1, 0],
            [Wout - 1, Hout - 1],
            [0, Hout - 1],
        ])

        mid_x = (dst[0, 0] + dst[1, 0]) / 2.0
        dst[0, 0] = mid_x - alpha * (mid_x - dst[0, 0])
        dst[1, 0] = mid_x + alpha * (dst[1, 0] - mid_x)

        M = cv2.getPerspectiveTransform(src, dst)

        if DEBUG:
            cv2.imshow("Debug Visual", vis_debug)

        return cv2.warpPerspective(frame, M, (Wout, Hout))


