import cv2
import numpy as np
from utils.config import LANE_W, LANE_H, DEBUG_BIRDS_EYE

class BaseTransformer:

    def __init__(self, out_size=(LANE_W, LANE_H), debug=DEBUG_BIRDS_EYE):
        self.out_size = out_size
        self.debug = debug

        self.avg_left_line = None
        self.avg_right_line = None

    def _ensure_grayscale(self, mask: np.ndarray):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask
    
    def _white_mask(self, mask: np.ndarray, thresh=127):
        """ Works on grayscale images"""
        mask = self._ensure_grayscale(mask)
        mask[mask > thresh] = 255  # mark all selected pixels white TODO: (wont work for light colored balls so need to switch to ML based)
        # mask = (mask > 127).astype(np.uint8) * 255 # full blakc and white 
        return mask
    
    def _nonzero_coords(self, mask):
        mask = self._ensure_grayscale(mask)
        ys, xs = np.where(mask > 127)
        return ys, xs
    

    # --- Find Lane Left / Right Boundaries For Stabilizaiton (All Downstream Transforms Utilize This) ---

    def _average_lines(self, lines, frame_size):
        """
        Compute an averaged line and its angle from a set of detected lines.

        Args:
            lines (list or np.ndarray): Collection of lines in the form (x1, y1, x2, y2).
            frame_height (int): Height of the image frame, used to anchor the averaged line.

        Returns:
            tuple: ((x1, y1, x2, y2), angle_radians) or None if no valid lines.
        """
        slopes = []
        intercepts = []
        for x1, y1, x2, y2 in lines:
            if x2 != x1:  # avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                slopes.append(slope)
                intercepts.append(intercept)
        if not slopes:
            msg = "[BaseTransformer._average_lines] No valid slopes/lines found"
            print(msg)
            # raise RuntimeError(msg)  # no valid lines
            return None

        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)

        # Pick two y-values to define the averaged line
        y1 = frame_size  # bottom of frame
        y2 = 0  # some height up

        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)

        # compute the resulting line's angle
        angle = np.arctan2(x2 - x1, y2 - y1)

        return (x1, y1, x2, y2), angle

    #TODO: implement pitch correction
    def _stabilize_rotation(self, mask: np.ndarray, vis_debug=None):

        """
        Stabilize the lane mask by detecting its outer lane lines,
        estimating the average tilt angle, and rotating the mask to align it vertically.

        Args:
            mask (np.ndarray): Binary or grayscale lane mask.
            vis_debug (np.ndarray, optional): Optional image for visualization.

        Returns:
            np.ndarray: Rotated (stabilized) mask, or None if no valid lines are found.
        """

        mask = self._ensure_grayscale(mask)
        mask = mask.astype(np.uint8)

        ys, xs = self._nonzero_coords(mask)

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
            msg = "[BaseTransformer._stabilize_rotation] HoughLinesP returned no lines"
            print(msg)
            raise RuntimeError(msg)

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
            
        # get averaged lines

        right_result = self._average_lines(right_lines, mask.shape[0])
        left_result = self._average_lines(left_lines, mask.shape[0])

        # if we cannot compute an average line on either side,
        # treat this as a hard error for stabilization
        if right_result is None or left_result is None:
            msg = "[BaseTransformer._stabilize_rotation] Could not compute average left/right lines"
            print(msg)
            raise RuntimeError(msg)

        avg_right, right_angle = right_result
        avg_left, left_angle = left_result

        # # store in the instance
        # self.avg_left_line = avg_left
        # self.avg_right_line = avg_right

        avg_angle = (left_angle + right_angle) / 2 # tells how much the whole lane has tilted

        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        rotation_gain = 1 #1.35 # gain constant for rotation

        M = cv2.getRotationMatrix2D(center, -np.degrees(avg_angle) * rotation_gain, 1.0)

        stabilized = cv2.warpAffine(mask, M, (w, h))

        # helper: rotate a line by the same affine matrix
        def _rotate_line(line, M):
            x1, y1, x2, y2 = line
            p1 = M @ np.array([x1, y1, 1.0])
            p2 = M @ np.array([x2, y2, 1.0])
            return int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])

        rot_left  = _rotate_line(avg_left,  M)
        rot_right = _rotate_line(avg_right, M)

        # store in the instance
        self.avg_left_line = rot_left
        self.avg_right_line = rot_right

        if vis_debug is not None:
            print("LANE TILT AMOUNT: ", avg_angle)
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
            # cv2.imshow("Stabalizing Debug ", vis_debug) # optionally show it seperately

        return stabilized

    def transform(self, data):
        raise NotImplementedError
