import cv2
import numpy as np

class AverageSmoother():
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.history = []

    def smooth(self, mask):
        self.history.append(mask.astype(np.float32))
        if len(self.history) > self.window_size:
            self.history.pop(0)
        avg = np.mean(self.history, axis=0)
        avg = (avg >= 128).astype(np.uint8) * 255  # convert back to valid OpenCV mask
        return avg  

class PostProcessor():
    def __init__(self, min_fraction:int =0.01, bin_thresh:int =0.5):
        self.min_fraction = min_fraction # for mask validation
        self.bin_thresh = bin_thresh # decide binary thershold
        self.smoother = AverageSmoother(window_size=2)

    def _extend_mask_up(self, mask, px):
        mask = (mask > 0).astype(np.uint8) * 255

        h, w = mask.shape[:2]
        out = mask.copy()
        out[0:h-px] |= mask[px:h]
        return out
    
    def _stabilize_rotation(self, mask: np.ndarray, left_angle, right_angle):
 
        avg_angle = (left_angle + right_angle) / 2 # tells how much the whole lane has tilted

        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        rotation_gain = 1 #1.35 # gain constant for rotation

        M = cv2.getRotationMatrix2D(center, -np.degrees(avg_angle) * rotation_gain, 1.0)

        stabilized = cv2.warpAffine(mask, M, (w, h))

        return stabilized, M

    def _prep_mask(self, mask, frame):
        if mask is None or frame is None:
            print(f"[BaseMaskProcessor._prep_mask] None image in module {__name__}")
            return None

        # resize to frame size if it isnt already
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        # size threshold
        if (np.count_nonzero(mask) / mask.size) < self.min_fraction:
            print(f"[BaseMaskProcessor._prep_mask] Mask fraction below min_fraction ({self.min_fraction})")
            return None  # dont count small masks
        # binarize for OpenCV use: {0,255} uint8 
        m = ((mask > self.bin_thresh).astype(np.uint8) * 255)
        return m

    def apply(self, mask, frame):
        m = self._prep_mask(mask, frame)
        if m is None:
            return None, None, None, None

        # strict confidence threshold
        THRESH = 120
        m = np.zeros_like(mask, dtype=np.uint8)
        m[mask >= THRESH] = 255

        if cv2.countNonZero(m) == 0:
            return None, None, None, None
        
        m = self._extend_mask_up(m, px=5)

        # smooth mask averaging
        m = self.smoother.smooth(m)

        # lane-only image
        cutout = cv2.bitwise_and(frame, frame, mask=m)
 
        # ---- ROBUST LINE FIT ----

        # find largest contour
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return cutout, None, None, None
    
        cnt = max(contours, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2)

        if len(pts) < 50:
            return cutout, None, None, None

        # split left / right by median x
        mid_x = np.median(pts[:, 0])
        left_pts = pts[pts[:, 0] < mid_x]
        right_pts = pts[pts[:, 0] > mid_x]

        # trim left points
        ys = left_pts[:, 1]
        y_lo = np.percentile(ys, 20)
        y_hi = np.percentile(ys, 60)
        left_pts = left_pts[(ys >= y_lo) & (ys <= y_hi)]

        # trim right points
        ys = right_pts[:, 1]
        y_lo = np.percentile(ys, 20)
        y_hi = np.percentile(ys, 60)
        right_pts = right_pts[(ys >= y_lo) & (ys <= y_hi)]

        # # visualize contour points
        # for (x, y) in right_pts:
        #     cv2.circle(cutout, (x, y), 4, (0, 0, 255), -1)
        # # visualize contour points
        # for (x, y) in left_pts:
        #     cv2.circle(cutout, (x, y), 4, (255, 0, 0), -1)

        top_offset = 2 # push top line up 
        ys, _ = np.where(m > 0)
        y_top = int(ys.min()) - top_offset
        y_bot = int(ys.max())

        left_angle = None
        right_angle = None
        eps = 1e-6

        lx_top = lx_bot = rx_top = rx_bot = None

        # fit + draw left line
        if len(left_pts) > 20:
            vx, vy, x0, y0 = cv2.fitLine(left_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            left_angle = np.arctan(vx / (vy + eps))
            lx_top = int(x0 + (y_top - y0) * vx / vy)
            lx_bot = int(x0 + (y_bot - y0) * vx / vy)
            cv2.line(cutout, (lx_top, y_top), (lx_bot, y_bot), (220, 245, 245), 1)

        # fit + draw right line
        if len(right_pts) > 20:
            vx, vy, x0, y0 = cv2.fitLine(right_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            right_angle = np.arctan(vx / (vy + eps))
            rx_top = int(x0 + (y_top - y0) * vx / vy)
            rx_bot = int(x0 + (y_bot - y0) * vx / vy)
            cv2.line(cutout, (rx_top, y_top), (rx_bot, y_bot), (220, 245, 245), 1)

        if lx_top is not None and rx_top is not None:

            lane_corners = (
                (lx_top, y_top),   # TL
                (rx_top, y_top),   # TR
                (rx_bot, y_bot),   # BR
                (lx_bot, y_bot)    # BL
            )

            lane_mask = np.zeros_like(m, dtype=np.uint8)

            poly = np.array([
                [lx_top, y_top],
                [rx_top, y_top],
                [rx_bot, y_bot],
                [lx_bot, y_bot]
            ], dtype=np.int32)

            cv2.fillPoly(lane_mask, [poly], 255)

            lane_mask_rot, R = self._stabilize_rotation(lane_mask, left_angle, right_angle)
            frame_rot, _ = self._stabilize_rotation(frame, left_angle, right_angle)

            cutout = cv2.bitwise_and(frame_rot, frame_rot, mask=lane_mask_rot)


            lane_corners_rot = []
            for (x, y) in lane_corners:
                p = np.array([x, y, 1.0])
                x_r, y_r = R @ p
                lane_corners_rot.append((int(x_r), int(y_r)))

            # for (x, y) in lane_corners_rot:
            #     cv2.circle(cutout, (int(x), int(y)), 10, (0, 0, 255), -1)

        return cutout, (left_angle, right_angle), lane_corners_rot, R
    
