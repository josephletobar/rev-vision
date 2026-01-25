import cv2
import numpy as np

# Expects a mask
def extraction_validator(mask):

    mask[mask > 127] = 255
    return mask

def extend_mask_up(mask, px):
    mask = (mask > 0).astype(np.uint8) * 255

    h, w = mask.shape[:2]
    out = mask.copy()
    out[0:h-px] |= mask[px:h]
    return out

def _get_lines(mask: np.ndarray):

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)

        ys, xs = np.where(mask > 127)

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
        
        return left_lines, right_lines, mask_center

def _average_lines(lines, frame_size):
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

# Validate and normalize mask for further processing
class BaseMaskProcessor:
    def __init__(self, min_fraction=0.01, bin_thresh=0.5, resize_to_frame=True,
                blur_kernel=41, post_thresh=140):
        # prep_mask
        self.min_fraction = min_fraction
        self.bin_thresh = bin_thresh
        self.resize_to_frame = resize_to_frame
        # smooth mask
        self.blur_kernel = blur_kernel      
        self.post_thresh = post_thresh     

    def _prep_mask(self, mask, frame):
        if mask is None or frame is None:
            print(f"[BaseMaskProcessor._prep_mask] None image in module {__name__}")
            return None

        # resize to frame size if it isnt already
        if self.resize_to_frame and (mask.shape[:2] != frame.shape[:2]):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        # size threshold
        if (np.count_nonzero(mask) / mask.size) < self.min_fraction:
            print(f"[BaseMaskProcessor._prep_mask] Mask fraction below min_fraction ({self.min_fraction})")
            return None  # dont count small masks
        # binarize for OpenCV use: {0,255} uint8 
        m = ((mask > self.bin_thresh).astype(np.uint8) * 255)
        return m
    
    def _smooth_mask(self, mask):
        # soften edges and binarize
        blurred = cv2.GaussianBlur(mask, (self.blur_kernel, self.blur_kernel), 0)
        _, m_smoothed = cv2.threshold(blurred, self.post_thresh, 255, cv2.THRESH_BINARY)
        return m_smoothed

class OverlayProcessor(BaseMaskProcessor):
    def __init__(self, blur_kernel=41, thresh=140, alpha=0.5, **kw):
        super().__init__(**kw)
        self.blur_kernel = blur_kernel
        self.thresh = thresh
        self.alpha = alpha

    def apply(self, mask, frame):
        m = self._prep_mask(mask, frame)
        if m is None:
            print("[OverlayProcessor.apply] Preprocessed mask is None, returning original frame")
            return frame

        # green tint mask
        mask_color = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        colored = np.zeros_like(mask_color)
        colored[:, :, 1] = mask_color[:, :, 1]

        # soften edges and binarize
        smoothed = self._smooth_mask(colored)

        # blend
        return cv2.addWeighted(frame, 1.0, smoothed, self.alpha, 0)
    
class ExtractProcessor(BaseMaskProcessor):
    def __init__(self, tight_crop=False, **kw):
        super().__init__(**kw)
        self.tight_crop = tight_crop

    def apply(self, mask, frame):
        m = self._prep_mask(mask, frame)
        if m is None:
            return None

        # strict confidence threshold
        THRESH = 120
        m = np.zeros_like(mask, dtype=np.uint8)
        m[mask >= THRESH] = 255

        if cv2.countNonZero(m) == 0:
            return None

        # lane-only image
        cutout = cv2.bitwise_and(frame, frame, mask=m)
 
        # ---- ROBUST LINE FIT (NO HOUGH) ----

        # find largest contour
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return cutout
        


        cnt = max(contours, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2)


        if len(pts) < 50:
            return cutout

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


        # visualize contour points
        for (x, y) in right_pts:
            cv2.circle(cutout, (x, y), 4, (0, 0, 255), -1)

        # visualize contour points
        for (x, y) in left_pts:
            cv2.circle(cutout, (x, y), 4, (255, 0, 0), -1)

        ys_mask, _ = np.where(m > 0)
        y_top = int(ys_mask.min())
        y_bot = int(ys_mask.max())

        # fit + draw left line
        if len(left_pts) > 20:
            vx, vy, x0, y0 = cv2.fitLine(left_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            x_top = int(x0 + (y_top - y0) * vx / vy)
            x_bot = int(x0 + (y_bot - y0) * vx / vy)
            cv2.line(cutout, (x_top, y_top), (x_bot, y_bot), (255, 255, 255), 3)

        # fit + draw right line
        if len(right_pts) > 20:
            vx, vy, x0, y0 = cv2.fitLine(right_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            x_top = int(x0 + (y_top - y0) * vx / vy)
            x_bot = int(x0 + (y_bot - y0) * vx / vy)
            cv2.line(cutout, (x_top, y_top), (x_bot, y_bot), (255, 255, 255), 3)

        return cutout



        # Tight crop to lane bounding box
        ys, xs = np.where(m > 0)
        if xs.size == 0 or ys.size == 0:
            print("[ExtractProcessor.apply] No nonzero pixels after mask preprocessing")
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        return cutout[y0:y1+1, x0:x1+1]
    
