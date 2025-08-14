import cv2
import numpy as np
from ml_utils.deeplab_predict import deeplab_predict

# Validate and normalize mask for further processing
class BaseMaskProcessor:
    def __init__(self, min_fraction=0.01, bin_thresh=0.5, resize_to_frame=True):
        self.min_fraction = min_fraction
        self.bin_thresh = bin_thresh
        self.resize_to_frame = resize_to_frame

    def _prep_mask(self, mask, frame):
        # resize to frame size if it isnt already
        if self.resize_to_frame and (mask.shape[:2] != frame.shape[:2]):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        # size threshold
        if (np.count_nonzero(mask) / mask.size) < self.min_fraction:
            return None  # dont count small masks
        # binarize for OpenCV use: {0,255} uint8 
        m = ((mask > self.bin_thresh).astype(np.uint8) * 255)
        return m

class OverlayProcessor(BaseMaskProcessor):
    def __init__(self, blur_kernel=41, thresh=80, alpha=0.5, **kw):
        super().__init__(**kw)
        self.blur_kernel = blur_kernel
        self.thresh = thresh
        self.alpha = alpha

    def apply(self, mask, frame):
        m = self._prep_mask(mask, frame)
        if m is None:
            return frame

        # green tint mask
        mask_color = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        colored = np.zeros_like(mask_color)
        colored[:, :, 1] = mask_color[:, :, 1]

        # soften edges and binarize
        blurred = cv2.GaussianBlur(colored, (self.blur_kernel, self.blur_kernel), 0)
        _, smoothed = cv2.threshold(blurred, self.thresh, 255, cv2.THRESH_BINARY)

        # blend
        return cv2.addWeighted(frame, 1.0, smoothed, self.alpha, 0)
    
class ExtractProcessor(BaseMaskProcessor):
    def __init__(self, tight_crop=True, **kw):
        super().__init__(**kw)
        self.tight_crop = tight_crop

    def apply(self, mask, frame):
        m = self._prep_mask(mask, frame)
        if m is None:
            return None  # no valid mask this frame
        
        # Smooth lines
        # TODO: use class
        # soften edges and binarize
        blurred = cv2.GaussianBlur(m, (41, 41), 0)
        _, smoothed = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
        

        # Keep only the lane pixels from the frame
        cutout = cv2.bitwise_and(frame, frame, mask=smoothed)

        if not self.tight_crop:
            return cutout

        # Tight crop to lane bounding box
        ys, xs = np.where(m > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        return cutout[y0:y1+1, x0:x1+1]
    

