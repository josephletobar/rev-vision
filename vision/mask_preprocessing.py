from ultralytics import YOLO
import cv2, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from quadrilateral_fitter import QuadrilateralFitter
from pyparsing import deque

class MaskWindowAvg:
    def __init__(self, N):
        self.N = N
        self.buf = deque(maxlen=N)

    def update(self, mask):
        self.buf.append(mask)
        return np.mean(self.buf, axis=0)

class MaskPreprocessor:
    def __init__(self, blend_alpha=0.4):
        self.blend_alpha = blend_alpha
        self.mask_window_avg = MaskWindowAvg(2)
        self.scale = 100.0

    def overlay_quad(self, quad, img: np.ndarray):
        
        quad = quad.astype(np.int32)
        quad = quad.reshape(-1, 1, 2)

        overlay = img.copy()
        cv2.fillPoly(overlay, [quad], (0, 255, 0))

        blended = cv2.addWeighted(overlay, self.blend_alpha, img, 1 - self.blend_alpha, 0)
        return blended

    def preprocess_mask(self, yolo_result):
        if not yolo_result.masks: return None
        polygons = yolo_result.masks.xy[0]

        polygons = polygons.copy()
        polygons[:, 0] *= self.scale # scale x axis (quad fitter struggles with narrow shapes)

        fitter = QuadrilateralFitter(polygons)
        fitted_quadrilateral = np.array(fitter.fit(), dtype=np.float32)

        fitted_quadrilateral[:, 0] /= self.scale

        fitter.plot()

        # mask = self.mask_window_avg.update(fitted_quadrilateral)

        return fitted_quadrilateral