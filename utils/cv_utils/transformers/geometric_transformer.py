import cv2
import numpy as np
from utils.cv_utils.transformers.base_transformer import BaseTransformer
from utils.config import LANE_W, LANE_H


class GeometricTransformer(BaseTransformer):

    def transform(self, stabilized_mask, rot_left, rot_right):
    
        return