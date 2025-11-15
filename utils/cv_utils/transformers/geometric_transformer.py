import cv2
import numpy as np
from utils.cv_utils.transformers.base_transformer import BaseTransformer
from utils.config import LANE_W, LANE_H
from utils.cv_utils.trajectory import Trajectory

buffer = Trajectory()

class GeometricTransformer(BaseTransformer):

    def _transform(self, stabilized_mask):

        threshold = 120

        mask = self._ensure_grayscale(stabilized_mask)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        mask = clahe.apply(mask)

        ys, xs = self._nonzero_coords(mask)

        # define middle x point to search        
        mid_x = int(xs.mean())

        # TODO: Probably use lightweight ML model for arrrow finding

        # --- Find Middle Arrow ---

        top_offset    = int(LANE_H * 0.60)   # % down from top
        bottom_offset = int(LANE_H * 0.80)   # % down from top
        col = mask[top_offset:bottom_offset, mid_x-10:mid_x+10].mean(axis=1)

        # show search space
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask, (mid_x-10, top_offset), (mid_x+10, bottom_offset), (255, 0, 0), 2)

        # Detect dark blob (arrow is darker than lane)
        arrow_idxs = np.where(col < threshold)[0]   # adjust threshold as needed

        if len(arrow_idxs) > 0:
            arrow_y = top_offset + int(arrow_idxs.mean())
            buffer.push(arrow_y)
            cv2.circle(mask, (mid_x, arrow_y), 5, (0,255,0), 5)
        elif buffer.last() is not None:
            arrow_y = buffer.last()
            cv2.circle(mask, (mid_x, arrow_y), 5, (0,255,0), 5)
        else:
            print("None found")
   
        # -- Find The Row The Dots Are On --    

        h = mask.shape[0]

        top_offset = 100
        bottom_offset = 50

        # slice inside bounds
        bottom_slice = mask[h-top_offset : h-bottom_offset, :]

        h, w = mask.shape
        y1 = h - top_offset      # top of search region
        y2 = h - bottom_offset      # bottom of search region

        cv2.rectangle(mask, (0, y1), (w, y2), (255, 255, 255), 2)

        ys, xs = np.where(bottom_slice < threshold)

        if len(ys) > 0:
            dot_y = (h - top_offset) + int(np.mean(ys))   # convert slice coords -> image coords
            cv2.line(mask, (0, dot_y), (LANE_W, dot_y), (255, 0, 0), 2)
        else: pass

        cv2.circle(mask, (mid_x, dot_y), 5, (0,255,0), 5)

        try:
            print("arrow_y:", arrow_y)
            print("dot_y:", dot_y)
            # TODO: geometric transform usng lane markers as references
            px_per_ft = (dot_y - arrow_y) / 3.0 # 3 feet
            
            foul_y = int(arrow_y + (15 * px_per_ft))
        
            print("FOUL:", foul_y)

            pad_needed = (foul_y - LANE_H) // 3 # to warp

            canvas = np.zeros((LANE_H + pad_needed,LANE_W), dtype=mask.dtype)
            canvas[0:LANE_H] = mask   # shift original image UP 
            mask = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            print("SHAPE:", canvas.shape)
            cv2.circle(canvas, (mid_x, foul_y), 5, (0,255,0), 5)
        except:
            return mask

        return mask