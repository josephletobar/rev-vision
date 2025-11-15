import cv2
import numpy as np
from utils.cv_utils.transformers.base_transformer import BaseTransformer
from utils.config import LANE_W, LANE_H
from utils.cv_utils.trajectory import Trajectory

buffer = Trajectory()

class GeometricTransformer(BaseTransformer):

    def _fit_mask(self, mask):

        # 1. find actual lane pixels
        lane_mask = self._white_mask(mask.copy()) > 0
        ys, xs = np.where(lane_mask)

        if len(xs) == 0 or len(ys) == 0:
            # fallback: no lane found, just return a blank canvas
            return np.zeros((LANE_H, LANE_W), dtype=mask.dtype)

        # bounding box of REAL lane
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # 2. crop only the actual lane region (no black borders)
        lane_crop = mask[y_min:y_max+1, x_min:x_max+1]

        h0, w0 = lane_crop.shape[:2]

        # 3. scale to fit inside canvas (no cropping)
        scale = min(LANE_W / w0, LANE_H / h0)
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)

        resized = cv2.resize(lane_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 4. place it top-left on the target
        canvas = np.zeros((LANE_H, LANE_W), dtype=resized.dtype)
        canvas[:new_h, :new_w] = resized

        return canvas

    def _transform(self, stabilized_mask):

        mask = self._fit_mask(stabilized_mask)

        threshold = 200

        mask = self._ensure_grayscale(mask)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        mask = clahe.apply(mask)

        ys, xs = self._nonzero_coords(mask)

        # define middle x point to search        
        mid_x = int(xs.mean())

        # TODO: Probably use lightweight ML model for arrrow finding

        # --- Find Middle Arrow ---

        top_offset = int(mask.shape[0] * 0.10)
        bottom_offset = int(mask.shape[0] * 0.20)

        # FIX: slice must be top → bottom, not reversed
        col = mask[top_offset:bottom_offset, mid_x-30:mid_x+30].mean(axis=1)

        # show search space
        # show rectangle search space
        cv2.rectangle(
            mask,
            (mid_x - 30, top_offset),      # top-left
            (mid_x + 30, bottom_offset),   # bottom-right
            (255, 0, 0),
            2
        )

        # Detect dark blob (arrow is darker than lane)
        arrow_idxs = np.where(col < threshold)[0]

        if len(arrow_idxs) > 0:
            arrow_y = top_offset + int(arrow_idxs.mean())
            buffer.push(arrow_y)
            cv2.circle(mask, (mid_x, arrow_y), 5, (0,255,0), 5)
        elif buffer.last() is not None:
            arrow_y = buffer.last()
            cv2.circle(mask, (mid_x, arrow_y), 5, (0,255,0), 5)
        else:
            print("NO ARROW")
   
        # -- Find The Row The Dots Are On --    

        h, w = mask.shape

        # slice inside bounds
        y1 = int(h * 0.25)
        y2 = int(h * 0.35)
        bottom_slice = mask[y1:y2, :]

        cv2.rectangle(mask, (0, y1), (w, y2), (255, 255, 255), 2)

        ys_dot, xs_dot = np.where(bottom_slice < threshold)

        if len(ys_dot) > 0:
            dot_y = y1 + int(np.mean(ys_dot))   # convert slice coords -> image coords
            cv2.line(mask, (0, dot_y), (LANE_W, dot_y), (255, 0, 0), 2)
            cv2.circle(mask, (mid_x, dot_y), 5, (0,255,0), 5)
        else:
            return mask  # no dots found, bail out for now

        # ---- expansion + foul line ----

        # full-mask coords for width
        ys_all, xs_all = np.where(self._white_mask(mask.copy()) > 200)

        # overlay green points
        mask_vis = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
        mask_vis[ys_all, xs_all] = (0, 255, 0)

        cv2.imshow("LaneMaskPixels", mask_vis)
        cv2.waitKey(1)

        try: 

            print("arrow_y:", arrow_y)
            print("dot_y:", dot_y)

            # width at arrow_y
            xs_arrow = xs_all[ys_all == arrow_y]
            xs_dotrow = xs_all[ys_all == dot_y]

            if xs_arrow.size == 0 or xs_dotrow.size == 0:
                print("empty row for width, skipping expansion")
                return mask

            w_arrow = xs_arrow.max() - xs_arrow.min()
            w_dot   = xs_dotrow.max() - xs_dotrow.min()

            if w_arrow <= 0:
                print("bad w_arrow, skipping")
                return mask

            expansion = w_dot / w_arrow

            # raw foul line estimate, corrected by expansion
            px_per_ft = (dot_y - arrow_y) / 3.0
            foul_y = int(arrow_y + (15 * px_per_ft) )
            print("FOUL:", foul_y)

            # pad only if needed
            pad_needed = max(0, foul_y - LANE_H)

            # build padded canvas
            canvas_h = LANE_H + pad_needed
            canvas = np.zeros((canvas_h, LANE_W), dtype=mask.dtype)
            canvas[:LANE_H] = mask

            # draw foul line marker
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            cv2.circle(canvas, (mid_x, foul_y), 6, (0,255,0), -1)

            mask = canvas

        except:
            pass

        return mask