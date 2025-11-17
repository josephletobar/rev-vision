import cv2
import numpy as np
from utils.cv_utils.transformers.base_transformer import BaseTransformer
from utils.config import LANE_W, LANE_H
from utils.cv_utils.trajectory import Trajectory

buffer = Trajectory()

def clip_to_intersection(lineL, lineR, h):
    # unpack
    x1L, y1L, x2L, y2L = lineL
    x1R, y1R, x2R, y2R = lineR

    # slopes
    mL = (y2L - y1L) / (x2L - x1L)
    mR = (y2R - y1R) / (x2R - x1R)

    # intercepts
    bL = y1L - mL * x1L
    bR = y1R - mR * x1R

    # intersection x
    xI = (bR - bL) / (mL - mR)
    # intersection y
    yI = mL * xI + bL

    # bottom y of mask
    yB = h - 1

    # bottom points for each line using their slope/intercept
    xBL = (yB - bL) / mL
    xBR = (yB - bR) / mR

    # return clipped segments
    left_clipped  = (int(xBL), int(yB), int(xI), int(yI))
    right_clipped = (int(xBR), int(yB), int(xI), int(yI))

    return left_clipped, right_clipped

def geometric_transform(t, mask):

    mask = mask.copy()

    

    # if t.debug:
    #     cv2.imshow("Geometric Visual", mask)
    #     cv2.waitKey(1)
    # return mask

    # t is BirdsEyeTransformer instance
    # to use BaseTransformer methods:

    
    # 1. find actual lane pixels
    lane_mask = t._white_mask(mask.copy()) > 0
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

    mask = canvas

    # add vertical + horizontal padding
    pad_bottom = 1200
    pad_left   = 400
    pad_right  = 400

    mask = np.pad(
        mask,
        ((0, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    # --- use l r lines to compute remainding lane --
    if t.avg_left_line is not None and t.avg_right_line is not None:

        # 1. clip in original coords
        L, R = clip_to_intersection(
            t.avg_left_line,
            t.avg_right_line,
            mask.shape[0]
        )

        # 2. transform original → cropped → scaled coords
        def transform(line):
            x1,y1,x2,y2 = line

            x1 = int((x1 - x_min) * scale)
            x2 = int((x2 - x_min) * scale)

            y1 = int((y1 - y_min) * scale)
            y2 = int((y2 - y_min) * scale)

            return (x1,y1,x2,y2)

        L = transform(L)
        R = transform(R)

        # 3. APPLY LEFT/RIGHT padding shift
        L = (L[0] + pad_left, L[1], L[2] + pad_left, L[3])
        R = (R[0] + pad_left, R[1], R[2] + pad_left, R[3])

        # 4. NOW extend to the real bottom of the full padded mask
        bottom_y = mask.shape[0] - 1   # <-- this is the tall, padded final mask

        x1,y1,x2,y2 = L
        if y1 < y2:
            L = (x1, y1, x2, bottom_y)
        else:
            L = (x1, bottom_y, x2, y2)

        x1,y1,x2,y2 = R
        if y1 < y2:
            R = (x1, y1, x2, bottom_y)
        else:
            R = (x1, bottom_y, x2, y2)

        # 5. draw
        cv2.line(mask, L[:2], L[2:], (255,255,255), 4)
        cv2.line(mask, R[:2], R[2:], (255,255,255), 4)

    else:
            print("NONE")


    threshold = 200

    mask = t._ensure_grayscale(mask)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    mask = clahe.apply(mask)

    ys, xs = t._nonzero_coords(mask)

    # define middle x point to search        
    mid_x = int(xs.mean())

    # TODO: Probably use lightweight ML model for arrrow finding

    # --- Find Middle Arrow ---

    top_offset = int(mask.shape[0] * 0.05)
    bottom_offset = int(mask.shape[0] * 0.15)

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
    y1 = int(h * 0.17)
    y2 = int(h * 0.22)
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
    ys_all, xs_all = np.where(t._white_mask(mask.copy()) > 200)

    # # overlay green points
    # mask_vis = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    # mask_vis[ys_all, xs_all] = (0, 255, 0)
    # cv2.imshow("LaneMaskPixels", mask_vis)
    # cv2.waitKey(1)

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

        # # pad only if needed
        # pad_needed = max(0, foul_y - LANE_H)

        # # build padded canvas
        # canvas_h = LANE_H + pad_needed
        # mask.copy() = np.zeros((canvas_h, LANE_W), dtype=mask.dtype)
        # mask[:LANE_H] = mask

        # draw foul line marker
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(mask, (mid_x, foul_y), 6, (0,255,0), -1)


    except:
        pass


    if t.debug:
        cv2.imshow("Geometric Visual", mask)
        cv2.waitKey(1)

    return mask