import cv2
import numpy as np

class BirdsEyeTransformer:
    def __init__(self, out_size=(400, 900)):
        self.out_size = out_size

    def _get_mask_corners(self, mask: np.ndarray, debug: bool=False):

        # make sure its black and white
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        """
        mask: binary (0/255)
        returns 4 corner points (TL, TR, BR, BL) as float32
        """
        # grow the mask outwards so its edges expand until the image border cuts them flat
        k = np.ones((25,5), np.uint8)   # "radius" of expansion
        mask = cv2.dilate(mask, k, iterations=1)

        ys, xs = np.where(mask > 127) # return pixel coords where the mask pixel value is > 0

        if len(xs) == 0:
            return None
        
        ytop = ys.min()    # top row of the mask
        xs_top = xs[ys == ytop]
        TL = (xs_top.min(), ytop)
        TR = (xs_top.max(), ytop)

        ybot = ys.max()   # bottom row of the mask
        xs_bot = xs[ys == ybot]
        BL = (xs_bot.min(), ybot)
        BR = (xs_bot.max(), ybot)

        if debug:
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis[ys, xs] = (255,255,255) # mark all selected pixels white
            # Print corners
            print(f"TL: ({int(TL[0])}, {int(TL[1])}), "
            f"TR: ({int(TR[0])}, {int(TR[1])}), "
            f"BL: ({int(BL[0])}, {int(BL[1])}), "
            f"BR: ({int(BR[0])}, {int(BR[1])})")
            # Show corners
            cv2.circle(vis, TL, 7, (0,0,255), -1) # Red
            cv2.circle(vis, TR, 7, (0,255,0), -1) # Green
            cv2.circle(vis, BL, 7, (255,0,0), -1) # Blue
            cv2.circle(vis, BR, 7, (0,255,255), -1) # Yellow
            cv2.imshow("Corner Visualization", vis)

        return TL, TR, BR, BL

    def warp(self, frame, mask):
        corners = self._get_mask_corners(mask, debug=False)
        if corners is None:
            return None

        src = np.float32(corners)  # TL, TR, BR, BL

        Wout, Hout = self.out_size
        dst = np.float32([[0,0],
                        [Wout-1,0],
                        [Wout-1,Hout-1],
                        [0,Hout-1]])

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (Wout, Hout))

