import cv2
import numpy as np
from utils.utils import LANE_W, LANE_H

class BirdsEyeTransformer:
    def __init__(self, out_size=(LANE_W, LANE_H)):
        self.out_size = out_size

    def _get_mask_corners(self, mask: np.ndarray, debug: bool=False):

        # make sure its black and white
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        """
        mask: binary (0/255)
        returns 4 corner points (TL, TR, BR, BL) as float32
        """

        edges = cv2.Canny(mask, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=100,
            maxLineGap=10
        )


        ys, xs = np.where(mask > 127) # return pixel coords where the mask pixel value is > 0

        if len(xs) == 0:
            return None
        
        #TODO: implement better segmentation 'cropping'
        offset = int(0.05 * (ys.max() - ys.min())) # offset to go x% up/down into the mask to ignore curve

        # top row of the mask
        ytop = ys.min() + offset
        xs_top = xs[ys == ytop]
        TL = (xs_top.min(), ytop)
        TR = (xs_top.max(), ytop)

        # bottom row of the mask
        ybot = ys.max() - offset*5
        xs_bot = xs[ys == ybot]
        BL = (xs_bot.min(), ybot)
        BR = (xs_bot.max(), ybot)

        # shows the found corner points
        if debug:
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis[ys, xs] = (255,255,255) # mark all selected pixels white
            # Print corners
            print(f"TL: ({int(TL[0])}, {int(TL[1])}), "
            f"TR: ({int(TR[0])}, {int(TR[1])}), "
            f"BL: ({int(BL[0])}, {int(BL[1])}), "
            f"BR: ({int(BR[0])}, {int(BR[1])})")

            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 10)


            # Show corners
            cv2.circle(vis, TL, 17, (0,0,255), -1) # Red
            cv2.circle(vis, TR, 17, (0,255,0), -1) # Green
            cv2.line(vis, TL, TR, (255, 0, 0), 3) # Connect the top
            cv2.circle(vis, BL, 17, (255,0,0), -1) # Blue
            cv2.circle(vis, BR, 17, (0,255,255), -1) # Yellow
            cv2.imshow("Corner Visualization", vis)

        return TL, TR, BR, BL

    #TODO: head rotation adjustment
    def warp(self, frame, mask, alpha=1):
        """alpha=1 keeps the full warp; smaller values relax the top edge toward its midpoint."""
        corners = self._get_mask_corners(mask, debug=True)
        if corners is None:
            return None

        src = np.float32(corners)  # TL, TR, BR, BL
        Wout, Hout = self.out_size

        dst = np.float32([
            [0, 0],
            [Wout - 1, 0],
            [Wout - 1, Hout - 1],
            [0, Hout - 1],
        ])

        mid_x = (dst[0, 0] + dst[1, 0]) / 2.0
        dst[0, 0] = mid_x - alpha * (mid_x - dst[0, 0])
        dst[1, 0] = mid_x + alpha * (dst[1, 0] - mid_x)

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (Wout, Hout))


