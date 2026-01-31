import cv2
import numpy as np
from config import LANE_W, LANE_H

DEBUG_BIRDS_EYE = False

class BirdsEyeTransformer():
    def __init__(self):
        self.out_size=(LANE_W, LANE_H)
        self.debug = DEBUG_BIRDS_EYE

    # needed for perspective transform
    def _get_mask_corners(self, mask: np.ndarray, vis_debug=True):
        """
        Extract the four approximate corner points (TL, TR, BR, BL)
        from a binary lane mask for perspective transformation.

        Args:
            mask (np.ndarray): Binary or grayscale mask where lane pixels > 0.
            vis_debug (np.ndarray, optional): Optional visualization image.

        Returns:
            tuple: (TL, TR, BR, BL) as (x, y) integer tuples, or None if mask is empty.
        """

        if mask is None:
            msg = "[BirdsEyeTransformer._get_mask_corners] Received None mask"
            print(msg)
            raise RuntimeError(msg)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            msg = "[BirdsEyeTransformer._get_mask_corners] No nonzero pixels in mask"
            print(msg)
            raise RuntimeError(msg)

        
        offset = int(0.05 * (ys.max() - ys.min())) # offset to go x% up/down into the mask to ignore curve

        # grab top row of the mask
        ytop = ys.min() + offset
        xs_top = xs[ys == ytop]
        TL = (xs_top.min(), ytop)
        TR = (xs_top.max(), ytop)

        # grab bottom row of the mask
        ybot = ys.max() - offset*5
        xs_bot = xs[ys == ybot]
        BL = (xs_bot.min(), ybot)
        BR = (xs_bot.max(), ybot)

        # shift the cropped region upward slightly to simulate full lane depth
        ytop -= offset 
        ybot += offset * 4
        TL = (xs_top.min(), ytop)
        TR = (xs_top.max(), ytop)
        mid_x = (xs_bot.min() + xs_bot.max()) // 2
        scale = 1.2  # widen by 20%
        half_width = (xs_bot.max() - xs_bot.min()) * scale / 2
        BL = (int(mid_x - half_width), ybot)
        BR = (int(mid_x + half_width), ybot)

        # shows the found corner points
        if vis_debug is not None:
            vis_debug[ys, xs] = (255,255,255) # mark all selected pixels white
            # Print corners
            print(f"TL: ({int(TL[0])}, {int(TL[1])}), "
            f"TR: ({int(TR[0])}, {int(TR[1])}), "
            f"BL: ({int(BL[0])}, {int(BL[1])}), "
            f"BR: ({int(BR[0])}, {int(BR[1])})")
            # Show corners
            cv2.circle(vis_debug, TL, 17, (0,0,255), -1) # Red
            cv2.circle(vis_debug, TR, 17, (0,255,0), -1) # Green
            cv2.line(vis_debug, TL, TR, (255, 0, 0), 3) # Connect the top
            cv2.circle(vis_debug, BL, 17, (255,0,0), -1) # Blue
            cv2.circle(vis_debug, BR, 17, (0,255,255), -1) # Yellow
            cv2.line(vis_debug, BL, BR, (255, 0, 0), 3) # Connect the bottom
        # cv2.imshow("Corners Debug ", vis_debug) # optionally show it seperately

        return TL, TR, BR, BL

    def _stabilize_rotation(self, mask: np.ndarray, left_angle, right_angle):
 
        avg_angle = (left_angle + right_angle) / 2 # tells how much the whole lane has tilted

        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        rotation_gain = 1 #1.35 # gain constant for rotation

        M = cv2.getRotationMatrix2D(center, -np.degrees(avg_angle) * rotation_gain, 1.0)

        stabilized = cv2.warpAffine(mask, M, (w, h))

        return stabilized, M


    def transform(self, frame, mask, 
                  left_angle, right_angle, 
                  alpha=1):

        try:
            """alpha=1 keeps the full warp; smaller values relax the top edge toward its midpoint."""

            if frame is None or mask is None:
                msg = f"[BirdsEyeTransformer.transform] None frame or mask in module {__name__}"
                print(msg)
                raise RuntimeError(msg)

            if self.debug:
                vis_debug = mask.copy()

            stabilized, R = self._stabilize_rotation(mask, left_angle, right_angle)
            if stabilized is None:
                msg = "[BirdsEyeTransformer.transform] Stabilization returned None"
                print(msg)
                raise RuntimeError(msg)
            R3 = np.vstack([R, [0, 0, 1]]) # the matrix used to stabilize rotation
                    
            corners = self._get_mask_corners(stabilized, vis_debug if self.debug else None)
            if corners is None:
                msg = "[BirdsEyeTransformer.transform] Corner detection returned None"
                print(msg)
                raise RuntimeError(msg)
            
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

            # make sure stabalized is BGR (warpPerspective needs it)
            if len(stabilized.shape) == 2:
                stabilized = cv2.cvtColor(stabilized, cv2.COLOR_GRAY2BGR)

            warp = (cv2.warpPerspective(stabilized, M, (Wout, Hout)))

            ## debug

            if self.debug:
                cv2.imshow("Debug Visual", vis_debug)
                cv2.waitKey(1)

                # Lazy init: only create the writer once
                if not hasattr(self, "_writer"):
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    h, w = vis_debug.shape[:2]
                    self._writer = cv2.VideoWriter("outputs/debug_visual.mp4", fourcc, 30.0, (w, h))

                # Write current debug frame
                self._writer.write(vis_debug)

                # If ESC is pressed, release writer and close
                if cv2.waitKey(1) == 27:
                    self._writer.release()
                    del self._writer
                    cv2.destroyWindow("Debug Visual")

            # Combine stabilization (R3) and perspective (M) into a single transform
            # so all points and images are warped in one consistent coordinate system
            M_total = M @ R3

            return warp, M_total
        except Exception as e:
            print(e)
            return None, None

