import cv2
import numpy as np
from config import LANE_W, LANE_H

DEBUG_BIRDS_EYE = False

class BirdsEyeTransformer():
    def __init__(self):
        self.out_size=(LANE_W, LANE_H)
        self.debug = DEBUG_BIRDS_EYE

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
                  TL, TR, BR, BL, 
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
                    
            corners = TL, TR, BR, BL
            
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

