import cv2
import numpy as np

def birds_eye(frame, mask, out_size=(400, 900)):
    # 1) binarize mask (expects 0/1 or logits)
    m = (mask > 0.5).astype(np.uint8) * 255
    h, w = m.shape

    # 2) pick a trapezoid from the mask (bottom span + one upper span)
    yb = h - 1
    xb = np.where(m[yb] > 0)[0]
    if xb.size < 2:
        return None  # not enough mask
    bl, br = (xb.min(), yb), (xb.max(), yb)

    # search for a usable “top” row
    for yt in range(int(0.35*h), int(0.7*h)):
        xt = np.where(m[yt] > 0)[0]
        if xt.size >= 20:  # tiny guardrail
            tl, tr = (xt.min(), yt), (xt.max(), yt)
            src = np.float32([tl, tr, br, bl])   # tl,tr,br,bl
            Wout, Hout = out_size
            dst = np.float32([[0,0],[Wout-1,0],[Wout-1,Hout-1],[0,Hout-1]])
            H = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(frame, H, (Wout, Hout))
    return None