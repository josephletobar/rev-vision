import cv2
import numpy as np

def get_mask_corners(mask: np.ndarray) -> np.ndarray:

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
    # cv2.imshow("expanded", mask)

    ys, xs = np.where(mask > 127) # return pixel coords where the mask pixel value is > 0

    # # Make a copy of the mask in color
    # vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # # Color all those points red
    # vis[ys, xs] = (0,0,255)
    # cv2.imshow("dilated where(mask>127)", vis)

    if len(xs) == 0:
        return None
    
    # Get the corners
    ytop = ys.min()    # top row of the mask
    xs_top = xs[ys == ytop]
    TL = (xs_top.min(), ytop)
    TR = (xs_top.max(), ytop)

    ybot = ys.max()   # bottom row of the mask
    xs_bot = xs[ys == ybot]
    BL = (xs_bot.min(), ybot)
    BR = (xs_bot.max(), ybot)

    print(f"TL: ({int(TL[0])}, {int(TL[1])}), "
      f"TR: ({int(TR[0])}, {int(TR[1])}), "
      f"BL: ({int(BL[0])}, {int(BL[1])}), "
      f"BR: ({int(BR[0])}, {int(BR[1])})")
    
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # convert to 3-channel
    cv2.circle(vis, TL, 7, (0,0,255), -1) # Red
    cv2.circle(vis, TR, 7, (0,255,0), -1) # Green
    cv2.circle(vis, BL, 7, (255,0,0), -1) # Blue
    cv2.circle(vis, BR, 7, (0,255,255), -1) # Yellow
    return vis

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