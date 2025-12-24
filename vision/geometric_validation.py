import cv2
import numpy as np

def _get_contour_and_hull(mask):

    mask = mask.copy()
    mask = cv2.copyMakeBorder(mask, 0, 100, 0, 0, cv2.BORDER_CONSTANT, value=0)

    if mask.ndim == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # must be grayscale
    mask = mask.astype(np.uint8)

    ret, threshold = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY) # binary threshold
    contours, hiearchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find external contours
    
    if len(contours) == 0:
        contour = None
    else:
        contour = max(contours, key=cv2.contourArea) # the biggest contour surrounds the lane

    countour_vis = mask.copy()
    cv2.drawContours(
        countour_vis,
        [contour],   # must be a list
        -1,          # draw all points in that contour
        (255, 255, 255), 
        -1          
    )

    hull = cv2.convexHull(contour)
    hull_vis = mask.copy()
    cv2.drawContours(hull_vis, [hull], -1, (255, 255, 255), -1)
    
    return countour_vis, hull_vis

def validate(mask, threshold=15000):
    contour, hull = _get_contour_and_hull(mask)

    missing = (hull == 255) & (contour == 0) # uses bitwise and to produce a boolean image where True is when when hull and contour are different values

    missing_sum = missing.sum() 

    # print("MISSING SUM:", missing_sum)

    return missing_sum <= threshold