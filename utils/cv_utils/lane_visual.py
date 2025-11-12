import numpy as np
import matplotlib.pyplot as plt
from .draw_lane import draw_lane, LANE_L_IN, LANE_W_IN
import cv2
from scipy.interpolate import splprep, splev
from utils.utils import LANE_W, LANE_H

def visual(file_path):

    # get the points
    pts = np.genfromtxt(file_path, delimiter=",", names=True)
    xs, ys = pts["x"], pts["y"]
    xs, ys = remove_outliers(xs, ys)

    # make a scatter/line plot
    plt.figure(figsize=(4, 7.5))
    ax = plt.gca() # get axes

    # draw the lane, return its boundary
    lane_bounds = draw_lane(ax, xs, ys)

    # using the boundary, remove stray points
    L, R, T, B = lane_bounds
    xs, ys = lane_mask(xs, ys, L, R, T, B)

    # smooth lines
    xs, ys = smooth_line(xs, ys)

    # draw points/line
    plt.plot(xs, ys, 'k-', lw=3)   

    # --- show scale in feet ---
    inch_to_ft = 1 / 12
    ax.set_xlabel("Width (boards)")
    ax.set_ylabel("Length (feet)")

    lane_width_ft = LANE_W_IN * inch_to_ft
    lane_length_ft = LANE_L_IN * inch_to_ft

    ax.set_aspect('equal')
    x_ticks = np.linspace(L, R, 5)
    y_ticks = np.linspace(T, B, 7)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(np.round(np.linspace(39, 1, 5, dtype=int))) # 39-1 boards
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(np.round(np.linspace(lane_length_ft, 0, 7), 1))    # reverse label order because axis is inverted

    # keep proportions but hide the box    
    ax.invert_yaxis() # flip y-axis to match image coordinates

    # remove border box
    for spine in ax.spines.values():
        spine.set_visible(False)

    # enable live coordinate readout in feet/boards
    def format_coord(x, y):
        width_boards = np.interp(x, [L, R], [39, 1])             # left→right
        length_feet  = np.interp(y, [T, B], [lane_length_ft, 0]) # top→bottom
        return f"Width: {width_boards:.1f} boards, Length: {length_feet:.1f} ft"
    ax.format_coord = format_coord

    plt.show()


def lane_mask(xs, ys, lane_left, lane_right, lane_top, lane_bottom):
    # keep only points inside the lane
    in_lane = (
        (xs >= lane_left) &
        (xs <= lane_right) &
        (ys >= lane_top) &
        (ys <= lane_bottom)
    )
    return xs[in_lane], ys[in_lane]

def remove_outliers(xs, ys, radius=30, threshold=5):
    pts = np.column_stack((xs, ys))
    kept = []
    for i in range(len(pts)):
        dist = np.hypot(pts[:,0] - pts[i,0], pts[:,1] - pts[i,1]) # get vector of distances for each point 
        near = dist < radius # bool array of how many distances are under the radius
        
        if np.sum(near) - 1 < threshold:  # too few neighbors
            continue  # drop this point
        kept.append(pts[i]) # else append it
        
    kept = np.array(kept)
    return kept[:,0], kept[:,1]



def smooth_line(xs, ys, THICKNESS=80, S=1000, H=LANE_H, W=LANE_W):
    canvas = np.zeros((H, W), dtype=np.uint8)

    # Assume xs, ys are already in lane coordinate space
    pts = np.column_stack((xs.astype(int), ys.astype(int)))

    # draw the thick line
    cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=THICKNESS)

    # centerline by averaging white pixels per row
    xs_center, ys_center = [], []
    for y in range(H):
        x_coords = np.where(canvas[y] > 0)[0]
        if x_coords.size:
            xs_center.append(x_coords.mean())
            ys_center.append(y)

    xs_center = np.array(xs_center)
    ys_center = np.array(ys_center)

    # spline smooth
    try:
        tck, _ = splprep([xs_center, ys_center], s=S)
        xs_center, ys_center = splev(np.linspace(0, 1, len(xs_center)), tck)
    except:
        pass

    # # --- optional visualization ---
    # cv2.polylines(canvas, [np.column_stack((xs_center.astype(int), ys_center.astype(int)))],
    #               isClosed=False, color=180, thickness=3)
    # cv2.imshow("Centerline", canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return xs_center, ys_center


# Test on existing points   
# python3 -m utils.cv_utils.lane_visual
if __name__ == "__main__":
    # visual("examples/points_run.csv")
    visual("outputs/points.csv")

    