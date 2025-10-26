import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import numpy as np
from typing import Tuple
from utils.utils import LANE_W, LANE_H

# Lane measurement constants (inches)
LANE_W_IN   = 41.5            # between gutters
LANE_L_IN   = 60*12           # foul line to head pin (720")
BOARD_W_IN  = LANE_W_IN/39.0  # ≈1.064"
ARROW_Y_IN  = 15*12           # 180"
ARROW_BOARDS = [5,10,15,20,25,30,35]
PIN_DECK_IN = 24              # 2 ft extension beyond the lane
GUTTER_W_IN = 9.25            # standard gutter width

# TODO: Fix lane plotting so partial trajectories don't get stretched upward; preserve true point scaling instead of auto-fitting.

def draw_lane(ax, xs, ys, lane_width_px=LANE_W, lane_height_px=LANE_H) -> Tuple[float, float, float, float]:

    """Draw a to-scale lane.
    Returns:
        (lane_left, lane_right, lane_top, lane_bottom): tuple of floats in pixel coordinates.
    """

    lane_left, lane_top = 0, 0
    lane_right, lane_bottom = lane_width_px, lane_height_px

    # draw lane background (beige)
    ax.add_patch(Rectangle((lane_left, lane_top),
                           lane_width_px, lane_height_px,
                           facecolor="#f5deb3", zorder=0))

    # --- inch→pixel scales INSIDE that box ---
    sx = lane_width_px  / LANE_W_IN   # px per inch (x)
    sy = lane_height_px / LANE_L_IN   # px per inch (y)

    # helpers: inches→pixels inside the lane box
    # inline coordinate conversion (flipped vertically)
    x_in_to_px = lambda x_in: lane_left + x_in * sx
    y_in_to_px = lambda y_in: lane_bottom - y_in * sy

    # ---- pin deck (slightly darker beige past lane end) ----
    pin_deck_px = PIN_DECK_IN * sy
    ax.add_patch(Rectangle(
        (lane_left, lane_top - pin_deck_px),
        lane_width_px,
        pin_deck_px,
        facecolor="#e5c79e",
        zorder=0
    ))

    # lane border highlight
    ax.plot([lane_left, lane_left], [lane_top, lane_bottom], color="#caa974", lw=1.0, zorder=1)
    ax.plot([lane_right, lane_right], [lane_top, lane_bottom], color="#caa974", lw=1.0, zorder=1)

    # ---- faint board lines (39 boards) ----
    for b in range(1, 39):
        x_b = lane_left + b * BOARD_W_IN * sx
        ax.plot([x_b, x_b], [lane_top, lane_bottom],
                color="#e0c89b",  # faint brownish line, slightly darker than lane
                linewidth=0.4,
                zorder=0.5)

     # ---- gutters (darker gray strips on both sides) ----
    gutter_w_px = GUTTER_W_IN * sx
    ax.add_patch(Rectangle(
        (lane_left - gutter_w_px, lane_top),
        gutter_w_px, lane_height_px,
        facecolor="#a9a9a9",  # dark gray
    ))
    ax.add_patch(Rectangle(
        (lane_right, lane_top),
        gutter_w_px, lane_height_px,
        facecolor="#a9a9a9",  # dark gray
    ))

    # ---- arrows (chevrons) at 15 ft, boards listed above ----
    for b in ARROW_BOARDS:
        x_center_in = (b - 0.5) * BOARD_W_IN
        xc = x_in_to_px(x_center_in)
        w_in, h_in = 2*BOARD_W_IN, 10  # ~2 boards half-width, 10" tall
        pts = np.array([
            [xc - w_in*sx, y_in_to_px(ARROW_Y_IN)],
            [xc,           y_in_to_px(ARROW_Y_IN + h_in)],
            [xc + w_in*sx, y_in_to_px(ARROW_Y_IN)],
        ])
        ax.add_patch(Polygon(pts, closed=True, facecolor="k", edgecolor="k", zorder=1))

    # ---- pin spots (true 12" spacing, pixel-ratio aware) ----
    row_dy_in = 12 * np.sin(np.deg2rad(60))  # ≈10.392"
    cx = x_in_to_px(LANE_W_IN / 2)
    cy = y_in_to_px(LANE_L_IN)

    pins = []
    pins += [(cx, cy)]  # head pin
    pins += [(cx - 6 * sx, cy - row_dy_in * sy), (cx + 6 * sx, cy - row_dy_in * sy)]
    pins += [(cx - 12 * sx, cy - 2 * row_dy_in * sy), (cx, cy - 2 * row_dy_in * sy), (cx + 12 * sx, cy - 2 * row_dy_in * sy)]
    pins += [(cx - 18 * sx, cy - 3 * row_dy_in * sy), (cx - 6 * sx, cy - 3 * row_dy_in * sy),
             (cx + 6 * sx, cy - 3 * row_dy_in * sy), (cx + 18 * sx, cy - 3 * row_dy_in * sy)]
    for (px, py) in pins:
        ax.add_patch(Circle(
            (px, py),
            radius=1.05 * sx,         # ~2.1" pin base diameter
            facecolor="white",
            edgecolor="k",
            linewidth=0.8,
            zorder=2
        ))

    return (lane_left, lane_right, lane_top, lane_bottom) # return boundaries