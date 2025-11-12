import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle
from utils.config import LANE_H, LANE_W

def draw_lane(ax, lane_width_px=LANE_W, lane_height_px=LANE_H):
    # Lane measurement constants (inches)
    LANE_W_IN   = 41.5            # between gutters
    LANE_L_IN   = 60*12           # foul line to head pin (720")
    BOARD_W_IN  = LANE_W_IN/39.0  # ≈1.064"

    lane_left, lane_top = 0, 0
    lane_right, lane_bottom = lane_width_px, lane_height_px

    # draw lane background (beige)
    ax.add_patch(Rectangle((lane_left, lane_top),
                           lane_width_px, lane_height_px,
                           facecolor="#f5deb3", zorder=0))

    sx = lane_width_px  / LANE_W_IN   # px per inch (x)
    sy = lane_height_px / LANE_L_IN   # px per inch (y)

    # ---- faint board lines (39 boards) ----
    for b in range(1, 39):
        x_b = lane_left + b * BOARD_W_IN * sx
        ax.plot([x_b, x_b], [lane_top, lane_bottom],
                color="#e0c89b",  # faint brownish line, slightly darker than lane
                linewidth=0.4,
                zorder=0.5)
            
    # helpers: inches→pixels inside the lane box
    # inline coordinate conversion (flipped vertically)
    x_in_to_px = lambda x_in: lane_left + x_in * sx
    y_in_to_px = lambda y_in: lane_bottom - y_in * sy

    # ---- indicator dots (7.5 ft from foul line), spaced every 5 boards ----
    DOT_Y_IN = 7.5 * 12
    DOT_BOARDS = [5, 10, 15, 20, 25, 30, 35]
    DOT_DIAM_IN = 2
    for b in DOT_BOARDS:
        x_center_in = (b - 0.5) * BOARD_W_IN
        xc = x_in_to_px(x_center_in)
        yc = y_in_to_px(DOT_Y_IN)
        radius_px = (DOT_DIAM_IN / 4) * sx  # inches→pixels
        ax.add_patch(Circle((xc, yc), radius=radius_px, color='k', zorder=2))
        
    # ---- arrows (chevrons) at 15 ft ----
    ARROW_BOARDS = [5,10,15,20,25,30,35]
    ARROW_Y_IN  = 15*12 # 180"
    for b in ARROW_BOARDS:
        x_center_in = (b - 0.5) * BOARD_W_IN
        xc = x_in_to_px(x_center_in)
        w_in, h_in = BOARD_W_IN/2, 20 
        pts = np.array([
            [xc - w_in*sx, y_in_to_px(ARROW_Y_IN)],
            [xc,           y_in_to_px(ARROW_Y_IN + h_in)],
            [xc + w_in*sx, y_in_to_px(ARROW_Y_IN)],
        ])
        ax.add_patch(Polygon(pts, closed=True, facecolor="k", edgecolor="k", zorder=1))

    # # ---- range finders (start at 37 ft from foul line) ----
    # RANGE_Y_IN = 37 * 12
    # RANGE_BOARDS = [15, 20, 25]
    # for b in RANGE_BOARDS:
    #     x_center_in = (b - 0.5) * BOARD_W_IN
    #     xc = x_in_to_px(x_center_in)
    #     w_in, h_in = 2 * BOARD_W_IN, 8
    #     pts = np.array([
    #         [xc - w_in, y_in_to_px(RANGE_Y_IN)],
    #         [xc,        y_in_to_px(RANGE_Y_IN + h_in)],
    #         [xc + w_in, y_in_to_px(RANGE_Y_IN)],
    #     ])
    #     ax.add_patch(Polygon(pts, closed=True, facecolor="k", edgecolor="k", zorder=1))

def visual(file_path):
    # Load points
    pts = np.genfromtxt(file_path, delimiter=",", names=True)
    xs, ys = pts["x"], pts["y"]

    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], 'b-', lw=2)  # blue dots with connecting line

    ax.set_xlim(0, LANE_W)
    ax.set_ylim(0, LANE_H)
    ax.invert_yaxis()

    draw_lane(ax)

    # --- show scale in feet ---
    inch_to_ft = 1 / 12
    ax.set_xlabel("Width (boards)")
    ax.set_ylabel("Length (feet)")

    LANE_W_IN   = 41.5            # between gutters
    LANE_L_IN   = 60*12           # foul line to head pin (720")

    lane_width_ft = LANE_W_IN * inch_to_ft
    lane_length_ft = LANE_L_IN * inch_to_ft

    ax.set_aspect('equal')
    x_ticks = np.linspace(0, LANE_W, 5)
    y_ticks = np.linspace(0, LANE_H, 7)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(np.round(np.linspace(39, 1, 5, dtype=int)))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(np.round(np.linspace(lane_length_ft, 0, 7), 1))

    # --- animation setup --- 
    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        line.set_data(xs[:frame], ys[:frame])
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(xs),
        init_func=init,
        interval=20, # ms
        blit=False,
        repeat=False
    )

    def format_coord(x, y):
        width_boards = np.interp(x, [0, LANE_W], [39, 1])
        length_feet = np.interp(y, [0, LANE_H], [lane_length_ft, 0])
        return f"Width: {width_boards:.1f} boards, Length: {length_feet:.1f} ft"
    ax.format_coord = format_coord

    plt.show()

# python3 -m utils.cv_utils.simple_plot
if __name__ == "__main__":
    visual("examples/points_run4.csv")