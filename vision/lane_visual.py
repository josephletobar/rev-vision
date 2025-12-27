import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon, Circle
from utils.config import LANE_H, LANE_W, STEP, VIDEO_FPS
import cv2
import argparse
import math
import socket
import json
import time

_video = None
_video_size = None

def calculate_speed(t1, t2, distance_ft=60.0):
    ty = abs(t2 - t1) 

    mph = (distance_ft/ty) * (3600/5280)
    return mph
 
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
    DOT_BOARDS = [3, 5, 8, 11, 14, 36, 34, 31, 28, 25]
    DOT_DIAM_IN = 2
    for b in DOT_BOARDS:
        x_center_in = (b - 0.5) * BOARD_W_IN
        xc = x_in_to_px(x_center_in)
        yc = y_in_to_px(DOT_Y_IN)
        radius_px = (DOT_DIAM_IN / 4) * sx  # inches→pixels
        ax.add_patch(Circle((xc, yc), radius=radius_px, color='k', zorder=2))
        
    # ---- arrows (chevrons), realistic 12–15 ft layout ----
    ARROW_BOARDS = [(5, 12), (10, 13), (15, 14), (20, 15), (25, 14), (30, 13), (35, 12)]

    for board, y_ft in ARROW_BOARDS:
        x_center_in = (board - 0.5) * BOARD_W_IN
        xc = x_in_to_px(x_center_in)
        w_in, h_in = BOARD_W_IN / 2, 20  # adjust for visual proportions
        pts = np.array([
            [xc - w_in * sx, y_in_to_px(y_ft * 12)],
            [xc,             y_in_to_px(y_ft * 12 + h_in)],
            [xc + w_in * sx, y_in_to_px(y_ft * 12)],
        ])
        ax.add_patch(Polygon(pts, closed=True, facecolor="k", edgecolor="k", zorder=1))

    # ---- range finders (short lines 3 ft long) ----
    RANGE_BOARDS = [(10, 40), (30, 40), (15, 34), (25, 34)]

    for board, y_start_ft in RANGE_BOARDS:
        x_center_in = (board - 0.5) * BOARD_W_IN
        xc = x_in_to_px(x_center_in)

        y_start_in = y_start_ft * 12
        y_end_in   = (y_start_ft + 3) * 12   # 3 ft farther downlane

        ax.plot(
            [xc, xc],
            [y_in_to_px(y_start_in), y_in_to_px(y_end_in)],
            color="k",
            linewidth=2,
            zorder=1
        )


def live_visual():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect(("127.0.0.1", 5000))
            break          # exit retry loop ONCE connected
        except ConnectionRefusedError:
            time.sleep(0.1)

    global _video, _video_size

    plt.ion() # interavtive mode so it can update withotu blocking
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color="blue", linewidth=5)

    ax.set_xlim(0, LANE_W)
    ax.set_ylim(0, LANE_H)
    ax.invert_yaxis()

    draw_lane(ax)
    plt.show()

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

    # render once so canvas size is valid
    fig.canvas.draw()

    # init video writer only if output is requested
    if args.output and _video is None:
        w, h = fig.canvas.get_width_height()
        _video_size = (w, h)
        _video = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (w, h)
        )


    xs, ys = [], []
    # continously get data from socket
    for msg in sock.makefile():
        x, y = json.loads(msg)
        xs.append(x)
        ys.append(y)

        line.set_data(xs, ys)

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()

        if args.output:
            frame_rgba = np.asarray(fig.canvas.buffer_rgba())
            frame_bgr = cv2.cvtColor(frame_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            _video.write(frame_bgr)

        plt.pause(0.05)

    # ---- WHEN YOU ARE DONE (once) ----
    if args.output and _video is not None:
        _video.release()
        _video = None

def post_visual(file_path=None):

    # Load points
    pts = np.genfromtxt(file_path, delimiter=",", names=True)
    xs, ys = pts["x"], pts["y"]

    t1 = pts["time_stamp"][0] 
    t2 = pts["time_stamp"][-1] 

    distance_ft = abs(ys[-1] - ys[0]) / LANE_H * 60.0

    avg_speed = calculate_speed(t1, t2, distance_ft)

    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], 'b-', lw=5)  # blue dots with connecting line

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

    fig.text(
        0.5, 0.96,
        f"Average Speed: {avg_speed:.1f} mph",
        ha="center",
        va="top",
        fontsize=12
    )

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



# python3 -m vision.lane_visual
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Path to save output video (optional)")
    args = parser.parse_args()
    
    live_visual()

