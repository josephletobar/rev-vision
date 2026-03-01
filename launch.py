import argparse
import subprocess
import os
from datetime import datetime
from time import sleep
from config import CSV_READ
from vision.lane_visual import post_visual

OUTPUT = True
INPUT_VIDEO = "test_videos/bowling12.mov"
SOCKET_PORT = 8765

# parse arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-i", "--input", action="store_true", help="Input video file mode")
group.add_argument("-w", "--websocket", action="store_true", help="WebSocket live mode")
args = parser.parse_args()

# default to input mode
if not args.input and not args.websocket:
    args.input = True

if OUTPUT:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    output_dr = f"outputs/{ts}"
    os.makedirs(output_dr, exist_ok=True)

if args.input:
    p1 = subprocess.Popen(
        ["python3", "main.py", "--input", INPUT_VIDEO]
        + (["--output", output_dr] if OUTPUT else [])
    )
if args.websocket:
    p1 = subprocess.Popen(
        ["python3", "main.py", "--websocket", str(SOCKET_PORT)]
        + (["--output", output_dr] if OUTPUT else [])
    )

sleep(5)

p2 = subprocess.Popen(
    ["python3", "-m", "vision.lane_visual"]
    + (["--output", f"{output_dr}/lane_visual.mp4"] if OUTPUT else [])
)

try:
    while True:
        if p1.poll() is not None:      # main.py finished
            p2.terminate()             # stop live plot
            break
        sleep(0.1)

    sleep(3)
    post_visual(CSV_READ, output_path=f"{output_dr}/lane_visual.mp4" if OUTPUT else None)
    
except KeyboardInterrupt:
    print("\nFORCE KILLING SUBPROCESSES")

    for p in (p1, p2):
        try:
            p.kill()   # immediate, unconditional
        except Exception:
            pass
    
    sleep(3)
    post_visual(CSV_READ)

    os._exit(0)  # hard exit launcher