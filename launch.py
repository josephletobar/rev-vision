import subprocess
import os
from datetime import datetime
from time import sleep
from vision.lane_visual import post_visual

OUTPUT = True
INPUT_VIDEO = "test_videos/bowling3.mp4"
CSV_READ = "outputs/points.csv"

if OUTPUT:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    output_dr = f"outputs/{ts}"
    os.makedirs(output_dr, exist_ok=True)

p1 = subprocess.Popen(
    ["python3", "main.py", "--input", INPUT_VIDEO]
    + (["--output", f"{output_dr}/output.mp4"] if OUTPUT else [])
)

p2 = subprocess.Popen(
    ["python3", "-m", "vision.lane_visual", "--input", CSV_READ]
    + (["--output", f"{output_dr}/lane_visual.mp4"] if OUTPUT else [])
)

try:
    while True:
        if p1.poll() is not None:      # main.py finished
            p2.terminate()             # stop live plot
            break
        sleep(0.1)

    post_visual(CSV_READ)
    
except KeyboardInterrupt:
    print("\nFORCE KILLING SUBPROCESSES")

    for p in (p1, p2):
        try:
            p.kill()   # immediate, unconditional
        except Exception:
            pass
    
    post_visual(CSV_READ)

    os._exit(0)  # hard exit launcher