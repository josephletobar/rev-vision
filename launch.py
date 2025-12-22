import subprocess
import os
from datetime import datetime
from time import sleep

ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
output_dr = f"outputs/{ts}"
os.makedirs(output_dr, exist_ok=True)

p1 = subprocess.Popen([
    "python3",
    "-m",
    "vision.lane_visual",
    "--output",
    f"{output_dr}/lane_visual.mp4"
])

p2 = subprocess.Popen([
    "python3",
    "main.py",
    "--input",
    "test_videos/bowling3.mp4",
    "--output",
    f"{output_dr}/output.mp4"
])

try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nFORCE KILLING SUBPROCESSES")

    for p in (p1, p2):
        try:
            p.kill()   # immediate, unconditional
        except Exception:
            pass

    os._exit(0)  # hard exit launcher