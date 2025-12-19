import subprocess

subprocess.Popen(["python3", "-m", "vision.lane_visual"])
subprocess.Popen(["python3", "main.py", "--input", "test_videos/bowling3.mp4"])