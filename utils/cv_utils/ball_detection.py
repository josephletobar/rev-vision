import cv2
import numpy as np
import csv

def detect_ball(img, preview, track=False, OUTPUT_PATH=None, trajectory_filter=None):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)

    # --- Blob detector params ---
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False  # don’t care about brightness
    params.filterByArea = True
    params.minArea = 75          # adjust to ball size
    params.maxArea = 20000

    params.filterByCircularity = True
    params.minCircularity = 0.6   # 1.0 = perfect circle

    params.filterByInertia = True
    params.minInertiaRatio = 0.3  # helps reject elongated blobs

    detector = cv2.SimpleBlobDetector_create(params)

    # --- Detect blobs ---
    keypoints = detector.detect(img_blur)

    best_kp = max(keypoints, key=lambda kp: kp.response, default=None)
    if best_kp is not None:
        x, y = int(best_kp.pt[0]), int(best_kp.pt[1])
        r = int(best_kp.size / 2)
        cv2.circle(preview, (x, y), r, (255, 0, 0), 2) # Draw blue outer circle

        if track:
            cv2.circle(preview, (x, y), 3, (0, 0, 255), -1) # Draw red inner dot
            out = trajectory_filter.update((x, y)) # filter the dot

            # If its a valid point, save it to the file
            if out is not None:
                with open(OUTPUT_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([x, y])

# TESTING   
if __name__ == "__main__":
    from trajectory import Trajectory
    filter = Trajectory(buffer_size=5, threshold=120)

    OUTPUT_PATH = "output/points.csv"
    EXAMPLE_PATH = "examples/points_run.csv"

    # set CSV at the start of each run
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])  # header

    # Start reading from the examples
    with open(EXAMPLE_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header if present
        for row in reader:
            x, y = map(float, row)
            # feed into your Trajectory
            out = filter.update((x, y)
                              )
            if out is None:
                with open(OUTPUT_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Not found", 0])
            if out is not None:
                with open(OUTPUT_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([x, y])

    from visualizer import visual
    visual(OUTPUT_PATH)