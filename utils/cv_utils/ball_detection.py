import cv2
import numpy as np
import csv

from utils.cv_utils.trajectory import Trajectory
trajectory = Trajectory()

def detect_ball(img, preview, track=False, output_path=None, trajectory_filter=None):

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

    best_kp = None
    best_contrast = 0

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)

        # Extract local region around keypoint
        y1, y2 = max(0, y - r), min(img_gray.shape[0], y + r)
        x1, x2 = max(0, x - r), min(img_gray.shape[1], x + r)
        roi = img_gray[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Compute local contrast (stddev of brightness)
        contrast = np.std(roi)

        # Keep the one with highest contrast
        if contrast > best_contrast:
            best_contrast = contrast
            best_kp = kp

    if track:
        # Draw all the previous points
        for i in range(1, len(trajectory.accepted)):
                    cv2.line(preview, trajectory.accepted[i-1], trajectory.accepted[i], (0, 0, 255), 5)

    # Pick only if contrast passes threshold
    if best_kp is not None and best_contrast > 10:  # threshold can be tuned (try 8–15)
        x, y = int(best_kp.pt[0]), int(best_kp.pt[1])
        r = int(best_kp.size / 2)
        cv2.circle(preview, (x, y), r, (255, 0, 0), 2)

        if track:
            cv2.circle(preview, (x, y), 3, (0, 0, 255), -1)
            trajectory.points_buffer((x, y)) # appends new points into the buffer

            with open(output_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([x, y])

        return [x, y]

# TESTING   
if __name__ == "__main__":
    from .trajectory import Trajectory
    filter = Trajectory(buffer_size=5, threshold=120)

    OUTPUT_PATH = "outputs/points.csv"
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

    from archive.lane_visual import visual
    visual(OUTPUT_PATH)