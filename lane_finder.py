import numpy as np
import cv2
from sklearn.cluster import KMeans

# global variables for line tracking
prev_left = None
prev_right = None
        

def draw_lines(img):

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    canny_image = cv2.Canny(gray, 50, 100)

    # Line detection
    lines = cv2.HoughLinesP(
        canny_image,
        rho=1,                     # distance resolution in pixels
        theta=np.pi / 180,         # angular resolution in radians
        threshold=40,              # minimum number of votes
        minLineLength=100,          # minimum length of line
        maxLineGap=50            # maximum allowed gap
    )

    img = np.copy(img)
    blank_image = np.zeros(img.shape, dtype=np.uint8)

    valid_lines = []
    features = []

    if lines is not None:
        left_lines = []
        right_lines = []

        left_features = []
        right_features = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1: # skip if horizontal
                continue

            slope = (y2 - y1) / (x2 - x1)   # Slope of the line
            if abs(slope) < 2:
                continue

            # valid_lines.append((x1, y1, x2, y2))  # Store this one since it's used

            mid_x = (x1 + x2) / 2   # horizontal midpoint

            # Draw good lines
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
                left_features.append([slope, mid_x])     # save the features were using

                # cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), 15) # draw white line
            else:
                right_lines.append((x1, y1, x2, y2))
                right_features.append([slope, mid_x])     # save the features were using

                # cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 15)  # draw white line


        # KMeans Fitting
        left_kmeans = KMeans(n_clusters=4, n_init='auto')    # Initialize KMeans clustering with 4 target clusters
        right_kmeans = KMeans(n_clusters=4, n_init='auto')    # Initialize KMeans clustering with 4 target clusters

        left_labels = left_kmeans.fit_predict(left_features)   # labels[i] = cluster number (0–3) assigned to features[i]
        right_labels = right_kmeans.fit_predict(right_features)   # labels[i] = cluster number (0–3) assigned to features[i]

        left_colors = [
            (0, 255, 0),      # Cluster 0: Green
            (0, 0, 255),      # Cluster 1: Red
            (255, 0, 255),    # Cluster 2: Magenta
            (128, 255, 0)     # Cluster 3: Lime-ish
        ]

        right_colors = [
            (255, 0, 0),      # Cluster 0: Blue
            (0, 255, 255),    # Cluster 1: Yellow
            (255, 128, 0),    # Cluster 2: Orange
            (255, 255, 255)   # Cluster 3: White
        ]

        for i, (x1, y1, x2, y2) in enumerate(left_lines):  # Use valid_lines
            cluster_id = left_labels[i]
            color = left_colors[cluster_id]
            cv2.line(blank_image, (x1, y1), (x2, y2), color, 10)

        for i, (x1, y1, x2, y2) in enumerate(right_lines):  # Use valid_lines
            cluster_id = right_labels[i]
            color = right_colors[cluster_id]
            cv2.line(blank_image, (x1, y1), (x2, y2), color, 10)


    combined = cv2.addWeighted(img, 1.0, blank_image, 1.0, 0)
    return combined