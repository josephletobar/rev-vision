
STEP = 1 # Video reading step
VIDEO_FPS = 30 # Glasses FPS

CSV_READ = "outputs/points.csv"

# Base pixel dimensions used for all LANE-aligned data (warp, tracking, plots)
LANE_W = 400   # width  (across lane boards)
LANE_H = 900   # height (lane length)

# Debug Flags
DEBUG_PIPELINE = False # high-level debug in main
DEBUG_BIRDS_EYE = False # debug birds-eye module

# Model paths
BALL_MODEL = "weights/ball/best_ball_7.pt"  # YOLOv8 ball detection model path 
# NOTE: 15 specific to bowling6, 7 more general
# LANE_MODEL = "weights/lane/lane_deeplab_model_2.pth"  # Lane segmentation model path
LANE_MODEL = "weights/lane_segmentation_model.pt"  # Lane segmentation model path