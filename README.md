# RevVision

RevVision delivers human-centered sports analytics using Meta glasses for computer vision, currently focused on bowling. The system extracts information from a first-person wearable perspective to help users understand their performance.

## Key Features
- Processes first-person video from Meta smart glasses and produces visualizations and performance metrics in real time (when deployed on a GPU-backed processing environment).
- Segments the bowling lane to isolate relevant playing surfaces and applies geometric transformations, including perspective correction and spatial warping, to normalize the view.
- Detects the bowling ball and tracks its trajectory through each shot, extracting metrics such as speed and curvature.

<p align="center">
  <video src="https://github.com/user-attachments/assets/984074f3-7ffb-42d1-9844-e7e1999da2b4"
         autoplay
         loop
         muted
         playsinline
         width="100%">
  </video>
</p>

## How to Use

RevVision is built to operate on first-person video captured from Meta smart glasses via the Meta Wearables Device Access Toolkit. In the full system, video is streamed from the glasses to a mobile device, forwarded into the RevVision vision pipeline, and processed to produce visualizations and metrics.

This repository exposes the same processing pipeline, but uses recorded video files as the input interface for simplicity and reproducibility. This allows users to run the exact vision and analysis stages without requiring live glasses streaming or mobile setup.

### Usage (Recorded Video Interface)

1. **Setup**
   ```bash
   git clone https://github.com/yourusername/rev-vision.git
   cd rev-vision
   pip install -r requirements.txt
   ```
2. **Run**
   Set the path to a recorded bowling session video (for example, from test_videos/) inside launcher.py, then run:
   ```bash
   python launcher.py
   ```

