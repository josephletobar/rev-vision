# RevVision

RevVision delivers human-centered sports analytics using Meta glasses for computer vision, currently focused on bowling. The system extracts information from a first-person wearable perspective to help users understand their performance.

## Key Features
- **Input**: Processes video from Meta (Ray-Ban) smart glasses and delivers results shortly after each shot.
- **Lane Segmentation**: Highlights the lane environment to isolate relevant playing surfaces.
- **Ball Detection & Tracking**: Identifies the ball and follows its motion through each shot.

<p align="center">
  <video src="https://github.com/user-attachments/assets/984074f3-7ffb-42d1-9844-e7e1999da2b4"
         autoplay
         loop
         muted
         playsinline
         width="100%">
  </video>
</p>

## Project Status and Future Work
- **Implemented**: Lane segmentation and ball tracking modules are operational.
- **In Progress**: Data analysis pipeline that converts tracked data into structured metrics and reports.
- **Planned**: Output metrics such as ball trajectory, lane alignment, and shot visualizations; shot accuracy scores and richer performance metrics; extension to other sports or tasks; optimization for lower latency and more responsive feedback.

## How to Use
1. **Setup**
   ```bash
   git clone https://github.com/yourusername/rev-vision.git
   cd rev-vision
   pip install -r requirements.txt
   ```
2. **Run**
   Provide a recorded bowling session video (e.g., from `test_videos/`) to the main script:
   ```bash
   python main.py --input path/to/bowling_session.mp4
   ```

