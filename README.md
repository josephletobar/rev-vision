# RevVision

RevVision delivers human-centered sports analytics using Meta glasses for computer vision, currently focused on bowling. The system extracts information from a first-person wearable perspective to help users understand their performance. This work advances sports analytics from a human-centered, wearable-device viewpoint, with applications in accessibility, coaching, and AR-enhanced interaction. It also connects to broader research in computer vision and human-computer interaction.

## Key Features
- **Input**: Processes video from Meta (Ray-Ban) smart glasses and delivers results shortly after each shot.
- **Lane Segmentation**: Highlights the lane environment to isolate relevant playing surfaces.
- **Ball Detection & Tracking**: Identifies the bowling ball and follows its motion through each shot.
- **Output Metrics**: Produces insights such as ball trajectory, lane alignment, and shot visualizations.

## Project Status and Future Work
- **Implemented**: Lane segmentation pipeline and bowling ball tracking modules are operational.
- **In Progress**: Data analysis pipeline that converts tracked data into structured metrics and reports.
- **Planned**: Add shot accuracy scores and richer performance analytics per session, extend the framework to other sports or human-movement analysis tasks, and optimize processing for lower latency and more responsive feedback.

## Basic Usage Instructions
1. **Setup**
   ```bash
   git clone https://github.com/yourusername/rev-vision.git
   cd rev-vision
   pip install -r requirements.txt
   ```
2. **Run**
   Provide a recorded bowling session video (e.g., from `test_videos/`) to the main script:
   ```bash
   python main.py --video path/to/bowling_session.mp4
   ```

