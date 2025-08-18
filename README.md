# RevVision

rev-vision is an early-stage sports computer-vision toolkit designed to detect and analyze activity in specialized playing surfaces. The current focus is on **bowling lanes**, but the framework is adaptable to other sports where boundary, or ball tracking is important. It combines classical computer-vision and deep-learning techniques to segment surfaces, overlay results on video, and experiment with object detection.

## Features
- **Lane segmentation**: DeepLab-based model generates per-frame lane masks for bowling lanes (extendable to other sports surfaces).  
- **Overlay visualization**: Draws the segmented lane or surface directly onto the original frame for easy viewing.  
- **Lane extraction**: Produces a “cut-out” view containing only the playing surface for further analysis.  
- **Experimental ball detection**: Prototype stage using Hough circle detection to highlight potential bowling balls.  

## Installation
Clone the repository:  
```bash
git clone https://github.com/yourusername/rev-vision.git
cd rev-vision
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

## Usage
Run the main script with a video file:  
```bash
python main.py --video path/to/video.mp4
```

The tool will display a window with the surface overlay and another with the extracted region. Press `q` to quit.

## Project Status
rev-vision is still under active development. Bowling is the current testbed, but the architecture is built to expand into other sports vision use cases.
