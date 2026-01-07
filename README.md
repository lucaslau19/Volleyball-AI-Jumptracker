# volleyball-ai
AI-powered volleyball jump height tracker using MediaPipe pose estimation and OpenCV. Calibrates using net height and tracks jumps in real-time with centimeter accuracy.

## Features
- Real-time pose detection
- Calibrated jump height measurements in cm
- Visual skeleton overlay
- Jump history graph
- CSV export of all jumps

## Installation
```bash
pip install mediapipe==0.10.9 opencv-python numpy
```

## Usage

1. Run the script:
```bash
python main.py
```

2. Click on the top of the volleyball net
3. Click on the ground level
4. Watch the jumps get detected!

## Controls
- **SPACEBAR**: Pause/Unpause
- **R**: Reset tracking
- **ESC**: Exit

## Requirements
- Python 3.7+
- mediapipe 0.10.9
- opencv-python
- numpy
