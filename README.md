# SAVI — Saccadic Assessment Via Imaging (v0.0.1)

**Prototype: Iris Tracking Core**

SAVI is a webcam-based neurological screening application prototype. Version 0.0.1 establishes the foundation: real-time webcam capture, MediaPipe `FaceLandmarker` iris tracking, pixels-to-degrees gaze angle conversion, a real-time heads-up display (HUD), scrolling live gaze trace, and CSV session logging.

---

## Setup Instructions

### 1. Create and Activate Virtual Environment
Ensure you have Python 3.11 installed. Run:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the MediaPipe Model File
Download the `face_landmarker.task` file into the `models/` directory:
```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 4. Run the Application
Start the tracker and live visualization:
```bash
python main.py
```

### 5. Run Unit Tests
To verify all calculations and schema requirements:
```bash
pytest tests/ -v
```

---

## Features Built in v0.0.1
- **Live Video Feed**: Mirrored camera display at 640x480 resolution.
- **Iris Tracking**: Real-time crosshairs rendered on left (468) and right (473) irises.
- **Gaze Vector Overlay**: Visual vector representation of eye direction from the frame center.
- **Gaze Conversion**: Conversion of pixel offsets to physical degrees of visual angle.
- **Blink Detection**: Real-time identification of blink events when the iris area drops by >50% relative to a 10-frame rolling median.
- **Real-time Charting**: Scrolling time-series plot (pyqtgraph) showing the horizontal gaze angle over a 3-second window.
- **Rest Jitter Measurement**: A 100-frame test calculating RMS of frame-to-frame pixel displacement.
- **CSV Data Logger**: Timestamped session logs saved to the `data/` directory.
