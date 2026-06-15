# SAVI — Saccadic Assessment Via Imaging (v0.1.0)

**Webcam-based Neurological Screening & Calibration System**

SAVI is a webcam-based eye tracking application designed for neurological screening. Version 0.1.0 introduces a headless clinical protocol engine (state machine, block runner, session runner) that automates standard clinical saccade batteries (Overlap, Gap, Antisaccade trials) using synthetic or real-time GazeFrame streams.

---

## Setup Instructions

### 1. Create and Activate Virtual Environment
Ensure you have Python 3.11/3.13 installed. Run:
```bash
python3 -m venv venv
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
To verify all calculations, calibration solvers, preprocessors, saccade detectors, and protocol state machines:
```bash
pytest tests/ -v
```

---

## Headless Clinical Protocol Engine (v0.1.0)
- **Headless State Machine (`TrialEngine`)**:
  - Implements the standard clinical visual trial timeline: `FIXATION_CHECK` → `FIXATION_HOLD` → `GAP_BLANK` (Gap trials only) → `TARGET_ON` → `POST_TARGET` → `ITI` → `COMPLETE`.
  - Driven entirely by pushing `GazeFrame` objects, computing state durations dynamically using `time.perf_counter()` without background threads or timers.
  - Central fixation window validation: checks calibrated gaze coordinates against a $\pm 2.5^\circ$ window.
  - Automatically invokes the `Preprocessor` and `detector` at the end of each trial to extract usability metrics, latency, amplitude, and saccadic correctness.
- **Block-Level Execution (`BlockRunner`)**:
  - Automatically runs a single block type (`OVERLAP`, `GAP`, or `ANTISACCADE`) with randomized trial specifications.
  - Direction assignment balances left/right trials while shuffling pairs to guarantee no more than 3 consecutive trials occur in the same direction.
  - Evaluates block-level statistics like trial count, usable trial count, and block duration.
- **Battery Orchestrator (`SessionRunner`)**:
  - Runs the full clinical saccade battery in the locked sequence required by literature: Overlap (22 trials) → Gap (22 trials) → Antisaccade (33 trials) to control for fatigue effects.

---

## Signal Preprocessing & Saccade Detection (v0.0.3)
- **Blink Masking and Dilation**: Combines raw blink flags with low confidence detections (< 0.5) to build a blink mask, dilated by 2 frames to eliminate blink boundary artifacts.
- **Blink Interpolation**: Linearly interpolates gaze coordinates during brief blinks (< 200ms) to preserve signal continuity, applying edge-filling when blink events occur at trial boundaries.
- **Savitzky-Golay Smoothing**: Filters high-frequency noise using a fixed Savitzay-Golay filter (`window_length=5`, `polyorder=2`) tailored for the 30fps sampling rate.
- **Velocity Estimation**: Computes signed horizontal and vertical velocities using a central difference method (`np.gradient()`).
- **Anatomical Velocity Clamp**: Automatically clamps physiologically impossible velocities (> 1000°/s) to 0.0 to prevent artifacts from being registered as saccades.
- **I-VT Saccade Detection**: Performs asymmetric velocity-threshold identification within a 600ms post-stimulus window:
  - **Onset**: Triggers when gaze velocity exceeds 30°/s for at least 3 consecutive frames.
  - **Direction**: Computed from the average horizontal velocity over the first 3 frames.
  - **Offset**: Triggers when gaze velocity drops below 20°/s for at least 3 consecutive frames.
  - **Validation & Rejection**: Evaluates physiological duration (10–150ms), minimum amplitude (>= 0.5°), and non-negative latencies, logging detailed rejection reasons for invalid trials.
- **Trial Usability Checks**: Automatically flags a trial as unusable if more than 20% of the post-stimulus search window is obscured by blink frames.

---

## Advanced Calibration & Math Features (v0.0.2 & v0.0.2-patch)
- **9-Point Polynomial Calibration**: Maps tiny iris pixel displacements (approx. 3-6px) to screen visual angles.
  - **Dynamic Z-Score Normalization**: Scales pixel coordinates to a range of $[-1.0, 1.0]$ based on the calibration set, reducing regression matrix conditioning errors from $>10^5$ to $<10$.
  - **Ridge Regularization**: Fits coefficients using L2 regularization ($\alpha=10^{-3}$) to prevent numerical blowup or overfitting.
- **Iris-in-Socket Anchors**: Extracts eye corners (landmarks 33, 133, 262, 363) to track head pose variations, reducing absolute validation error to 1.03° without a chin rest.
- **Full-Screen Calibration UI**: Interactive full-screen widget featuring:
  - **Breathing Target Dots**: High-contrast active dots breathing dynamically (scale 1.0 to 1.1, period 2.2s).
  - **macOS Space-Bypass**: Instantly loads the view by adjusting geometry bounds rather than triggering slow native space transitions.
  - **Settling Time Optimization**: Validates points using a 2000ms duration (allowing 1000ms settling time for natural saccadic latency, collecting frames in the remaining 1000ms).
- **Calibrated Gaze HUD & Plots**: Updates the tracker window HUD labels dynamically (highlighting "Gaze X (cal)" and "Gaze Y (cal)" in blue) and streams calibrated values directly to the live scrolling chart and logs.
- **Architecture Documentation (ADRs)**: Standardized architecture decision tracking (located in the [architecture_decisions/](file:///Users/asgidado/Documents/savi/architecture_decisions) directory) to record strategic milestones, such as chin-rest-free head-pose compensation.

---

## Core Features (v0.0.1)
- **Live Video Feed**: Mirrored camera display at 640x480 resolution.
- **Iris Tracking**: Real-time crosshairs rendered on left (468) and right (473) irises.
- **Gaze Vector Overlay**: Visual vector representation of eye direction from the frame center.
- **Gaze Conversion**: Conversion of pixel offsets to physical degrees of visual angle.
- **Blink Detection**: Real-time identification of blink events when the iris area drops by >50% relative to a 10-frame rolling median.
- **Real-time Charting**: Scrolling time-series plot (pyqtgraph) showing the horizontal gaze angle over a 3-second window.
- **Rest Jitter Measurement**: A 100-frame test calculating RMS of frame-to-frame pixel displacement.
- **CSV Data Logger**: Timestamped session logs saved to the `data/` directory.

---

## Folder Structure
- `savi/`: Main application source code.
  - `savi/calibration.py`: Polynomial mapping, Ridge regression solver, and calibration JSON persistence.
  - `savi/tracker.py`: Threaded tracker pipeline processing iris meshes and blinks.
  - `savi/preprocessor.py`: Blink interpolation, Savitzky-Golay smoothing, velocity estimation, and usability check.
  - `savi/detector.py`: I-VT saccade detection and validation logic.
  - `savi/protocol.py`: Headless trial state machine, block runner, and battery orchestrator.
  - `savi/ui/`: PySide6 graphical user interfaces.
    - `savi/ui/tracker_window.py`: Visual telemetry board, scrolling trace, HUD indicators, and control buttons.
    - `savi/ui/calibration_window.py`: Borderless calibration and validation presenter.
    - `savi/ui/theme.py`: Modern dark-theme colors, fonts, and borders.
- `tests/`: Automated unit tests verifying tracking math, blink detection, calibration regression, preprocessors, detectors, and protocols.
- `architecture_decisions/`: Markdown files tracking architecture decisions and design proposals.
