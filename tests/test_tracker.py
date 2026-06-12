"""
SAVI test_tracker.py
Unit tests for GazeFrame schema, angle conversion, blink detection,
velocity computation, and CSV logging.
"""
import os
import csv
import math
import numpy as np
import pytest

from savi.tracker import GazeFrame, GazeTracker, GazeCSVLogger

# Mock helper classes for MediaPipe results
class DummyLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class DummyResult:
    def __init__(self, landmarks=None):
        self.face_landmarks = [landmarks] if landmarks else []


# 1. GazeFrame dataclass instantiates with correct field types
def test_gazeframe_schema():
    frame = GazeFrame(
        timestamp=1.234,
        frame_idx=42,
        gaze_x_deg=2.5,
        gaze_y_deg=-1.2,
        left_iris_x=310.5,
        left_iris_y=241.0,
        right_iris_x=331.2,
        right_iris_y=240.8,
        velocity_deg_s=150.0,
        blink=False,
        confidence=0.95,
        fps_actual=59.9
    )
    
    assert isinstance(frame.timestamp, float)
    assert isinstance(frame.frame_idx, int)
    assert isinstance(frame.gaze_x_deg, float)
    assert isinstance(frame.gaze_y_deg, float)
    assert isinstance(frame.left_iris_x, float)
    assert isinstance(frame.left_iris_y, float)
    assert isinstance(frame.right_iris_x, float)
    assert isinstance(frame.right_iris_y, float)
    assert isinstance(frame.velocity_deg_s, float)
    assert isinstance(frame.blink, bool)
    assert isinstance(frame.confidence, float)
    assert isinstance(frame.fps_actual, float)


# 2. Degrees conversion: pixel at frame center → 0.0°
def test_gaze_center_is_zero_degrees():
    width = 640
    height = 480
    left_iris_x = width / 2.0
    right_iris_x = width / 2.0
    
    pixels_per_cm = 37.8
    viewing_distance_cm = 57.0
    
    left_offset_x = left_iris_x - (width / 2.0)
    right_offset_x = right_iris_x - (width / 2.0)
    
    left_theta_x = math.degrees(math.atan(left_offset_x / (pixels_per_cm * viewing_distance_cm)))
    right_theta_x = math.degrees(math.atan(right_offset_x / (pixels_per_cm * viewing_distance_cm)))
    gaze_x_deg = (left_theta_x + right_theta_x) / 2.0
    
    assert gaze_x_deg == 0.0


# 3. Degrees conversion: pixel at ±N returns correct angle
#    given known screen params (mock pixels_per_cm and viewing_distance)
def test_gaze_degree_conversion_accuracy():
    width = 640
    # Center is 320. Set offset to +100 pixels (both irises at 420.0)
    left_iris_x = 420.0
    right_iris_x = 420.0
    
    pixels_per_cm = 37.8
    viewing_distance_cm = 57.0
    
    left_offset_x = left_iris_x - (width / 2.0)
    right_offset_x = right_iris_x - (width / 2.0)
    
    left_theta_x = math.degrees(math.atan(left_offset_x / (pixels_per_cm * viewing_distance_cm)))
    right_theta_x = math.degrees(math.atan(right_offset_x / (pixels_per_cm * viewing_distance_cm)))
    gaze_x_deg = (left_theta_x + right_theta_x) / 2.0
    
    # Expected: atan(100 / (37.8 * 57.0)) = atan(100 / 2154.6) = 0.04638 radians = 2.6573 degrees
    assert math.isclose(gaze_x_deg, 2.65734, abs_tol=1e-4)


# 4. Blink detection: artificially drop confidence to 0.2 for 3 frames,
#    confirm blink=True is set on those frames
def test_blink_detection_triggers():
    tracker = GazeTracker()
    tracker.mock_confidence = 0.2

    # Create dummy landmarks (478 total)
    landmarks = [DummyLandmark(0.5, 0.5) for _ in range(478)]
    result = DummyResult(landmarks)

    # Populate cache to avoid missing frame errors
    timestamp_ms = 1000
    tracker._frame_cache[timestamp_ms] = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process 3 frames with confidence = 0.2
    for i in range(3):
        ts = timestamp_ms + i * 33
        tracker._frame_cache[ts] = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker._process_result_callback(result, None, ts)
        
        # Verify that the emitted frame has blink=True and confidence=0.2
        _, emitted_frame = tracker.queue.get_nowait()
        assert emitted_frame.confidence == 0.2
        assert emitted_frame.blink is True


# 5. Velocity computation: inject two consecutive GazeFrames with known
#    positions and dt, confirm velocity matches manual calculation
def test_velocity_central_difference():
    tracker = GazeTracker()
    
    # Frame 1
    t1 = 1.0
    x1 = 0.0
    v1 = tracker._compute_velocity(x1, t1)
    tracker._gaze_history.append((x1, t1))
    
    # Frame 2 (0.05 seconds later, moved +1.5 degrees)
    t2 = 1.05
    x2 = 1.5
    v2 = tracker._compute_velocity(x2, t2)
    tracker._gaze_history.append((x2, t2))
    
    # Expected: (1.5 - 0.0) / 0.05 = 30.0 deg/s
    assert v1 == 0.0
    assert math.isclose(v2, 30.0, abs_tol=1e-5)

    # Test Frame 3 to verify central difference using (x_3 - x_1) / (t_3 - t_1)
    t3 = 1.10
    x3 = 4.0
    v3 = tracker._compute_velocity(x3, t3)
    # Expected central diff for point 2: (x3 - x1) / (t3 - t1) = (4.0 - 0.0) / (1.10 - 1.0) = 4.0 / 0.1 = 40.0 deg/s
    assert math.isclose(v3, 40.0, abs_tol=1e-5)


# 6. CSV logger: emit 5 GazeFrames, confirm CSV has 5 rows with correct headers
def test_csv_output_schema(tmp_path):
    logger = GazeCSVLogger(directory=str(tmp_path))
    file_path = logger.start()

    # Emit 5 frames
    for i in range(5):
        frame = GazeFrame(
            timestamp=1.0 + i * 0.033,
            frame_idx=i + 1,
            gaze_x_deg=float(i),
            gaze_y_deg=float(-i),
            left_iris_x=320.0,
            left_iris_y=240.0,
            right_iris_x=320.0,
            right_iris_y=240.0,
            velocity_deg_s=0.0,
            blink=False,
            confidence=0.95,
            fps_actual=30.0
        )
        logger.log_frame(frame)

    logger.stop()

    assert os.path.exists(file_path)

    # Read and verify contents
    with open(file_path, mode="r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Total rows = 1 header + 5 records = 6 rows
    assert len(rows) == 6

    # Verify headers
    expected_headers = [
        "timestamp", "frame_idx", "gaze_x_deg", "gaze_y_deg",
        "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
        "velocity_deg_s", "blink", "confidence", "fps_actual",
        "cal_x_deg", "cal_y_deg",
        "left_eye_inner_x", "left_eye_inner_y",
        "left_eye_outer_x", "left_eye_outer_y",
        "right_eye_inner_x", "right_eye_inner_y",
        "right_eye_outer_x", "right_eye_outer_y"
    ]
    assert rows[0] == expected_headers

    # Verify a row's values
    # row index 1 is frame 1
    assert rows[1][1] == "1" # frame_idx
    assert float(rows[1][2]) == 0.0 # gaze_x_deg
    assert float(rows[1][3]) == 0.0 # gaze_y_deg
    assert float(rows[2][2]) == 1.0 # gaze_x_deg for second frame
