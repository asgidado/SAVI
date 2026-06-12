"""
SAVI tracker.py
Implements the core GazeTracker class which manages camera frames,
MediaPipe FaceLandmarker, calculations (gaze, velocity, blink, jitter),
and CSV logging.
"""
import math
import time
import logging
import collections
import queue
import threading
import os
import csv
from dataclasses import dataclass
import cv2
import numpy as np

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

logger = logging.getLogger("savi.tracker")

@dataclass
class GazeFrame:
    timestamp: float          # time.perf_counter(), seconds
    frame_idx: int            # monotonic frame counter
    gaze_x_deg: float         # horizontal gaze, degrees (+ = right)
    gaze_y_deg: float         # vertical gaze, degrees (+ = down)
    left_iris_x: float        # left iris center x, pixels
    left_iris_y: float        # left iris center y, pixels
    right_iris_x: float       # right iris center x, pixels
    right_iris_y: float       # right iris center y, pixels
    velocity_deg_s: float     # instantaneous horizontal velocity, deg/s
    blink: bool               # True if blink detected this frame
    confidence: float         # 0.0–1.0, face detection confidence
    fps_actual: float         # measured fps (rolling 30-frame window)
    cal_x_deg: float | None = None   # calibrated horizontal gaze, degrees
    cal_y_deg: float | None = None   # calibrated vertical gaze, degrees
    calibration_applied: bool = False

    # Eye socket anchor landmarks (pixel coordinates)
    left_eye_inner_x: float = 0.0    # landmark 133
    left_eye_inner_y: float = 0.0
    left_eye_outer_x: float = 0.0    # landmark 33
    left_eye_outer_y: float = 0.0
    right_eye_inner_x: float = 0.0   # landmark 362
    right_eye_inner_y: float = 0.0
    right_eye_outer_x: float = 0.0   # landmark 263
    right_eye_outer_y: float = 0.0



class GazeCSVLogger:
    """
    Handles timestamped CSV logging for GazeFrame records.
    """
    def __init__(self, directory="data"):
        self.directory = directory
        self.file_path = None
        self.file = None
        self.writer = None

    def start(self) -> str:
        os.makedirs(self.directory, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.directory, f"savi_session_{timestamp_str}.csv")
        self.file = open(self.file_path, mode="w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "timestamp", "frame_idx", "gaze_x_deg", "gaze_y_deg",
            "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
            "velocity_deg_s", "blink", "confidence", "fps_actual",
            "cal_x_deg", "cal_y_deg",
            "left_eye_inner_x", "left_eye_inner_y",
            "left_eye_outer_x", "left_eye_outer_y",
            "right_eye_inner_x", "right_eye_inner_y",
            "right_eye_outer_x", "right_eye_outer_y"
        ])
        return self.file_path

    def log_frame(self, frame: GazeFrame):
        if self.writer and self.file:
            self.writer.writerow([
                frame.timestamp,
                frame.frame_idx,
                frame.gaze_x_deg,
                frame.gaze_y_deg,
                frame.left_iris_x,
                frame.left_iris_y,
                frame.right_iris_x,
                frame.right_iris_y,
                frame.velocity_deg_s,
                int(frame.blink),
                frame.confidence,
                frame.fps_actual,
                "" if frame.cal_x_deg is None else frame.cal_x_deg,
                "" if frame.cal_y_deg is None else frame.cal_y_deg,
                "" if frame.blink else frame.left_eye_inner_x,
                "" if frame.blink else frame.left_eye_inner_y,
                "" if frame.blink else frame.left_eye_outer_x,
                "" if frame.blink else frame.left_eye_outer_y,
                "" if frame.blink else frame.right_eye_inner_x,
                "" if frame.blink else frame.right_eye_inner_y,
                "" if frame.blink else frame.right_eye_outer_x,
                "" if frame.blink else frame.right_eye_outer_y
            ])
            self.file.flush()

    def stop(self):
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
            self.file_path = None


class GazeTracker:
    """
    Main tracking controller. Manages the camera thread and FaceLandmarker API
    in LIVE_STREAM mode, emitting annotated images and GazeFrame records.
    """
    def __init__(self, camera_index: int = 0, model_path: str = "models/face_landmarker.task"):
        self.camera_index = camera_index
        self.model_path = model_path
        self.running = False
        self.queue = queue.Queue()
        self.queues = [self.queue]
        self._calibration = None
        
        # Logging & CSV
        self.csv_logger = GazeCSVLogger()
        self.logging_active = False
        
        # State tracking histories
        self._fps_timestamps = []
        self._gaze_history = []  # items: (gaze_x_deg, timestamp)
        self._rolling_history = collections.deque(maxlen=200) # for jitter calculation
        
        # Blink detection histories (iris areas)
        self._left_area_history = collections.deque(maxlen=10)
        self._right_area_history = collections.deque(maxlen=10)
        
        # Thread safety caches
        self._frame_cache = {}
        self.frame_idx = 0
        
        self.cap = None
        self.landmarker = None
        self.thread = None

    def load_calibration(self, cal_map):
        self._calibration = cal_map

    def _apply_calibration_if_loaded(self, raw_x_px, raw_y_px, socket_anchor=None):
        if self._calibration is None:
            return None, None
        from savi.calibration import apply_calibration
        return apply_calibration(raw_x_px, raw_y_px, self._calibration, current_socket_anchor=socket_anchor)

    def register_queue(self, q):
        if q not in self.queues:
            self.queues.append(q)

    def unregister_queue(self, q):
        if q in self.queues:
            self.queues.remove(q)

    def open_camera(self) -> bool:
        """Opens and configures the camera on the main thread (macOS requirement)."""
        from savi.utils.camera import open_and_configure_camera
        try:
            self.cap = open_and_configure_camera(self.camera_index)
            return True
        except Exception as e:
            logger.error(f"Failed to open camera on main thread: {e}")
            return False

    def start(self):
        """Starts the background video capture and tracking thread."""
        if self.running:
            return
        # macOS AVFoundation requirement: cv2.VideoCapture must be initialized on main thread
        if self.cap is None or not self.cap.isOpened():
            if not self.open_camera():
                logger.error("Tracker start aborted: Camera failed to open.")
                return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the tracking loop and releases camera and model resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.stop_csv_logging()

    def start_csv_logging(self) -> str:
        self.logging_active = True
        return self.csv_logger.start()

    def stop_csv_logging(self):
        self.logging_active = False
        self.csv_logger.stop()

    def get_latest_jitter(self) -> float:
        """
        Computes the rolling jitter across the last 100 frames.
        Returns average RMS of displacements.
        """
        history = list(self._rolling_history)[-100:]
        if len(history) < 2:
            return 0.0
            
        displacements_left = []
        displacements_right = []
        
        for i in range(len(history) - 1):
            f1 = history[i]
            f2 = history[i+1]
            if f1.confidence > 0.0 and not f1.blink and f2.confidence > 0.0 and not f2.blink:
                dl = math.hypot(f2.left_iris_x - f1.left_iris_x, f2.left_iris_y - f1.left_iris_y)
                dr = math.hypot(f2.right_iris_x - f1.right_iris_x, f2.right_iris_y - f1.right_iris_y)
                displacements_left.append(dl ** 2)
                displacements_right.append(dr ** 2)
                
        if not displacements_left or not displacements_right:
            return 0.0
            
        rms_l = math.sqrt(sum(displacements_left) / len(displacements_left))
        rms_r = math.sqrt(sum(displacements_right) / len(displacements_right))
        return (rms_l + rms_r) / 2.0

    def measure_rest_jitter(self, n_frames: int = 100) -> float:
        """
        Synchronously blocks or polls the rolling history until n_frames
        are collected, then computes the rest jitter and prints the result.
        """
        start_time = time.time()
        # Wait until we have enough frames in history
        while len(self._rolling_history) < n_frames:
            if not self.running:
                logger.error("Tracker not running during jitter measurement.")
                return 0.0
            time.sleep(0.05)
            if time.time() - start_time > 10.0:
                raise TimeoutError(f"Timeout waiting for {n_frames} frames to measure jitter.")

        # Compute RMS displacement
        history = list(self._rolling_history)[-n_frames:]
        displacements_left = []
        displacements_right = []

        for i in range(len(history) - 1):
            f1 = history[i]
            f2 = history[i+1]
            if f1.confidence > 0.0 and not f1.blink and f2.confidence > 0.0 and not f2.blink:
                dl = math.hypot(f2.left_iris_x - f1.left_iris_x, f2.left_iris_y - f1.left_iris_y)
                dr = math.hypot(f2.right_iris_x - f1.right_iris_x, f2.right_iris_y - f1.right_iris_y)
                displacements_left.append(dl ** 2)
                displacements_right.append(dr ** 2)

        if not displacements_left or not displacements_right:
            rms = 0.0
        else:
            rms_l = math.sqrt(sum(displacements_left) / len(displacements_left))
            rms_r = math.sqrt(sum(displacements_right) / len(displacements_right))
            rms = (rms_l + rms_r) / 2.0

        status = "PASS (<3px)" if rms < 3.0 else "FAIL (>3px)"
        print(f"Jitter: {rms:.2f}px — {status}")
        return rms

    def _run_loop(self):
        """Webcam capture loop running in a separate thread."""
        # Setup MediaPipe Options
        def result_callback(result, image, timestamp_ms):
            self._process_result_callback(result, image, timestamp_ms)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=result_callback,
            num_faces=1
        )
        
        try:
            self.landmarker = FaceLandmarker.create_from_options(options)
        except Exception as e:
            logger.error(f"Failed to create MediaPipe landmarker: {e}")
            self.running = False
            return

        # Ensure camera is opened
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera is not opened. Tracker thread cannot run.")
            self.running = False
            if self.landmarker:
                self.landmarker.close()
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Accurate high-resolution timestamp
            timestamp_sec = time.perf_counter()
            timestamp_ms = int(timestamp_sec * 1000)

            # Store the frame associated with this timestamp
            self._frame_cache[timestamp_ms] = frame

            try:
                self.landmarker.detect_async(mp_image, timestamp_ms)
            except Exception as e:
                logger.error(f"Error in MediaPipe detect_async: {e}")
                self._frame_cache.pop(timestamp_ms, None)

            # Simple sleep to prevent loop from hogging CPU if FPS is not throttled by driver
            time.sleep(0.001)

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    def _process_result_callback(self, result, image, timestamp_ms):
        """Asynchronous callback invoked on MediaPipe worker thread."""
        frame = self._frame_cache.pop(timestamp_ms, None)
        if frame is None:
            # Fallback
            frame = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)

        h, w, _ = frame.shape
        self.frame_idx += 1
        now_sec = time.perf_counter()

        # Update FPS history
        self._fps_timestamps.append(now_sec)
        if len(self._fps_timestamps) > 30:
            self._fps_timestamps.pop(0)

        if len(self._fps_timestamps) > 1:
            fps_actual = (len(self._fps_timestamps) - 1) / (self._fps_timestamps[-1] - self._fps_timestamps[0])
        else:
            fps_actual = 0.0

        confidence = 0.0
        blink = False
        gaze_x_deg = 0.0
        gaze_y_deg = 0.0
        left_iris_x = 0.0
        left_iris_y = 0.0
        right_iris_x = 0.0
        right_iris_y = 0.0
        cal_x = None
        cal_y = None

        left_eye_inner_x = 0.0
        left_eye_inner_y = 0.0
        left_eye_outer_x = 0.0
        left_eye_outer_y = 0.0
        right_eye_inner_x = 0.0
        right_eye_inner_y = 0.0
        right_eye_outer_x = 0.0
        right_eye_outer_y = 0.0

        if result.face_landmarks:
            confidence = getattr(self, "mock_confidence", None)
            if confidence is None:
                confidence = 0.95
            landmarks = result.face_landmarks[0]

            # Ensure landmarks size is sufficient for iris indices
            if len(landmarks) > 477:
                # Left Iris center
                l_ctr = landmarks[468]
                left_iris_x = l_ctr.x * w
                left_iris_y = l_ctr.y * h

                # Right Iris center
                r_ctr = landmarks[473]
                right_iris_x = r_ctr.x * w
                right_iris_y = r_ctr.y * h

                # Eye socket anchors for iris-in-socket calibration (ADR-0001)
                L_INNER = 133   # left eye inner corner
                L_OUTER = 33    # left eye outer corner
                R_INNER = 362   # right eye inner corner
                R_OUTER = 263   # right eye outer corner

                l_inner = landmarks[L_INNER]
                l_outer = landmarks[L_OUTER]
                r_inner = landmarks[R_INNER]
                r_outer = landmarks[R_OUTER]

                left_eye_inner_x  = l_inner.x * w
                left_eye_inner_y  = l_inner.y * h
                left_eye_outer_x  = l_outer.x * w
                left_eye_outer_y  = l_outer.y * h
                right_eye_inner_x = r_inner.x * w
                right_eye_inner_y = r_inner.y * h
                right_eye_outer_x = r_outer.x * w
                right_eye_outer_y = r_outer.y * h

                # Conversion Parameters
                pixels_per_cm = 37.8
                viewing_distance_cm = 57.0

                # Horizontal gaze (average of left/right)
                left_offset_x = left_iris_x - (w / 2.0)
                right_offset_x = right_iris_x - (w / 2.0)
                left_theta_x = math.degrees(math.atan(left_offset_x / (pixels_per_cm * viewing_distance_cm)))
                right_theta_x = math.degrees(math.atan(right_offset_x / (pixels_per_cm * viewing_distance_cm)))
                gaze_x_deg = (left_theta_x + right_theta_x) / 2.0

                # Vertical gaze (average of left/right)
                left_offset_y = left_iris_y - (h / 2.0)
                right_offset_y = right_iris_y - (h / 2.0)
                left_theta_y = math.degrees(math.atan(left_offset_y / (pixels_per_cm * viewing_distance_cm)))
                right_theta_y = math.degrees(math.atan(right_offset_y / (pixels_per_cm * viewing_distance_cm)))
                gaze_y_deg = (left_theta_y + right_theta_y) / 2.0

                # Calculate iris areas for blink detection
                left_area = self._calculate_eye_area(landmarks, [469, 470, 471, 472], w, h)
                right_area = self._calculate_eye_area(landmarks, [474, 475, 476, 477], w, h)

                left_blink = self._check_eye_blink(left_area, self._left_area_history)
                right_blink = self._check_eye_blink(right_area, self._right_area_history)
                blink = left_blink or right_blink

                # Apply calibration if map loaded and NOT blinking
                if not blink:
                    mean_iris_x_px = (left_iris_x + right_iris_x) / 2.0
                    mean_iris_y_px = (left_iris_y + right_iris_y) / 2.0

                    # Compute socket anchor for head-pose tolerant calibration
                    la_x = (left_eye_inner_x + left_eye_outer_x) / 2.0
                    la_y = (left_eye_inner_y + left_eye_outer_y) / 2.0
                    ra_x = (right_eye_inner_x + right_eye_outer_x) / 2.0
                    ra_y = (right_eye_inner_y + right_eye_outer_y) / 2.0
                    socket_anchor = ((la_x + ra_x) / 2.0, (la_y + ra_y) / 2.0)

                    cal_x, cal_y = self._apply_calibration_if_loaded(
                        mean_iris_x_px, mean_iris_y_px,
                        socket_anchor=socket_anchor
                    )
                else:
                    cal_x, cal_y = None, None
            else:
                confidence = 0.0
                blink = True
        else:
            confidence = 0.0
            blink = True

        # Additional blink trigger
        if confidence < 0.3:
            blink = True

        # Velocity calculation
        velocity_deg_s = self._compute_velocity(gaze_x_deg, now_sec)

        # Update histories
        self._gaze_history.append((gaze_x_deg, now_sec))
        if len(self._gaze_history) > 10:
            self._gaze_history.pop(0)

        # Construct frame
        gaze_frame = GazeFrame(
            timestamp=now_sec,
            frame_idx=self.frame_idx,
            gaze_x_deg=gaze_x_deg,
            gaze_y_deg=gaze_y_deg,
            left_iris_x=0.0 if (blink or confidence == 0.0) else left_iris_x,
            left_iris_y=0.0 if (blink or confidence == 0.0) else left_iris_y,
            right_iris_x=0.0 if (blink or confidence == 0.0) else right_iris_x,
            right_iris_y=0.0 if (blink or confidence == 0.0) else right_iris_y,
            velocity_deg_s=velocity_deg_s,
            blink=blink,
            confidence=confidence,
            fps_actual=fps_actual,
            cal_x_deg=cal_x,
            cal_y_deg=cal_y,
            calibration_applied=cal_x is not None,
            left_eye_inner_x=0.0 if (blink or confidence == 0.0) else left_eye_inner_x,
            left_eye_inner_y=0.0 if (blink or confidence == 0.0) else left_eye_inner_y,
            left_eye_outer_x=0.0 if (blink or confidence == 0.0) else left_eye_outer_x,
            left_eye_outer_y=0.0 if (blink or confidence == 0.0) else left_eye_outer_y,
            right_eye_inner_x=0.0 if (blink or confidence == 0.0) else right_eye_inner_x,
            right_eye_inner_y=0.0 if (blink or confidence == 0.0) else right_eye_inner_y,
            right_eye_outer_x=0.0 if (blink or confidence == 0.0) else right_eye_outer_x,
            right_eye_outer_y=0.0 if (blink or confidence == 0.0) else right_eye_outer_y
        )

        self._rolling_history.append(gaze_frame)

        if self.logging_active:
            self.csv_logger.log_frame(gaze_frame)

        # Draw OpenCV overlay
        annotated_frame = self._draw_annotations(frame, gaze_frame)

        # Emit to all registered queues
        for q in list(self.queues):
            q.put((annotated_frame, gaze_frame))

    def _calculate_eye_area(self, landmarks, boundary_indices: list[int], w: float, h: float) -> float:
        """Computes approximate area of the iris polygon."""
        p_h1 = landmarks[boundary_indices[0]]
        p_v1 = landmarks[boundary_indices[1]]
        p_h2 = landmarks[boundary_indices[2]]
        p_v2 = landmarks[boundary_indices[3]]

        dh = math.hypot((p_h1.x - p_h2.x) * w, (p_h1.y - p_h2.y) * h)
        dv = math.hypot((p_v1.x - p_v2.x) * w, (p_v1.y - p_v2.y) * h)
        return dh * dv

    def _check_eye_blink(self, area: float, history: collections.deque) -> bool:
        """Detects if eye iris area drops below 50% of the 10-frame rolling median."""
        if len(history) < 5:
            history.append(area)
            return False

        median_area = np.median(list(history))
        history.append(area)

        if area < 0.5 * median_area:
            return True
        return False

    def _compute_velocity(self, current_gaze_x: float, current_time: float) -> float:
        """
        Computes horizontal gaze velocity.
        - Falls back to backward difference if only 2 data points are available.
        - Uses central difference over 2-frame intervals if 3 data points are available.
        """
        if not self._gaze_history:
            return 0.0

        if len(self._gaze_history) == 1:
            prev_gaze, prev_time = self._gaze_history[0]
            dt = current_time - prev_time
            return (current_gaze_x - prev_gaze) / dt if dt > 0 else 0.0
        else:
            # We have at least 2 prior points + the current point.
            # Central difference for the point at _gaze_history[-1] using current and _gaze_history[-2]
            prev_prev_gaze, prev_prev_time = self._gaze_history[-2]
            dt = current_time - prev_prev_time
            return (current_gaze_x - prev_prev_gaze) / dt if dt > 0 else 0.0

    def _draw_annotations(self, frame: np.ndarray, frame_data: GazeFrame) -> np.ndarray:
        """Draws tracking crosshairs, gaze vector, and blink indicators on the frame."""
        annotated = frame.copy()
        h, w, _ = annotated.shape

        blue_bgr = (96, 165, 250)
        amber_bgr = (36, 191, 251) # #FBBF24 in BGR

        # Draw center reference point
        cv2.circle(annotated, (w // 2, h // 2), 3, (120, 120, 120), -1)

        if frame_data.confidence > 0.0 and not frame_data.blink:
            # 1. Left crosshair (+)
            lx, ly = int(frame_data.left_iris_x), int(frame_data.left_iris_y)
            cv2.line(annotated, (lx - 8, ly), (lx + 8, ly), blue_bgr, 2)
            cv2.line(annotated, (lx, ly - 8), (lx, ly + 8), blue_bgr, 2)

            # 2. Right crosshair (+)
            rx, ry = int(frame_data.right_iris_x), int(frame_data.right_iris_y)
            cv2.line(annotated, (rx - 8, ry), (rx + 8, ry), blue_bgr, 2)
            cv2.line(annotated, (rx, ry - 8), (rx, ry + 8), blue_bgr, 2)

            # 3. Gaze Vector Line
            avg_x = (lx + rx) // 2
            avg_y = (ly + ry) // 2
            cv2.line(annotated, (w // 2, h // 2), (avg_x, avg_y), blue_bgr, 2)
            cv2.circle(annotated, (avg_x, avg_y), 4, blue_bgr, -1)

        # 4. Blink overlay
        if frame_data.blink:
            cv2.putText(annotated, "BLINK", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, amber_bgr, 3, cv2.LINE_AA)

        return annotated
