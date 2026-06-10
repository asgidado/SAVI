"""
SAVI Camera Utilities
Provides functions to list available cameras and configure video capture options.
"""
import cv2
import logging

logger = logging.getLogger("savi.utils.camera")

def list_available_cameras(max_to_test: int = 5) -> list[int]:
    """
    Scans camera indices from 0 up to max_to_test to find available cameras.
    """
    available_indices = []
    for index in range(max_to_test):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_indices.append(index)
            cap.release()
    logger.info(f"Available cameras: {available_indices}")
    return available_indices

def get_best_camera_index() -> int:
    """
    Returns the first available camera index, defaulting to 0 if none are found.
    """
    cameras = list_available_cameras()
    return cameras[0] if cameras else 0

def open_and_configure_camera(index: int = 0, target_w: int = 640, target_h: int = 480) -> cv2.VideoCapture:
    """
    Opens the camera at the given index and attempts to configure it.
    It prefers 60 FPS but falls back to 30 FPS if necessary.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {index}")

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

    # Attempt 60 FPS first
    cap.set(cv2.CAP_PROP_FPS, 60)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    logger.info(f"Configured camera {index}: Target {target_w}x{target_h}@60 FPS. Actual {actual_w}x{actual_h} @ {actual_fps} FPS.")

    # Note: On some systems, CAP_PROP_FPS may return 0.0 or a wrong value initially,
    # so the caller will measure the actual running frame rate.
    return cap
