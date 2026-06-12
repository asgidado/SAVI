"""
SAVI calibration.py
Implements 9-point bivariate polynomial calibration logic, mapping
raw iris pixel coordinates to calibrated degrees of visual angle.
"""
import math
import json
import dataclasses
from dataclasses import dataclass, field

@dataclass
class CalibrationMap:
    # 9 target positions in screen degrees (computed at runtime from
    # screen dimensions and viewing distance)
    target_positions_deg: list[tuple[float, float]]  # [(x_deg, y_deg), ...]

    # Raw iris pixel samples collected at each target (mean of final 500ms)
    sample_points_px: list[tuple[float, float]]      # [(x_px, y_px), ...]

    # Fitted polynomial coefficients (x and y separately)
    # 2nd-order bivariate: 6 coefficients each
    # [c00, c10, c01, c20, c11, c02]
    poly_coeffs_x: list[float]
    poly_coeffs_y: list[float]

    # Validation result
    validation_error_deg: float     # mean angular error at validation point
    validation_passed: bool         # True if error < 2.0°

    # Metadata
    screen_width_px: int
    screen_height_px: int
    viewing_distance_cm: float      # default 57.0
    fps: float                      # 30.0
    created_at: str                 # ISO timestamp
    session_id: str

    # Targets flagged as low-quality (if fewer than 10 valid frames were collected)
    low_quality_targets: list[int] = field(default_factory=list)

    # Eye socket anchor samples at each calibration point
    # Each entry: (left_anchor_x, left_anchor_y, right_anchor_x, right_anchor_y)
    socket_anchor_points: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )


def iris_to_socket_coords(
    iris_x: float,
    iris_y: float,
    inner_corner_x: float,
    inner_corner_y: float,
    outer_corner_x: float,
    outer_corner_y: float
) -> tuple[float, float]:
    """
    Compute iris position relative to the eye socket anchor.

    The socket anchor is the midpoint of the inner and outer eye corners.
    This anchor moves rigidly with the head, so the iris-relative coordinate
    is approximately invariant to head translation and rotation.

    Returns (rel_x, rel_y) in pixels relative to socket anchor.

    Source: ADR-0001 — Head-Pose Compensated Calibration.
    Implemented in v0.0.2-patch as the iris-in-socket simplification
    of the full head-pose compensation plan.
    """
    anchor_x = (inner_corner_x + outer_corner_x) / 2.0
    anchor_y = (inner_corner_y + outer_corner_y) / 2.0
    return iris_x - anchor_x, iris_y - anchor_y


def screen_fraction_to_deg(fx: float, fy: float, screen_w_px: int, screen_h_px: int,
                            px_per_cm: float, viewing_dist_cm: float) -> tuple[float, float]:
    """
    Convert a screen fraction (e.g. fx=0.5, fy=0.5) to degrees of visual angle
    relative to the center of the screen (0°, 0°).
    """
    offset_x_px = fx * screen_w_px - screen_w_px / 2.0
    offset_y_px = fy * screen_h_px - screen_h_px / 2.0
    deg_x = math.degrees(math.atan(offset_x_px / (px_per_cm * viewing_dist_cm)))
    deg_y = math.degrees(math.atan(offset_y_px / (px_per_cm * viewing_dist_cm)))
    return deg_x, deg_y


def fit_polynomial(sample_points_px: list[tuple[float, float]],
                   target_positions_deg: list[tuple[float, float]],
                   socket_anchor_points: list[tuple[float, float, float, float]] | None = None,
                   normalize: bool = True) -> tuple[list[float], list[float]]:
    """
    Fit 2nd-order bivariate polynomial mapping:
      x_px, y_px → x_deg  (separately for y_deg)

    Feature vector for each sample:
      [1, x, y, x^2, x*y, y^2]

    Returns poly_coeffs_x, poly_coeffs_y (each 6 floats)
    """
    import numpy as np

    if socket_anchor_points and len(socket_anchor_points) == len(sample_points_px):
        converted = []
        for i, (ix, iy) in enumerate(sample_points_px):
            la_x, la_y, ra_x, ra_y = socket_anchor_points[i]
            # Use mean of left and right socket anchors
            anchor_x = (la_x + ra_x) / 2.0
            anchor_y = (la_y + ra_y) / 2.0
            rel_x = ix - anchor_x
            rel_y = iy - anchor_y
            converted.append((rel_x, rel_y))
        input_points = converted
    else:
        # Fallback: use absolute coords (preserves backward compatibility)
        input_points = sample_points_px
    
    if normalize and len(input_points) > 1:
        # Z-score normalize coordinates dynamically using the samples themselves
        xs = [p[0] for p in input_points]
        ys = [p[1] for p in input_points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x)**2 for x in xs) / len(xs)
        var_y = sum((y - mean_y)**2 for y in ys) / len(ys)
        std_x = math.sqrt(var_x) if var_x > 1e-8 else 1.0
        std_y = math.sqrt(var_y) if var_y > 1e-8 else 1.0
        
        norm_samples = []
        for p in input_points:
            nx = (p[0] - mean_x) / std_x
            ny = (p[1] - mean_y) / std_y
            norm_samples.append((nx, ny))
    else:
        norm_samples = input_points

    X = np.array([[1.0, p[0], p[1], p[0]**2, p[0]*p[1], p[1]**2]
                  for p in norm_samples])
    y_x = np.array([t[0] for t in target_positions_deg])
    y_y = np.array([t[1] for t in target_positions_deg])
    
    # Solve via regularized Ridge Regression to prevent overfitting/oscillations
    XTX = np.dot(X.T, X)
    XTy_x = np.dot(X.T, y_x)
    XTy_y = np.dot(X.T, y_y)
    
    alpha = 1e-3
    reg = alpha * np.eye(X.shape[1])
    reg[0, 0] = 0.0 # Do not regularize the intercept term
    
    try:
        coeffs_x = np.linalg.solve(XTX + reg, XTy_x)
        coeffs_y = np.linalg.solve(XTX + reg, XTy_y)
    except np.linalg.LinAlgError:
        coeffs_x, _, _, _ = np.linalg.lstsq(X, y_x, rcond=None)
        coeffs_y, _, _, _ = np.linalg.lstsq(X, y_y, rcond=None)
        
    return coeffs_x.tolist(), coeffs_y.tolist()


def apply_calibration(raw_x_px: float, raw_y_px: float, calibration_map: CalibrationMap,
                      current_socket_anchor: tuple[float, float] | None = None,
                      normalize: bool = True) -> tuple[float, float]:
    """
    Apply fitted polynomial to convert raw iris pixel coords
    to calibrated gaze degrees.
    """
    import numpy as np
    c = calibration_map
    
    # Fallback to absolute coordinates if calibration map has no socket anchors
    if current_socket_anchor and c.socket_anchor_points and len(c.socket_anchor_points) == len(c.sample_points_px):
        anchor_x, anchor_y = current_socket_anchor
        input_x = raw_x_px - anchor_x
        input_y = raw_y_px - anchor_y
        use_relative = True
    else:
        input_x = raw_x_px
        input_y = raw_y_px
        use_relative = False
    
    if normalize and c.sample_points_px and len(c.sample_points_px) > 1:
        if use_relative:
            ref_points = []
            for i, (ix, iy) in enumerate(c.sample_points_px):
                la_x, la_y, ra_x, ra_y = c.socket_anchor_points[i]
                ref_anchor_x = (la_x + ra_x) / 2.0
                ref_anchor_y = (la_y + ra_y) / 2.0
                ref_points.append((ix - ref_anchor_x, iy - ref_anchor_y))
        else:
            ref_points = c.sample_points_px

        # Re-derive same Z-score parameters from the reference points
        xs = [p[0] for p in ref_points]
        ys = [p[1] for p in ref_points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x)**2 for x in xs) / len(xs)
        var_y = sum((y - mean_y)**2 for y in ys) / len(ys)
        std_x = math.sqrt(var_x) if var_x > 1e-8 else 1.0
        std_y = math.sqrt(var_y) if var_y > 1e-8 else 1.0
        
        nx = (input_x - mean_x) / std_x
        ny = (input_y - mean_y) / std_y
    else:
        nx = input_x
        ny = input_y

    feat = np.array([1.0, nx, ny,
                     nx**2, nx*ny, ny**2])
    cal_x = float(np.dot(feat, c.poly_coeffs_x))
    cal_y = float(np.dot(feat, c.poly_coeffs_y))
    return cal_x, cal_y


def save_calibration(cal_map: CalibrationMap, path: str):
    """Save CalibrationMap to JSON file."""
    with open(path, 'w') as f:
        json.dump(dataclasses.asdict(cal_map), f, indent=2)


def load_calibration(path: str) -> CalibrationMap:
    """Load CalibrationMap from JSON file."""
    with open(path) as f:
        d = json.load(f)
    if "target_positions_deg" in d:
        d["target_positions_deg"] = [tuple(p) for p in d["target_positions_deg"]]
    if "sample_points_px" in d:
        d["sample_points_px"] = [tuple(p) for p in d["sample_points_px"]]
    
    # Backward compatibility for socket_anchor_points
    d.setdefault("socket_anchor_points", [])
    if d["socket_anchor_points"]:
        d["socket_anchor_points"] = [tuple(p) for p in d["socket_anchor_points"]]
        
    return CalibrationMap(**d)
