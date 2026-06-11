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
                   normalize: bool = True) -> tuple[list[float], list[float]]:
    """
    Fit 2nd-order bivariate polynomial mapping:
      x_px, y_px → x_deg  (separately for y_deg)

    Feature vector for each sample:
      [1, x, y, x^2, x*y, y^2]

    Returns poly_coeffs_x, poly_coeffs_y (each 6 floats)
    """
    import numpy as np
    
    if normalize and len(sample_points_px) > 1:
        # Z-score normalize coordinates dynamically using the samples themselves
        xs = [p[0] for p in sample_points_px]
        ys = [p[1] for p in sample_points_px]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x)**2 for x in xs) / len(xs)
        var_y = sum((y - mean_y)**2 for y in ys) / len(ys)
        std_x = math.sqrt(var_x) if var_x > 1e-8 else 1.0
        std_y = math.sqrt(var_y) if var_y > 1e-8 else 1.0
        
        norm_samples = []
        for p in sample_points_px:
            nx = (p[0] - mean_x) / std_x
            ny = (p[1] - mean_y) / std_y
            norm_samples.append((nx, ny))
    else:
        norm_samples = sample_points_px

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
                      normalize: bool = True) -> tuple[float, float]:
    """
    Apply fitted polynomial to convert raw iris pixel coords
    to calibrated gaze degrees.
    """
    import numpy as np
    c = calibration_map
    
    if normalize and c.sample_points_px and len(c.sample_points_px) > 1:
        # Re-derive same Z-score parameters from the saved sample points
        xs = [p[0] for p in c.sample_points_px]
        ys = [p[1] for p in c.sample_points_px]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x)**2 for x in xs) / len(xs)
        var_y = sum((y - mean_y)**2 for y in ys) / len(ys)
        std_x = math.sqrt(var_x) if var_x > 1e-8 else 1.0
        std_y = math.sqrt(var_y) if var_y > 1e-8 else 1.0
        
        nx = (raw_x_px - mean_x) / std_x
        ny = (raw_y_px - mean_y) / std_y
    else:
        nx = raw_x_px
        ny = raw_y_px

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
    return CalibrationMap(**d)
