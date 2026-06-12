"""
SAVI test_calibration.py
Unit tests for 9-point bivariate polynomial calibration logic,
persistence, and screen fraction conversion.
"""
import os
import math
import pytest
from savi.calibration import (
    CalibrationMap, screen_fraction_to_deg, fit_polynomial,
    apply_calibration, save_calibration, load_calibration
)

# 1. fit_polynomial: inject 9 synthetic sample→target pairs with a known
#    linear relationship, confirm polynomial recovers that mapping within
#    0.1° at a held-out test point
def test_polynomial_fit_accuracy():
    # Linear relation: x_deg = 0.1 * x_px - 5.0, y_deg = 0.2 * y_px + 2.0
    xs_px = [100.0, 320.0, 600.0]
    ys_px = [80.0, 240.0, 400.0]
    
    sample_points_px = []
    target_positions_deg = []
    for x_px in xs_px:
        for y_px in ys_px:
            sample_points_px.append((x_px, y_px))
            x_deg = 0.1 * x_px - 5.0
            y_deg = 0.2 * y_px + 2.0
            target_positions_deg.append((x_deg, y_deg))
            
    # Fit polynomial coefficients
    coeffs_x, coeffs_y = fit_polynomial(sample_points_px, target_positions_deg)
    
    # Held-out test point: (200.0, 150.0) 
    # Expected: x_deg = 0.1 * 200.0 - 5.0 = 15.0, y_deg = 0.2 * 150.0 + 2.0 = 32.0
    cal_map = CalibrationMap(
        target_positions_deg=target_positions_deg,
        sample_points_px=sample_points_px,
        poly_coeffs_x=coeffs_x,
        poly_coeffs_y=coeffs_y,
        validation_error_deg=0.0,
        validation_passed=True,
        screen_width_px=640,
        screen_height_px=480,
        viewing_distance_cm=57.0,
        fps=30.0,
        created_at="2026-06-10T12:00:00Z",
        session_id="test_session"
    )
    
    cal_x, cal_y = apply_calibration(200.0, 150.0, cal_map)
    assert math.isclose(cal_x, 15.0, abs_tol=0.1)
    assert math.isclose(cal_y, 32.0, abs_tol=0.1)


# 2. apply_calibration: given known coefficients and a specific pixel input,
#    confirm output matches manual calculation
def test_apply_calibration_math():
    # Bivariate 2nd-order poly: c00 + c10*x + c01*y + c20*x^2 + c11*x*y + c02*y^2
    coeffs_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    coeffs_y = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    
    cal_map = CalibrationMap(
        target_positions_deg=[],
        sample_points_px=[],
        poly_coeffs_x=coeffs_x,
        poly_coeffs_y=coeffs_y,
        validation_error_deg=0.0,
        validation_passed=True,
        screen_width_px=640,
        screen_height_px=480,
        viewing_distance_cm=57.0,
        fps=30.0,
        created_at="2026-06-10T12:00:00Z",
        session_id="test_session"
    )
    
    # Input: x_px = 2.0, y_px = 3.0
    # Manual X: 1.0 + 2.0*2 + 3.0*3 + 4.0*4 + 5.0*6 + 6.0*9 
    #           = 1.0 + 4.0 + 9.0 + 16.0 + 30.0 + 54.0 = 114.0
    # Manual Y: 0.5 + 1.5*2 + 2.5*3 + 3.5*4 + 4.5*6 + 5.5*9
    #           = 0.5 + 3.0 + 7.5 + 14.0 + 27.0 + 49.5 = 101.5
    cal_x, cal_y = apply_calibration(2.0, 3.0, cal_map, normalize=False)
    assert math.isclose(cal_x, 114.0, abs_tol=1e-5)
    assert math.isclose(cal_y, 101.5, abs_tol=1e-5)


# 3. save_calibration / load_calibration round-trip: create a CalibrationMap,
#    save to a temp file, reload, confirm all fields identical
def test_calibration_json_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "test_cal.json")
    cal_map = CalibrationMap(
        target_positions_deg=[(1.0, 2.0), (3.0, 4.0)],
        sample_points_px=[(100.0, 200.0), (300.0, 400.0)],
        poly_coeffs_x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        poly_coeffs_y=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        validation_error_deg=1.45,
        validation_passed=True,
        screen_width_px=1920,
        screen_height_px=1080,
        viewing_distance_cm=57.0,
        fps=30.0,
        created_at="2026-06-10T12:00:00Z",
        session_id="test_session",
        low_quality_targets=[3, 5]
    )
    
    save_calibration(cal_map, path)
    loaded = load_calibration(path)
    
    assert loaded.target_positions_deg == cal_map.target_positions_deg
    assert loaded.sample_points_px == cal_map.sample_points_px
    assert loaded.poly_coeffs_x == cal_map.poly_coeffs_x
    assert loaded.poly_coeffs_y == cal_map.poly_coeffs_y
    assert loaded.validation_error_deg == cal_map.validation_error_deg
    assert loaded.validation_passed == cal_map.validation_passed
    assert loaded.screen_width_px == cal_map.screen_width_px
    assert loaded.screen_height_px == cal_map.screen_height_px
    assert loaded.viewing_distance_cm == cal_map.viewing_distance_cm
    assert loaded.fps == cal_map.fps
    assert loaded.created_at == cal_map.created_at
    assert loaded.session_id == cal_map.session_id
    assert loaded.low_quality_targets == cal_map.low_quality_targets


# 4. screen_fraction_to_deg: confirm (0.5, 0.5) → (0.0°, 0.0°) center
#    and corner fractions return correct sign (left = negative x, top = negative y)
def test_screen_fraction_to_deg_center_is_zero():
    w = 1920
    h = 1080
    px_per_cm = 37.8
    viewing_dist = 57.0
    
    # Center fraction (0.5, 0.5)
    cx, cy = screen_fraction_to_deg(0.5, 0.5, w, h, px_per_cm, viewing_dist)
    assert math.isclose(cx, 0.0, abs_tol=1e-5)
    assert math.isclose(cy, 0.0, abs_tol=1e-5)
    
    # Top-Left (0.1, 0.1) -> offset left (negative x) and above (negative y)
    tlx, tly = screen_fraction_to_deg(0.1, 0.1, w, h, px_per_cm, viewing_dist)
    assert tlx < 0.0
    assert tly < 0.0
    
    # Bottom-Right (0.9, 0.9) -> offset right (positive x) and below (positive y)
    brx, bry = screen_fraction_to_deg(0.9, 0.9, w, h, px_per_cm, viewing_dist)
    assert brx > 0.0
    assert bry > 0.0


# 5. test_iris_to_socket_coords: confirm subtraction of midpoint anchors
def test_iris_to_socket_coords():
    from savi.calibration import iris_to_socket_coords
    # Iris at (320, 240), inner corner at (290, 240), outer corner at (350, 240)
    # Anchor = (320, 240). Relative = (0.0, 0.0)
    rel_x, rel_y = iris_to_socket_coords(320, 240, 290, 240, 350, 240)
    assert math.isclose(rel_x, 0.0, abs_tol=1e-5)
    assert math.isclose(rel_y, 0.0, abs_tol=1e-5)

    # Iris shifted right by 5px: iris at (325, 240)
    # Anchor still (320, 240). Relative = (5.0, 0.0)
    rel_x, rel_y = iris_to_socket_coords(325, 240, 290, 240, 350, 240)
    assert math.isclose(rel_x, 5.0, abs_tol=1e-5)
    assert math.isclose(rel_y, 0.0, abs_tol=1e-5)


# 6. test_fit_polynomial_with_socket_anchors: fit with constant anchor and check recovery
def test_fit_polynomial_with_socket_anchors():
    xs_px = [100.0, 320.0, 600.0]
    ys_px = [80.0, 240.0, 400.0]
    
    sample_points_px = []
    target_positions_deg = []
    for x_px in xs_px:
        for y_px in ys_px:
            sample_points_px.append((x_px, y_px))
            x_deg = 0.1 * x_px - 5.0
            y_deg = 0.2 * y_px + 2.0
            target_positions_deg.append((x_deg, y_deg))
            
    socket_anchor_points = [
        (280.0, 240.0, 320.0, 240.0)  # anchor = (300, 240) for all 9 points
        for _ in range(9)
    ]
    
    coeffs_x, coeffs_y = fit_polynomial(
        sample_points_px, target_positions_deg,
        socket_anchor_points=socket_anchor_points
    )
    
    cal_map = CalibrationMap(
        target_positions_deg=target_positions_deg,
        sample_points_px=sample_points_px,
        poly_coeffs_x=coeffs_x,
        poly_coeffs_y=coeffs_y,
        validation_error_deg=0.0,
        validation_passed=True,
        screen_width_px=640,
        screen_height_px=480,
        viewing_distance_cm=57.0,
        fps=30.0,
        created_at="2026-06-10T12:00:00Z",
        session_id="test_session",
        socket_anchor_points=socket_anchor_points
    )
    
    cal_x, cal_y = apply_calibration(
        200.0, 150.0, cal_map,
        current_socket_anchor=(300.0, 240.0)
    )
    assert math.isclose(cal_x, 15.0, abs_tol=0.5)
    assert math.isclose(cal_y, 32.0, abs_tol=0.5)
