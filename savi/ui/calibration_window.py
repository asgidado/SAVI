"""
SAVI calibration_window.py
Implements the full-screen PySide6 calibration window.
Presents the 9-dot sequence, collects samples, validates,
saves results, and displays accuracy.
"""
import math
import time
import os
import queue
import logging
from PySide6.QtCore import QObject, Property, QPropertyAnimation, Signal, Slot, QTimer, QRectF, QPointF, Qt, QRect
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QApplication

from savi.calibration import (
    CalibrationMap, screen_fraction_to_deg, fit_polynomial,
    apply_calibration, save_calibration
)
from savi.ui.theme import COLORS, FONTS, RADIUS

logger = logging.getLogger("savi.ui.calibration_window")


class BreathingController(QObject):
    """
    Helper QObject to drive the breathing circle radius animation
    via QPropertyAnimation.
    """
    scale_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scale = 1.0

    def get_scale(self) -> float:
        return self._scale

    def set_scale(self, val: float):
        if self._scale != val:
            self._scale = val
            self.scale_changed.emit(val)

    scale = Property(float, get_scale, set_scale)


class CalibrationWindow(QWidget):
    """
    Full-screen calibration widget.
    """
    calibration_complete = Signal(CalibrationMap)

    def __init__(self, tracker, parent=None):
        super().__init__(parent)
        self.tracker = tracker
        self.queue = queue.Queue()
        self.tracker.register_queue(self.queue)

        # Setup full screen window attributes
        self.setWindowTitle("SAVI — Calibration")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # State: "CALIBRATING", "VALIDATING", "RESULTS"
        self.state = "CALIBRATING"

        # 9 Target coordinates in screen fractions (raster order)
        self.target_fractions = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        
        self.current_point_idx = 0
        self.current_point_samples = []
        self.sample_points_px = []
        self.low_quality_targets = []
        self.target_positions_deg = []
        
        self.poly_coeffs_x = []
        self.poly_coeffs_y = []
        self.validation_samples = []
        self.validation_error = 0.0
        self.cal_map = None
        self.point_start_time = 0.0

        # Animated Breathing Ring setup
        self.breathing_controller = BreathingController(self)
        self.breathing_controller.scale_changed.connect(self.update)

        self.anim = QPropertyAnimation(self.breathing_controller, b"scale")
        self.anim.setDuration(2200) # period 2.2s
        self.anim.setStartValue(1.0)
        self.anim.setKeyValueAt(0.5, 1.1)
        self.anim.setEndValue(1.0)
        self.anim.setLoopCount(-1)
        self.anim.start()

        # Build UI layout (mostly hidden until RESULTS state)
        self._init_ui()

        # Main timer to poll the tracker queue and advance dots
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_loop)

    def _init_ui(self):
        # Full screen layout to center the results box
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Centered Results dialog container
        self.results_container = QWidget(self)
        self.results_container.setFixedSize(500, 300)
        self.results_container.setObjectName("results_container")
        
        self.res_layout = QVBoxLayout(self.results_container)
        self.res_layout.setContentsMargins(32, 32, 32, 32)
        self.res_layout.setSpacing(20)

        self.lbl_title = QLabel("Calibration complete", self.results_container)
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {COLORS['text_primary']}; border: none;")

        self.lbl_accuracy = QLabel("Accuracy: --", self.results_container)
        self.lbl_accuracy.setAlignment(Qt.AlignCenter)
        self.lbl_accuracy.setStyleSheet("font-size: 16px; border: none;")

        self.lbl_warning = QLabel("", self.results_container)
        self.lbl_warning.setAlignment(Qt.AlignCenter)
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setStyleSheet("font-size: 13px; border: none;")

        self.btn_layout = QHBoxLayout()
        self.btn_layout.setSpacing(16)

        self.btn_recalibrate = QPushButton("Recalibrate", self.results_container)
        self.btn_recalibrate.clicked.connect(self.start_calibration)

        self.btn_continue = QPushButton("Continue →", self.results_container)
        self.btn_continue.clicked.connect(self.accept_calibration)

        self.btn_layout.addWidget(self.btn_recalibrate)
        self.btn_layout.addWidget(self.btn_continue)

        self.res_layout.addWidget(self.lbl_title)
        self.res_layout.addWidget(self.lbl_accuracy)
        self.res_layout.addWidget(self.lbl_warning)
        self.res_layout.addLayout(self.btn_layout)

        self.main_layout.addWidget(self.results_container)
        
        # Style sheet for results box and buttons
        self.results_container.setStyleSheet(f"""
            QWidget#results_container {{
                background-color: {COLORS['bg_surface']};
                border: 1px solid {COLORS['border_soft']};
                border-radius: {RADIUS['md']}px;
            }}
            QPushButton {{
                background-color: {COLORS['bg_raised']};
                border: 1px solid {COLORS['border_soft']};
                border-radius: {RADIUS['sm']}px;
                padding: 10px 20px;
                color: {COLORS['text_primary']};
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_surface']};
                border-color: {COLORS['accent']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent_dim']};
            }}
        """)
        
        self.results_container.hide()

    def showEvent(self, event):
        super().showEvent(event)
        self.start_calibration()

    def start_calibration(self):
        """Starts (or restarts) the 9-point calibration run."""
        self.state = "CALIBRATING"
        self.current_point_idx = 0
        self.current_point_samples = []
        self.sample_points_px = []
        self.low_quality_targets = []
        self.results_container.hide()
        
        W = self.width()
        H = self.height()
        
        # Calculate target positions in screen degrees
        px_per_cm = 37.8
        viewing_dist_cm = 57.0
        self.target_positions_deg = []
        for fx, fy in self.target_fractions:
            deg_x, deg_y = screen_fraction_to_deg(fx, fy, W, H, px_per_cm, viewing_dist_cm)
            self.target_positions_deg.append((deg_x, deg_y))
            
        self.point_start_time = time.perf_counter()
        
        # Drain queue to start fresh
        while not self.queue.empty():
            self.queue.get_nowait()
            
        self.timer.start(16)
        self.update()

    def _update_loop(self):
        now = time.perf_counter()
        
        if self.state == "CALIBRATING":
            elapsed = now - self.point_start_time
            
            # Consume new frames
            while not self.queue.empty():
                try:
                    _, gaze_frame = self.queue.get_nowait()
                    if 0.7 <= elapsed < 1.2:
                        if gaze_frame.confidence > 0.0 and not gaze_frame.blink:
                            mean_x = (gaze_frame.left_iris_x + gaze_frame.right_iris_x) / 2.0
                            mean_y = (gaze_frame.left_iris_y + gaze_frame.right_iris_y) / 2.0
                            self.current_point_samples.append((mean_x, mean_y))
                except queue.Empty:
                    break

            if elapsed >= 1.2:
                # Point complete
                if len(self.current_point_samples) >= 1:
                    mean_px = (
                        sum(p[0] for p in self.current_point_samples) / len(self.current_point_samples),
                        sum(p[1] for p in self.current_point_samples) / len(self.current_point_samples)
                    )
                else:
                    mean_px = (self.width() / 2.0, self.height() / 2.0) # default fallback
                
                if len(self.current_point_samples) < 10:
                    self.low_quality_targets.append(self.current_point_idx)
                    logger.warning(f"Target {self.current_point_idx} marked low-quality: only {len(self.current_point_samples)} frames.")
                
                self.sample_points_px.append(mean_px)
                
                self.current_point_idx += 1
                self.current_point_samples = []
                
                if self.current_point_idx < 9:
                    self.point_start_time = time.perf_counter()
                else:
                    # Move to validation phase
                    self.state = "VALIDATING"
                    self.validation_start_time = time.perf_counter()
                    self.validation_samples = []
                    
                    # Fit polynomial to get initial coeffs
                    try:
                        self.poly_coeffs_x, self.poly_coeffs_y = fit_polynomial(
                            self.sample_points_px, self.target_positions_deg
                        )
                    except Exception as e:
                        logger.error(f"Error fitting polynomial: {e}")
                        # Fallback simple coeffs
                        self.poly_coeffs_x = [0.0, 0.02, 0.0, 0.0, 0.0, 0.0]
                        self.poly_coeffs_y = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0]

            self.update()

        elif self.state == "VALIDATING":
            elapsed = now - self.validation_start_time
            
            # Consume new frames
            while not self.queue.empty():
                try:
                    _, gaze_frame = self.queue.get_nowait()
                    if 1.0 <= elapsed < 2.0:
                        if gaze_frame.confidence > 0.0 and not gaze_frame.blink:
                            mean_x = (gaze_frame.left_iris_x + gaze_frame.right_iris_x) / 2.0
                            mean_y = (gaze_frame.left_iris_y + gaze_frame.right_iris_y) / 2.0
                            self.validation_samples.append((mean_x, mean_y))
                except queue.Empty:
                    break

            if elapsed >= 2.0:
                # Validation complete
                self.state = "RESULTS"
                self.timer.stop()
                self._compute_results_and_save()
                
            self.update()

    def _compute_results_and_save(self):
        W = self.width()
        H = self.height()
        
        # Validation point is at (0.75, 0.35)
        val_target_x_deg, val_target_y_deg = screen_fraction_to_deg(0.75, 0.35, W, H, 37.8, 57.0)
        
        created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"

        self.cal_map = CalibrationMap(
            target_positions_deg=self.target_positions_deg,
            sample_points_px=self.sample_points_px,
            poly_coeffs_x=self.poly_coeffs_x,
            poly_coeffs_y=self.poly_coeffs_y,
            validation_error_deg=0.0,
            validation_passed=False,
            screen_width_px=W,
            screen_height_px=H,
            viewing_distance_cm=57.0,
            fps=30.0,
            created_at=created_at_iso,
            session_id=session_id,
            low_quality_targets=self.low_quality_targets
        )

        errors = []
        for px_x, px_y in self.validation_samples:
            cal_x, cal_y = apply_calibration(px_x, px_y, self.cal_map)
            err = math.sqrt((cal_x - val_target_x_deg)**2 + (cal_y - val_target_y_deg)**2)
            errors.append(err)

        if errors:
            self.validation_error = sum(errors) / len(errors)
        else:
            self.validation_error = 99.0 # Default bad accuracy if no samples collected

        self.cal_map.validation_error_deg = self.validation_error
        self.cal_map.validation_passed = self.validation_error < 2.0

        # Save to data directory
        os.makedirs("data", exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join("data", f"calibration_{timestamp_str}.json")
        try:
            save_calibration(self.cal_map, self.save_path)
            logger.info(f"Calibration saved to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

        # Show Results UI
        self._show_results_screen()

    def _show_results_screen(self):
        err = self.validation_error
        passed = self.cal_map.validation_passed

        pass_symbol = "● PASS" if passed else "● FAIL"
        pass_color = COLORS['ok'] if passed else COLORS['bad']

        self.lbl_accuracy.setText(
            f"Accuracy: <span style='font-weight: bold; color: {pass_color};'>{err:.1f}°</span>"
            f" &nbsp;&nbsp; <span style='color: {pass_color};'>{pass_symbol}</span>"
        )
        self.lbl_accuracy.setTextFormat(Qt.RichText)

        if passed:
            self.lbl_warning.setText("(target: < 2°)")
            self.lbl_warning.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        else:
            self.lbl_warning.setText("Accuracy above threshold. Recalibration recommended.")
            self.lbl_warning.setStyleSheet(f"color: {COLORS['bad']}; font-weight: bold; border: none;")

        self.results_container.show()

    def accept_calibration(self):
        """Emits results and closes the window."""
        self.calibration_complete.emit(self.cal_map)
        self.close()

    def _draw_status_and_progress(self, painter: QPainter, W: int, H: int):
        now = time.perf_counter()
        
        if self.state == "CALIBRATING":
            status_text = f"Point {self.current_point_idx + 1} of 9"
            progress = (self.current_point_idx + min(1.2, now - self.point_start_time) / 1.2) / 10.0
        elif self.state == "VALIDATING":
            status_text = "Validating..."
            progress = (9.0 + min(2.0, now - self.validation_start_time) / 2.0) / 10.0
        else:
            status_text = ""
            progress = 1.0

        # Draw status text at bottom center
        if status_text:
            painter.setFont(QFont(FONTS['ui'], 10))
            painter.setPen(QColor(COLORS['text_muted']))
            rect = QRect(0, H - 40, W, 20)
            painter.drawText(rect, Qt.AlignCenter, status_text)

        # Draw progress bar
        if progress > 0.0:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(COLORS['accent'])))
            painter.drawRect(0, H - 2, int(W * progress), 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Black background
        painter.fillRect(self.rect(), QColor(COLORS['bg_base']))

        W = self.width()
        H = self.height()

        if self.state in ["CALIBRATING", "VALIDATING"]:
            # Draw 9 calibration dots
            for idx, (fx, fy) in enumerate(self.target_fractions):
                cx = fx * W
                cy = fy * H
                if idx < self.current_point_idx:
                    # Complete: small faded dot (gray, 4px)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(QColor(COLORS['text_muted'])))
                    painter.drawEllipse(QPointF(cx, cy), 2.0, 2.0)
                elif idx == self.current_point_idx and self.state == "CALIBRATING":
                    # Active: white circle (12px) + breathing ring (accent border)
                    scale = self.breathing_controller.scale
                    ring_radius = 12.0 * scale
                    painter.setPen(QPen(QColor(COLORS['accent']), 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(QPointF(cx, cy), ring_radius, ring_radius)

                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(QColor(COLORS['text_primary'])))
                    painter.drawEllipse(QPointF(cx, cy), 6.0, 6.0)

            # Draw validation dot
            if self.state == "VALIDATING":
                vx = 0.75 * W
                vy = 0.35 * H
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(COLORS['accent'])))
                painter.drawEllipse(QPointF(vx, vy), 6.0, 6.0)

            # Draw status & progress bar
            self._draw_status_and_progress(painter, W, H)

    def closeEvent(self, event):
        self.tracker.unregister_queue(self.queue)
        self.timer.stop()
        event.accept()
