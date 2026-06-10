"""
SAVI tracker_window.py
Implements the PySide6 QMainWindow for SAVI v0.0.1.
Features a dark premium layout, annotated camera feed, HUD telemetry panel,
live scrolling pyqtgraph gaze trace, and background worker for jitter tests.
"""
import logging
import queue
import time
import collections
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QGridLayout
)
from PySide6.QtGui import QImage, QPixmap, QFont
import pyqtgraph

import cv2
import numpy as np

from savi.tracker import GazeTracker, GazeFrame
from savi.ui.theme import COLORS, FONTS, RADIUS

logger = logging.getLogger("savi.ui.tracker_window")

class JitterWorker(QThread):
    """
    Background worker thread to run the 100-frame jitter test
    without freezing the GUI.
    """
    finished = Signal(float)

    def __init__(self, tracker: GazeTracker):
        super().__init__()
        self.tracker = tracker

    def run(self):
        try:
            rms = self.tracker.measure_rest_jitter(100)
            self.finished.emit(rms)
        except Exception as e:
            logger.error(f"Error measuring jitter in thread: {e}")
            self.finished.emit(-1.0)


class TrackerWindow(QMainWindow):
    """
    Main PySide6 application window.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAVI — Saccadic Assessment Via Imaging")
        self.setFixedSize(960, 700)

        # Initialize tracker
        # Defaults to camera 0 and relative models folder path
        self.tracker = GazeTracker(camera_index=0, model_path="models/face_landmarker.task")
        
        # Live state
        self.latest_frame = None
        self.latest_gaze_frame = None
        self.last_jitter_measurement = -1.0
        
        # Scrolling trace history: elements are (timestamp, gaze_x_deg)
        self.trace_history = collections.deque()

        # Build UI
        self._init_ui()
        
        # Set stylesheet
        self._apply_styles()

        # Start camera/tracking thread
        self.tracker.start()

        # Setup polling QTimer (16ms -> ~60 FPS update rate)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_tracker_queue)
        self.timer.start(16)

    def _init_ui(self):
        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(12)

        # 1. TOP NAV BAR
        self.top_nav = QFrame(self)
        self.top_nav.setFixedHeight(40)
        self.top_nav_layout = QHBoxLayout(self.top_nav)
        self.top_nav_layout.setContentsMargins(0, 0, 0, 0)
        
        self.nav_title = QLabel("SAVI", self)
        self.nav_title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['accent']}; letter-spacing: 1px;")
        
        self.nav_meta = QLabel("v0.0.1  [Research]", self)
        self.nav_meta.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']}; font-weight: 500;")
        self.nav_meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        self.top_nav_layout.addWidget(self.nav_title)
        self.top_nav_layout.addStretch()
        self.top_nav_layout.addWidget(self.nav_meta)
        
        self.main_layout.addWidget(self.top_nav)

        # 2. MIDDLE AREA (Camera & HUD)
        self.middle_container = QWidget(self)
        self.middle_layout = QHBoxLayout(self.middle_container)
        self.middle_layout.setContentsMargins(0, 0, 0, 0)
        self.middle_layout.setSpacing(16)

        # 2a. Camera Label (640x480)
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            f"background-color: {COLORS['bg_surface']}; "
            f"border: 1px solid {COLORS['border_soft']}; "
            f"border-radius: {RADIUS['md']}px;"
        )
        self.camera_label.setText("Starting camera feed...")
        self.middle_layout.addWidget(self.camera_label)

        # 2b. HUD Panel (right side)
        self.hud_panel = QFrame(self)
        self.hud_panel.setObjectName("hud_panel")
        self.hud_panel.setFixedWidth(288)  # 960 - 640 - 16 spacing - 16 margins
        self.hud_panel.setFixedHeight(480)
        self.hud_layout = QVBoxLayout(self.hud_panel)
        self.hud_layout.setContentsMargins(16, 16, 16, 16)
        
        self.hud_title = QLabel("TELEMETRY HUD", self)
        self.hud_title.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {COLORS['text_muted']}; letter-spacing: 0.5px;")
        self.hud_layout.addWidget(self.hud_title)
        self.hud_layout.addSpacing(8)

        # Grid of stats
        self.grid_widget = QWidget(self)
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(12)
        self.grid_layout.setVerticalSpacing(16)

        hud_items = [
            ("Gaze X", "gaze_x"),
            ("Gaze Y", "gaze_y"),
            ("L Iris", "l_iris"),
            ("R Iris", "r_iris"),
            ("FPS", "fps"),
            ("Jitter", "jitter"),
            ("Blink", "blink"),
            ("Confidence", "confidence"),
        ]

        self.hud_labels = {}
        for row, (label_text, key) in enumerate(hud_items):
            lbl_key = QLabel(label_text, self)
            lbl_key.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: 500;")
            
            lbl_val = QLabel("--", self)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            self.grid_layout.addWidget(lbl_key, row, 0)
            self.grid_layout.addWidget(lbl_val, row, 1)
            self.hud_labels[key] = lbl_val

        self.hud_layout.addWidget(self.grid_widget)
        self.hud_layout.addStretch()
        self.middle_layout.addWidget(self.hud_panel)
        
        self.main_layout.addWidget(self.middle_container)

        # 3. BOTTOM AREA (Scrolling Gaze Trace & Buttons)
        # Gaze Trace Plot Widget
        self.plot_widget = pyqtgraph.PlotWidget(self)
        self.plot_widget.setFixedHeight(100)
        self.plot_widget.setBackground(COLORS["bg_base"])
        self.plot_widget.setYRange(-20, 20)
        self.plot_widget.setXRange(-3.0, 0.0)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        
        # Hide default pyqtgraph buttons and configure styling
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.hideButtons()
        
        # Style grid axes with proper QColor parsing
        from PySide6.QtGui import QColor
        def parse_color(c):
            if c.startswith("rgba"):
                parts = c.replace("rgba(", "").replace(")", "").split(",")
                return QColor(int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip()), int(float(parts[3].strip()) * 255))
            return QColor(c)

        border_soft_q = parse_color(COLORS["border_soft"])
        text_secondary_q = parse_color(COLORS["text_secondary"])
        
        self.plot_widget.getAxis("left").setPen(border_soft_q)
        self.plot_widget.getAxis("bottom").setPen(border_soft_q)
        self.plot_widget.getAxis("left").setTextPen(text_secondary_q)
        self.plot_widget.getAxis("bottom").setTextPen(text_secondary_q)

        # Reference line at y=0 (dim white dashed)
        ref_pen = pyqtgraph.mkPen(color=parse_color(COLORS["text_muted"]), width=1, style=Qt.DashLine)
        self.plot_widget.addLine(y=0, pen=ref_pen)

        # Gaze curve
        curve_pen = pyqtgraph.mkPen(color=parse_color(COLORS["accent"]), width=2)
        self.curve = self.plot_widget.plot(pen=curve_pen)
        
        self.main_layout.addWidget(self.plot_widget)

        # Buttons Panel
        self.buttons_container = QWidget(self)
        self.buttons_layout = QHBoxLayout(self.buttons_container)
        self.buttons_layout.setContentsMargins(0, 4, 0, 0)
        self.buttons_layout.setSpacing(12)

        self.btn_log = QPushButton("Start logging CSV", self)
        self.btn_log.clicked.connect(self._toggle_csv_logging)
        
        self.btn_jitter = QPushButton("Measure jitter", self)
        self.btn_jitter.clicked.connect(self._run_jitter_measurement)
        
        self.btn_quit = QPushButton("Quit", self)
        self.btn_quit.clicked.connect(self.close)
        
        self.buttons_layout.addWidget(self.btn_log)
        self.buttons_layout.addWidget(self.btn_jitter)
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.btn_quit)
        
        self.main_layout.addWidget(self.buttons_container)

    def _apply_styles(self):
        style_sheet = f"""
        QMainWindow {{
            background-color: {COLORS['bg_base']};
        }}
        QWidget {{
            color: {COLORS['text_primary']};
            font-family: "{FONTS['ui']}";
        }}
        QPushButton {{
            background-color: {COLORS['bg_surface']};
            border: 1px solid {COLORS['border_soft']};
            border-radius: {RADIUS['sm']}px;
            padding: 8px 16px;
            color: {COLORS['text_primary']};
            font-size: 13px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {COLORS['bg_raised']};
            border-color: {COLORS['accent']};
        }}
        QPushButton:pressed {{
            background-color: {COLORS['accent_dim']};
        }}
        QPushButton:disabled {{
            color: {COLORS['text_muted']};
            border-color: {COLORS['border_faint']};
            background-color: {COLORS['bg_surface']};
        }}
        QFrame#hud_panel {{
            background-color: {COLORS['bg_surface']};
            border: 1px solid {COLORS['border_soft']};
            border-radius: {RADIUS['md']}px;
        }}
        """
        self.setStyleSheet(style_sheet)

    def _poll_tracker_queue(self):
        """Timer callback to drain frames and update GUI."""
        drained_any = False
        
        # Drain all frames currently in the queue to maintain low latency
        while not self.tracker.queue.empty():
            try:
                annotated, gaze_frame = self.tracker.queue.get_nowait()
                self.latest_frame = annotated
                self.latest_gaze_frame = gaze_frame
                
                # Append to scrolling chart history
                self.trace_history.append((gaze_frame.timestamp, gaze_frame.gaze_x_deg))
                drained_any = True
            except queue.Empty:
                break

        if drained_any and self.latest_frame is not None and self.latest_gaze_frame is not None:
            # 1. Update Video Feed
            self._display_video_frame(self.latest_frame)
            
            # 2. Update HUD reads
            self._update_hud(self.latest_gaze_frame)
            
            # 3. Update Chart
            self._update_chart(self.latest_gaze_frame.timestamp)

    def _display_video_frame(self, frame: np.ndarray):
        """Converts OpenCV BGR image to QPixmap and displays it."""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage and copy it to prevent garbage collection memory access bugs
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap)

    def _update_hud(self, f: GazeFrame):
        """Updates the numeric labels in the right-side HUD panel."""
        mono_font_family = f"'{FONTS['mono']}', {FONTS['mono_fallback']}"
        base_style = f"font-family: {mono_font_family}; font-size: 14px; font-weight: bold;"

        # X, Y coordinates
        self.hud_labels["gaze_x"].setText(f"{f.gaze_x_deg:+.1f}°")
        self.hud_labels["gaze_x"].setStyleSheet(f"{base_style} color: {COLORS['text_primary']};")
        self.hud_labels["gaze_y"].setText(f"{f.gaze_y_deg:+.1f}°")
        self.hud_labels["gaze_y"].setStyleSheet(f"{base_style} color: {COLORS['text_primary']};")

        # L / R Iris status
        l_status = "LOST" if (f.left_iris_x == 0.0 and f.left_iris_y == 0.0) or f.blink else "OK"
        r_status = "LOST" if (f.right_iris_x == 0.0 and f.right_iris_y == 0.0) or f.blink else "OK"
        
        self.hud_labels["l_iris"].setText(l_status)
        self.hud_labels["l_iris"].setStyleSheet(f"{base_style} color: {COLORS['ok'] if l_status == 'OK' else COLORS['bad']};")
        self.hud_labels["r_iris"].setText(r_status)
        self.hud_labels["r_iris"].setStyleSheet(f"{base_style} color: {COLORS['ok'] if r_status == 'OK' else COLORS['bad']};")

        # FPS Actual (>=55 green, 30-54 amber, <30 red)
        fps_val = f.fps_actual
        self.hud_labels["fps"].setText(f"{fps_val:.1f}")
        if fps_val >= 55.0:
            fps_color = COLORS["ok"]
        elif fps_val >= 30.0:
            fps_color = COLORS["warn"]
        else:
            fps_color = COLORS["bad"]
        self.hud_labels["fps"].setStyleSheet(f"{base_style} color: {fps_color};")

        # Jitter (<2px green, 2-3px amber, >3px red)
        # We display the rolling jitter, or if a dedicated jitter test has completed, we display that.
        rolling_jitter = self.tracker.get_latest_jitter()
        jitter_val = rolling_jitter if self.last_jitter_measurement < 0 else self.last_jitter_measurement
        
        self.hud_labels["jitter"].setText(f"{jitter_val:.2f}px")
        if jitter_val < 2.0:
            jit_color = COLORS["ok"]
        elif jitter_val <= 3.0:
            jit_color = COLORS["warn"]
        else:
            jit_color = COLORS["bad"]
        self.hud_labels["jitter"].setStyleSheet(f"{base_style} color: {jit_color};")

        # Blink (YES/NO)
        blink_text = "YES" if f.blink else "NO"
        blink_color = COLORS["warn"] if f.blink else COLORS["ok"]
        self.hud_labels["blink"].setText(blink_text)
        self.hud_labels["blink"].setStyleSheet(f"{base_style} color: {blink_color};")

        # Confidence (>0.85 green, 0.6-0.85 amber, <0.6 red)
        conf = f.confidence
        self.hud_labels["confidence"].setText(f"{conf:.2f}")
        if conf > 0.85:
            conf_color = COLORS["ok"]
        elif conf >= 0.6:
            conf_color = COLORS["warn"]
        else:
            conf_color = COLORS["bad"]
        self.hud_labels["confidence"].setStyleSheet(f"{base_style} color: {conf_color};")

    def _update_chart(self, current_time: float):
        """Updates the scrolling pyqtgraph live gaze trace (3s window)."""
        # Prune items older than 3 seconds
        while self.trace_history and (current_time - self.trace_history[0][0]) > 3.0:
            self.trace_history.popleft()

        if not self.trace_history:
            return

        # Prepare X and Y data
        x_data = []
        y_data = []
        for t, val in self.trace_history:
            x_data.append(t - current_time)  # Relative to current time (goes from -3.0 to 0.0)
            y_data.append(val)

        # Plot
        self.curve.setData(np.array(x_data), np.array(y_data))

    def _toggle_csv_logging(self):
        """Starts or stops the logging of GazeFrame data to a CSV file."""
        if not self.tracker.logging_active:
            filepath = self.tracker.start_csv_logging()
            self.btn_log.setText("Stop logging CSV")
            self.btn_log.setStyleSheet(
                f"background-color: {COLORS['bg_surface']}; "
                f"border: 1px solid {COLORS['bad']}; "
                f"border-radius: {RADIUS['sm']}px; "
                f"padding: 8px 16px; "
                f"color: {COLORS['bad']}; font-weight: bold;"
            )
            logger.info(f"Started CSV logging: {filepath}")
        else:
            self.tracker.stop_csv_logging()
            self.btn_log.setText("Start logging CSV")
            self._apply_styles() # Restores normal button stylesheet
            logger.info("Stopped CSV logging.")

    def _run_jitter_measurement(self):
        """Triggers the background thread to run the 100-frame jitter measurement."""
        self.btn_jitter.setEnabled(False)
        self.btn_jitter.setText("Measuring...")
        
        self.worker = JitterWorker(self.tracker)
        self.worker.finished.connect(self._on_jitter_finished)
        self.worker.start()

    def _on_jitter_finished(self, rms: float):
        """Callback when the background jitter test completes."""
        self.btn_jitter.setEnabled(True)
        self.btn_jitter.setText("Measure jitter")
        
        if rms >= 0.0:
            self.last_jitter_measurement = rms
            logger.info(f"Jitter test completed. Result: {rms:.2f}px")
        else:
            logger.error("Jitter test failed.")

    def closeEvent(self, event):
        """Ensures that the tracker background thread shuts down when closing."""
        logger.info("Closing window, stopping tracker...")
        self.timer.stop()
        self.tracker.stop()
        event.accept()
