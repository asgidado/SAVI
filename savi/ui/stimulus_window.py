import time
import queue
import logging
import math
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget, QApplication

from savi.tracker import GazeTracker
from savi.protocol import (
    SessionRunner, BlockType, TrialState,
    BLOCK_ORDER, FIXATION_WINDOW_DEG
)
from savi.calibration import CalibrationMap
from savi.ui.theme import COLORS, FONTS, RADIUS

logger = logging.getLogger("savi.ui.stimulus_window")

# Stimulus visual constants
# Source: savi_clinical_protocol_spec.md — Stimulus Specification
BACKGROUND_COLOR  = QColor(0, 0, 0)          # pure black for clinical contrast
FIXATION_COLOR    = QColor(255, 255, 255)     # white
TARGET_COLOR      = QColor(255, 255, 255)     # white
TARGET_AMP_DEG    = 15.0                      # degrees — Hopf et al. 2018 ref amplitude

# Instruction strings
# Source: savi_clinical_protocol_spec.md — Instruction Text
INSTRUCTIONS = {
    BlockType.OVERLAP: {
        "title": "Look at the Dot",
        "body": (
            "A small dot will appear on the left or right side of the screen. "
            "Move your eyes to the dot as quickly as you can."
        ),
        "bullets": [
            "Keep your eyes on the cross until the dot appears",
            "Move as quickly as possible when you see the dot",
        ]
    },
    BlockType.GAP: {
        "title": "Look at the Dot (Fast Mode)",
        "body": (
            "Same as before, but the cross will briefly disappear "
            "before the dot appears. This short pause is normal — "
            "keep looking at the center until the dot appears."
        ),
        "bullets": [
            "The cross will disappear briefly — this is normal",
            "Look at the dot as soon as it appears",
        ]
    },
    BlockType.ANTISACCADE: {
        "title": "Look the OPPOSITE Way",
        "body": (
            "This time, when the dot appears, look to the OPPOSITE side. "
            "If the dot is on the LEFT, look RIGHT. "
            "If the dot is on the RIGHT, look LEFT."
        ),
        "bullets": [
            "Look AWAY from the dot — to the mirror position",
            "If you look at the dot by mistake, correct yourself quickly",
            "This is harder — take a moment before starting",
        ]
    }
}


class StimulusWindow(QWidget):
    """
    Full-screen stimulus presentation window for the SAVI test battery.

    Lifecycle:
        win = StimulusWindow(tracker, cal_map)
        win.show()
        # window runs until all 3 blocks complete, then emits battery_complete

    Signals:
        battery_complete(list[BlockResult]) — emitted when SessionRunner
        returns a result. Caller stores results and closes window.
    """
    battery_complete = Signal(list)

    def __init__(self, tracker: GazeTracker, cal_map: CalibrationMap, parent=None):
        super().__init__(parent)
        self.tracker = tracker
        self.cal_map = cal_map

        # Dedicated frame queue — does not share tracker_window's queue
        self._queue = queue.Queue()
        self.tracker.register_queue(self._queue)

        # Window flags — same macOS-safe approach as calibration_window
        self.setWindowFlags(
            Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setWindowTitle("SAVI — Test Battery")

        # Compute pixel geometry from calibration
        self._W = 0
        self._H = 0
        self._cx = 0        # screen center x
        self._cy = 0        # screen center y
        self._px_per_deg = 0.0
        self._target_radius_px = 0
        self._fixation_arm_px = 0
        self._target_left_x = 0
        self._target_right_x = 0

        # Display state
        # "INSTRUCTIONS" | "REST" | "RUNNING" | "COMPLETE"
        self._display_state = "INSTRUCTIONS"
        self._current_block_type = BLOCK_ORDER[0]
        self._current_block_idx = 0

        # Trial rendering state
        self._show_fixation = False
        self._show_target = False
        self._target_x = 0       # current target x in pixels
        self._onset_pending = False   # True: next paintEvent records onset

        # Session engine
        self._session = SessionRunner()

        # Rest break state
        self._rest_start_time = 0.0
        self._rest_min_duration_s = 30.0

        # Frame polling timer — 16ms ≈ 60Hz UI refresh
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_and_advance)

        # Key press to advance from instruction/rest screens
        self.setFocusPolicy(Qt.StrongFocus)

    def showEvent(self, event):
        super().showEvent(event)
        # Apply macOS geometry bypass (same pattern as calibration_window)
        screen = QApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())

        self._W = self.width()
        self._H = self.height()
        self._cx = self._W // 2
        self._cy = self._H // 2

        # Compute pixel scale from calibration map
        self._px_per_deg = self._compute_px_per_deg()
        self._target_radius_px = max(4, int(0.25 * self._px_per_deg))
        self._fixation_arm_px  = max(8, int(0.5  * self._px_per_deg))
        self._target_left_x  = self._cx - int(TARGET_AMP_DEG * self._px_per_deg)
        self._target_right_x = self._cx + int(TARGET_AMP_DEG * self._px_per_deg)

        self._display_state = "INSTRUCTIONS"
        self._current_block_type = BLOCK_ORDER[0]
        self._timer.start(16)
        self.update()

    def _compute_px_per_deg(self) -> float:
        """
        Derive pixels-per-degree from calibration map metadata.
        Uses: screen_width_px, viewing_distance_cm, and the known
        physical screen width implied by pixels_per_cm = 37.8.

        Formula: px_per_deg = pixels_per_cm * viewing_distance_cm * tan(1°)
        Source: savi_math_metrics_spec.md Formula 1
        """
        px_per_cm = 37.8
        dist_cm = self.cal_map.viewing_distance_cm
        return px_per_cm * dist_cm * math.tan(math.radians(1.0))

    def keyPressEvent(self, event):
        """
        Space advances from instruction/rest screens.
        Escape aborts the battery (for development only).
        """
        if event.key() == Qt.Key_Space:
            if self._display_state == "INSTRUCTIONS":
                if self._current_block_idx == 0:
                    self._start_battery()
                else:
                    self._display_state = "RUNNING"
                    self._show_fixation = True
                    self._show_target = False
                    self._register_onset_callback()
                    self.update()
            elif self._display_state == "REST":
                elapsed = time.perf_counter() - self._rest_start_time
                if elapsed >= self._rest_min_duration_s:
                    self._start_next_block_after_rest()
                # If < 30s, ignore spacebar

        elif event.key() == Qt.Key_Escape:
            logger.warning("Battery aborted by user (Escape key).")
            self._timer.stop()
            self.tracker.unregister_queue(self._queue)
            self.close()

    def _start_battery(self):
        """Called when user presses Space on the instruction screen."""
        self._display_state = "RUNNING"
        self._current_block_type = BLOCK_ORDER[0]
        self._current_block_idx = 0

        # Register onset callback on the first engine
        # SessionRunner creates its first BlockRunner and TrialEngine on start()
        self._session.start()
        self._register_onset_callback()

        self._show_fixation = True
        self._show_target = False
        self.update()

    def _register_onset_callback(self):
        """
        Navigate the SessionRunner → BlockRunner → TrialEngine hierarchy
        to register the onset callback on the currently active TrialEngine.

        This must be called:
          - After _session.start()
          - After each trial completes (new engine is created by BlockRunner)
          - After each block transition (new BlockRunner is created)
        """
        try:
            engine = self._session._current_runner._current_engine
            if engine is not None:
                engine.set_onset_callback(self._on_onset_signal)
        except AttributeError:
            logger.warning("Could not register onset callback — engine not ready.")

    def _on_onset_signal(self):
        """
        Called by TrialEngine at the moment it transitions to TARGET_ON.
        Sets _onset_pending = True so the next paintEvent records the timestamp.

        Do NOT record time.perf_counter() here — this fires from the polling
        timer callback, not from the paint thread. The render has not happened yet.
        """
        self._onset_pending = True
        self.update()   # trigger paintEvent immediately

    def _poll_and_advance(self):
        """
        QTimer callback at 16ms.
        Drains the frame queue and pushes each frame to the SessionRunner.
        Updates display state based on SessionRunner output.
        Also updates trial rendering state based on active engine state.
        """
        if self._display_state != "RUNNING":
            # Drain the queue to prevent frame accumulation
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            return

        changed = False

        while not self._queue.empty():
            try:
                _, gaze_frame = self._queue.get_nowait()
            except queue.Empty:
                break

            result = self._session.push_frame(gaze_frame)

            if result is not None:
                # Full battery complete
                self._display_state = "COMPLETE"
                self._show_fixation = False
                self._show_target = False
                self._timer.stop()
                self.tracker.unregister_queue(self._queue)
                self.battery_complete.emit(result)
                self.update()
                return

            # Inside the frame-push loop, check for block completion signal.
            if len(self._session._block_results) > self._current_block_idx:
                # A new block result was just added
                if self._current_block_idx < len(BLOCK_ORDER) - 1:
                    # More blocks remain — show rest break
                    self._enter_rest_break()
                    return

            # Re-register callback after each frame push
            # (BlockRunner may have created a new TrialEngine this frame)
            self._register_onset_callback()
            changed = True

        if changed:
            self._sync_display_state()
            self.update()

    def _sync_display_state(self):
        """
        Read the active TrialEngine's state and update rendering flags.

        _show_fixation: True during FIXATION_CHECK, FIXATION_HOLD,
                        and TARGET_ON for Overlap and Antisaccade.
                        False during GAP_BLANK and POST_TARGET.
        _show_target:   True only during TARGET_ON (and not a catch trial).
                        False at all other times.
        _target_x:      Set from spec.target_direction when entering TARGET_ON.
        """
        try:
            engine = self._session._current_runner._current_engine
            if engine is None:
                return

            state = engine.state
            block_type = engine.spec.block_type
            direction = engine.spec.target_direction

            # Fixation cross visibility
            if state in (TrialState.FIXATION_CHECK, TrialState.FIXATION_HOLD):
                self._show_fixation = True
                self._show_target = False

            elif state == TrialState.GAP_BLANK:
                self._show_fixation = False   # cross removed during gap
                self._show_target = False

            elif state == TrialState.TARGET_ON:
                # Overlap and Antisaccade keep fixation cross during target
                self._show_fixation = (
                    block_type in (BlockType.OVERLAP, BlockType.ANTISACCADE)
                )
                self._show_target = True
                self._target_x = (
                    self._target_right_x if direction == "right"
                    else self._target_left_x
                )

            elif state in (TrialState.POST_TARGET, TrialState.ITI):
                self._show_fixation = False
                self._show_target = False

            elif state in (TrialState.COMPLETE, TrialState.ABORTED):
                self._show_fixation = False
                self._show_target = False

        except AttributeError:
            pass

    def paintEvent(self, event):
        """
        Renders the current stimulus frame.

        CRITICAL TIMING: If _onset_pending is True, this paintEvent is
        the first render of the target stimulus. Record t_target_onset
        HERE using time.perf_counter() and write it to the active engine
        via set_target_onset(). This is the closest measurable timestamp
        to actual photon emission.

        Source: savi_clinical_protocol_spec.md
        "Record IMMEDIATELY inside the paintEvent() that renders the target dot.
        Do NOT record at QTimer callback — record at paint."
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        W = self.width()
        H = self.height()
        cx = W // 2
        cy = H // 2

        # Always fill with pure black to maximize pupil contrast (clinical override)
        painter.fillRect(self.rect(), BACKGROUND_COLOR)

        if self._display_state == "INSTRUCTIONS":
            self._paint_instructions(painter, W, H)
            return

        if self._display_state == "REST":
            self._paint_rest(painter, W, H)
            return

        if self._display_state == "COMPLETE":
            self._paint_complete(painter, W, H)
            return

        # RUNNING state — draw stimulus elements

        # ── ONSET TIMESTAMP (CRITICAL) ──────────────────────────────────
        # This block must execute BEFORE drawing anything.
        # Reason: time.perf_counter() here is as close to paint start
        # as possible. Drawing operations follow.
        if self._onset_pending:
            t_onset = time.perf_counter()
            self._onset_pending = False
            try:
                engine = self._session._current_runner._current_engine
                if engine is not None:
                    engine.set_target_onset(t_onset)
                    logger.debug(f"t_target_onset recorded in paintEvent: {t_onset:.6f}s")
            except AttributeError:
                logger.warning("Could not set target onset — engine not available.")
        # ────────────────────────────────────────────────────────────────

        # Fixation cross
        if self._show_fixation:
            arm = self._fixation_arm_px
            painter.setPen(QPen(FIXATION_COLOR, 2))
            painter.drawLine(cx - arm, cy, cx + arm, cy)
            painter.drawLine(cx, cy - arm, cx, cy + arm)

        # Target dot
        if self._show_target:
            r = self._target_radius_px
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(TARGET_COLOR))
            painter.drawEllipse(QPoint(self._target_x, cy), r, r)

    def _paint_instructions(self, painter: QPainter, W: int, H: int):
        """
        Renders the pre-block instruction screen for the current block.
        Text is white on black. Layout: title, body paragraph, bullet points,
        spacebar prompt at bottom.
        """
        block_type = self._current_block_type
        inst = INSTRUCTIONS.get(block_type, {})

        title = inst.get("title", "")
        body  = inst.get("body", "")
        bullets = inst.get("bullets", [])

        # Title
        painter.setPen(QColor(COLORS["text_primary"]))
        title_font = QFont(FONTS["ui"], 28)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            QRect(W // 4, H // 4, W // 2, 60),
            Qt.AlignCenter, title
        )

        # Body
        body_font = QFont(FONTS["ui"], 16)
        painter.setFont(body_font)
        painter.setPen(QColor(COLORS["text_secondary"]))
        painter.drawText(
            QRect(W // 4, H // 4 + 80, W // 2, 80),
            Qt.AlignCenter | Qt.TextWordWrap, body
        )

        # Bullets
        bullet_font = QFont(FONTS["ui"], 14)
        painter.setFont(bullet_font)
        painter.setPen(QColor(COLORS["text_secondary"]))
        y_offset = H // 4 + 180
        for bullet in bullets:
            painter.drawText(
                QRect(W // 4, y_offset, W // 2, 32),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"• {bullet}"
            )
            y_offset += 36

        # Spacebar prompt
        prompt_font = QFont(FONTS["ui"], 13)
        painter.setFont(prompt_font)
        painter.setPen(QColor(COLORS["text_muted"]))
        painter.drawText(
            QRect(0, H - 80, W, 40),
            Qt.AlignCenter,
            "Press Space to begin"
        )

    def _paint_rest(self, painter: QPainter, W: int, H: int):
        """
        Renders the inter-block rest screen.
        Shows elapsed rest time and enables spacebar after 30 seconds.
        """
        elapsed = time.perf_counter() - self._rest_start_time
        remaining = max(0.0, self._rest_min_duration_s - elapsed)

        painter.setPen(QColor(COLORS["text_primary"]))
        title_font = QFont(FONTS["ui"], 24)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            QRect(0, H // 3, W, 60),
            Qt.AlignCenter, "Rest Break"
        )

        painter.setPen(QColor(COLORS["text_secondary"]))
        body_font = QFont(FONTS["ui"], 15)
        painter.setFont(body_font)
        painter.drawText(
            QRect(0, H // 3 + 70, W, 40),
            Qt.AlignCenter,
            "Take a moment to relax your eyes."
        )

        if remaining > 0:
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.setFont(QFont(FONTS["mono"], 14))
            painter.drawText(
                QRect(0, H - 80, W, 40),
                Qt.AlignCenter,
                f"Continue available in {int(remaining) + 1}s"
            )
        else:
            painter.setPen(QColor(COLORS["accent"]))
            painter.setFont(QFont(FONTS["ui"], 14))
            painter.drawText(
                QRect(0, H - 80, W, 40),
                Qt.AlignCenter,
                "Press Space to continue"
            )

    def _paint_complete(self, painter: QPainter, W: int, H: int):
        """Rendered after all 3 blocks complete."""
        painter.setPen(QColor(COLORS["text_primary"]))
        painter.setFont(QFont(FONTS["ui"], 22))
        painter.drawText(self.rect(), Qt.AlignCenter,
                         "Assessment complete.\nThank you.")

    def _enter_rest_break(self):
        """Called between blocks."""
        self._display_state = "REST"
        self._rest_start_time = time.perf_counter()
        self._show_fixation = False
        self._show_target = False
        self.update()

    def _start_next_block_after_rest(self):
        """
        Called when user presses Space after rest minimum has elapsed.
        Advance block index, show instructions for next block.
        """
        self._current_block_idx += 1
        if self._current_block_idx < len(BLOCK_ORDER):
            self._current_block_type = BLOCK_ORDER[self._current_block_idx]
            self._display_state = "INSTRUCTIONS"
            self.update()

    def closeEvent(self, event):
        self.tracker.unregister_queue(self._queue)
        self._timer.stop()
        event.accept()
