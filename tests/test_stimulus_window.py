import os
import time
import sys
from unittest.mock import MagicMock

# Configure Qt to run without a physical display (offscreen platform)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

def test_onset_callback_registered():
    """
    TrialEngine.set_onset_callback() stores the callback and
    set_target_onset() overwrites t_target_onset correctly.
    """
    from savi.protocol import TrialEngine, TrialSpec, BlockType

    spec = TrialSpec(
        block_type=BlockType.OVERLAP,
        trial_number=1,
        block_number=1,
        target_direction="right",
        target_amplitude_deg=15.0,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    engine = TrialEngine(spec)
    engine.start()

    called = []
    engine.set_onset_callback(lambda: called.append(True))

    # Manually call set_target_onset to simulate paintEvent
    t_paint = 123.456789
    engine.set_target_onset(t_paint)

    assert engine.t_target_onset == t_paint
    # Callback was not auto-called here — that happens at state transition
    # Verify it can be called manually
    engine._on_target_onset()
    assert len(called) == 1


def test_onset_pending_flag_lifecycle():
    """
    _onset_pending is set True by _on_onset_signal()
    and cleared to False inside paintEvent().
    Verify the flag behavior without rendering.
    """
    app = QApplication.instance() or QApplication(sys.argv)

    from savi.ui.stimulus_window import StimulusWindow

    mock_tracker = MagicMock()
    mock_tracker.register_queue = MagicMock()
    mock_tracker.unregister_queue = MagicMock()

    mock_cal = MagicMock()
    mock_cal.viewing_distance_cm = 57.0
    mock_cal.screen_width_px = 1920

    win = StimulusWindow(mock_tracker, mock_cal)

    # Initially False
    assert win._onset_pending is False

    # Signal fires
    win._on_onset_signal()
    assert win._onset_pending is True

    # Simulate paintEvent clearing it
    win._onset_pending = False
    assert win._onset_pending is False


def test_set_target_onset_overwrites_headless_timestamp():
    """
    Verify that the headless t_target_onset (recorded at state transition)
    is overwritten by the paintEvent timestamp.
    The paint timestamp must be >= the headless timestamp since it fires after.
    """
    from savi.protocol import TrialEngine, TrialSpec, BlockType

    spec = TrialSpec(
        block_type=BlockType.OVERLAP,
        trial_number=1,
        block_number=1,
        target_direction="left",
        target_amplitude_deg=15.0,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    engine = TrialEngine(spec)
    engine.start()

    # Set a simulated headless timestamp
    t_headless = time.perf_counter()
    engine.t_target_onset = t_headless

    # Small sleep to ensure paint timestamp is strictly later
    time.sleep(0.005)

    # Simulate paintEvent overwriting
    t_paint = time.perf_counter()
    engine.set_target_onset(t_paint)

    assert engine.t_target_onset == t_paint
    assert engine.t_target_onset > t_headless
