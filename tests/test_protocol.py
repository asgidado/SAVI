import time
import pytest
from savi.tracker import GazeFrame
from savi.preprocessor import Preprocessor
from savi.detector import DetectedSaccade
from savi.protocol import (
    BlockRunner, BlockType, TrialSpec, TrialState, TrialEngine, SessionRunner,
    TARGET_AMPLITUDES_DEG, ITI_DURATION_MS_MIN, ITI_DURATION_MS_MAX,
    GAP_DURATION_MS, TrialLog
)

def make_frame(
    timestamp: float,
    frame_idx: int,
    gaze_x: float = 0.0,
    gaze_y: float = 0.0,
    cal_x: float | None = None,
    cal_y: float | None = None,
    blink: bool = False,
    confidence: float = 0.95
) -> GazeFrame:
    return GazeFrame(
        timestamp=timestamp,
        frame_idx=frame_idx,
        gaze_x_deg=gaze_x,
        gaze_y_deg=gaze_y,
        left_iris_x=320.0,
        left_iris_y=240.0,
        right_iris_x=320.0,
        right_iris_y=240.0,
        velocity_deg_s=0.0,
        blink=blink,
        confidence=confidence,
        fps_actual=30.0,
        cal_x_deg=cal_x if cal_x is not None else gaze_x,
        cal_y_deg=cal_y if cal_y is not None else gaze_y,
        calibration_applied=True
    )

def test_trial_spec_generation():
    runner = BlockRunner(BlockType.OVERLAP, block_number=1)
    for i in range(1, 11):
        spec = runner._generate_spec(trial_number=i)
        assert spec.block_type == BlockType.OVERLAP
        assert spec.target_direction in ("left", "right")
        assert spec.target_amplitude_deg in TARGET_AMPLITUDES_DEG
        assert spec.gap_duration_ms == 0.0
        assert spec.iti_duration_ms >= ITI_DURATION_MS_MIN
        assert spec.iti_duration_ms <= ITI_DURATION_MS_MAX

def test_gap_block_has_gap_duration():
    runner = BlockRunner(BlockType.GAP, block_number=2)
    spec = runner._generate_spec(trial_number=1)
    assert spec.gap_duration_ms == GAP_DURATION_MS

def test_antisaccade_has_no_gap():
    runner = BlockRunner(BlockType.ANTISACCADE, block_number=3)
    spec = runner._generate_spec(trial_number=1)
    assert spec.gap_duration_ms == 0.0

def test_fixation_check_requires_stable_gaze():
    spec = TrialSpec(
        block_type=BlockType.OVERLAP,
        trial_number=1,
        block_number=1,
        target_direction="right",
        target_amplitude_deg=10.0,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    engine = TrialEngine(spec)
    engine.start()

    # Push 10 frames outside the fixation window (gaze_x=5.0 > 2.5)
    for i in range(10):
        frame = make_frame(timestamp=i * 0.033, frame_idx=i, gaze_x=5.0)
        engine.push_frame(frame)
        assert engine.state == TrialState.FIXATION_CHECK

    # Push 1 frame inside window (gaze_x=0.0)
    frame = make_frame(timestamp=10 * 0.033, frame_idx=10, gaze_x=0.0)
    engine.push_frame(frame)
    assert engine.state == TrialState.FIXATION_HOLD

def test_fixation_broken_resets_to_check():
    spec = TrialSpec(
        block_type=BlockType.OVERLAP,
        trial_number=1,
        block_number=1,
        target_direction="right",
        target_amplitude_deg=10.0,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    engine = TrialEngine(spec)
    engine.start()

    # Push 5 frames inside window to acquire and hold fixation
    for i in range(5):
        frame = make_frame(timestamp=i * 0.033, frame_idx=i, gaze_x=0.0)
        engine.push_frame(frame)
    assert engine.state == TrialState.FIXATION_HOLD

    # Push 1 frame outside window to break fixation
    frame = make_frame(timestamp=5 * 0.033, frame_idx=5, gaze_x=5.0)
    engine.push_frame(frame)
    assert engine.state == TrialState.FIXATION_CHECK

def test_overlap_trial_skips_gap(monkeypatch):
    time_values = [0.0, 0.0, 0.9]
    iterator = iter(time_values)
    
    def mock_perf_counter():
        try:
            return next(iterator)
        except StopIteration:
            return 1.0

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    spec = TrialSpec(
        block_type=BlockType.OVERLAP,
        trial_number=1,
        block_number=1,
        target_direction="right",
        target_amplitude_deg=10.0,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    engine = TrialEngine(spec)
    engine.start()
    assert engine.state == TrialState.FIXATION_CHECK

    # 1st push_frame transitions to FIXATION_HOLD
    f1 = make_frame(timestamp=0.0, frame_idx=1, gaze_x=0.0)
    engine.push_frame(f1)
    assert engine.state == TrialState.FIXATION_HOLD

    # 2nd push_frame: time advanced to 0.9s, hold completed.
    # Overlap trial must skip GAP_BLANK and go directly to TARGET_ON.
    f2 = make_frame(timestamp=0.033, frame_idx=2, gaze_x=0.0)
    engine.push_frame(f2)
    assert engine.state == TrialState.TARGET_ON

def test_gap_trial_enters_gap_blank(monkeypatch):
    time_values = [0.0, 0.0, 0.9]
    iterator = iter(time_values)
    
    def mock_perf_counter():
        try:
            return next(iterator)
        except StopIteration:
            return 1.0

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    spec = TrialSpec(
        block_type=BlockType.GAP,
        trial_number=1,
        block_number=2,
        target_direction="right",
        target_amplitude_deg=10.0,
        gap_duration_ms=200.0,
        iti_duration_ms=1000.0
    )
    engine = TrialEngine(spec)
    engine.start()
    assert engine.state == TrialState.FIXATION_CHECK

    # 1st push_frame transitions to FIXATION_HOLD
    f1 = make_frame(timestamp=0.0, frame_idx=1, gaze_x=0.0)
    engine.push_frame(f1)
    assert engine.state == TrialState.FIXATION_HOLD

    # 2nd push_frame: time advanced to 0.9s, hold completed.
    # Gap trial must transition to GAP_BLANK.
    f2 = make_frame(timestamp=0.033, frame_idx=2, gaze_x=0.0)
    engine.push_frame(f2)
    assert engine.state == TrialState.GAP_BLANK

def test_antisaccade_correctness_scoring(monkeypatch):
    # Mock preprocessor to return a usable GazeTrace
    class MockTrace:
        is_usable = True

    monkeypatch.setattr(Preprocessor, "process_trial", lambda *args, **kwargs: MockTrace())

    # Case A: target_direction="right", saccade.direction="left"
    saccade_a = DetectedSaccade(
        onset_frame=10,
        offset_frame=15,
        onset_timestamp_s=1.2,
        offset_timestamp_s=1.3,
        latency_ms=200.0,
        duration_ms=100.0,
        peak_velocity_dps=150.0,
        amplitude_deg=10.0,
        direction="left",
        is_anticipatory=False,
        is_valid=True,
        rejection_reason=""
    )

    monkeypatch.setattr("savi.protocol.detect_primary_saccade", lambda *args, **kwargs: saccade_a)

    spec_a = TrialSpec(
        block_type=BlockType.ANTISACCADE,
        trial_number=1,
        block_number=3,
        target_direction="right",
        target_amplitude_deg=10.0,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    engine_a = TrialEngine(spec_a)
    # Put at least 5 frames so _finalize runs processing
    engine_a.frames = [object()] * 5
    log_a = engine_a._finalize()
    assert log_a.is_correct_antisaccade is True

    # Case B: target_direction="right", saccade.direction="right"
    saccade_b = DetectedSaccade(
        onset_frame=10,
        offset_frame=15,
        onset_timestamp_s=1.2,
        offset_timestamp_s=1.3,
        latency_ms=200.0,
        duration_ms=100.0,
        peak_velocity_dps=150.0,
        amplitude_deg=10.0,
        direction="right",
        is_anticipatory=False,
        is_valid=True,
        rejection_reason=""
    )

    monkeypatch.setattr("savi.protocol.detect_primary_saccade", lambda *args, **kwargs: saccade_b)

    engine_b = TrialEngine(spec_a)
    engine_b.frames = [object()] * 5
    log_b = engine_b._finalize()
    assert log_b.is_correct_antisaccade is False


def test_session_runner_integration(monkeypatch):
    # Mock preprocessor to avoid actual processing of short traces
    class MockTrace:
        is_usable = False

    monkeypatch.setattr(Preprocessor, "process_trial", lambda *args, **kwargs: MockTrace())

    t = [0.0]
    def mock_perf_counter():
        t[0] += 10.0
        return t[0]

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    session = SessionRunner()
    session.start()

    frame_idx = 1
    result = None

    # Push frames to complete all 77 trials (22 Overlap, 22 Gap, 33 Antisaccade)
    for _ in range(1000):
        frame = make_frame(timestamp=t[0], frame_idx=frame_idx, gaze_x=0.0)
        frame_idx += 1
        result = session.push_frame(frame)
        if result is not None:
            break

    assert result is not None
    assert len(result) == 3
    assert result[0].block_type == BlockType.OVERLAP
    assert result[1].block_type == BlockType.GAP
    assert result[2].block_type == BlockType.ANTISACCADE
    assert len(result[0].trials) == 22
    assert len(result[1].trials) == 22
    assert len(result[2].trials) == 33

