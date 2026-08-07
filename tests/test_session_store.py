"""
Tests for savi/session_store.py
3 tests verifying JSON roundtrips for SessionMetrics and RiskProfile,
and exception handling on missing files.
"""
import os
import pytest
import numpy as np

from savi.session_store import (
    save_session_metrics,
    load_session_metrics,
    save_risk_profile,
    load_risk_profile,
    save_raw_session,
    load_raw_session
)
from savi.analyzer import SessionMetrics, ConditionMetrics
from savi.risk_engine import RiskProfile, MetricFlag, RiskEngine
from savi.protocol import BlockResult, TrialLog, TrialSpec, BlockType
from savi.detector import DetectedSaccade
from savi.preprocessor import GazeTrace
from savi.tracker import GazeFrame


def make_test_session_metrics():
    overlap = ConditionMetrics(
        condition_name="overlap", n_total_trials=22, n_valid_trials=20,
        latency_mean_ms=200.0, latency_sd_ms=15.0, latency_median_ms=198.0,
        peak_velocity_mean_dps=350.0, peak_velocity_sd_dps=30.0,
        gain_mean=0.95, gain_sd=0.03, msp_rate=0.05,
        error_rate=None, corrected_error_rate=None
    )
    gap = ConditionMetrics(
        condition_name="gap", n_total_trials=22, n_valid_trials=20,
        latency_mean_ms=150.0, latency_sd_ms=12.0, latency_median_ms=149.0,
        peak_velocity_mean_dps=355.0, peak_velocity_sd_dps=28.0,
        gain_mean=0.96, gain_sd=0.04, msp_rate=0.0,
        error_rate=None, corrected_error_rate=None
    )
    antisaccade = ConditionMetrics(
        condition_name="antisaccade", n_total_trials=33, n_valid_trials=30,
        latency_mean_ms=340.0, latency_sd_ms=45.0, latency_median_ms=335.0,
        peak_velocity_mean_dps=345.0, peak_velocity_sd_dps=32.0,
        gain_mean=0.93, gain_sd=0.05, msp_rate=0.10,
        error_rate=0.15, corrected_error_rate=float('nan')
    )
    return SessionMetrics(
        session_id="session_test_123",
        participant_age=30,
        overlap=overlap,
        gap=gap,
        antisaccade=antisaccade,
        gap_effect_ms=50.0,
        main_sequence_v_max=710.0,
        main_sequence_a0=16.5,
        main_sequence_r_squared=0.94,
        fps=30.0,
        n_total_usable_trials=70,
        timestamp_s=123456.78
    )


def test_session_metrics_json_roundtrip(tmp_path):
    metrics = make_test_session_metrics()
    path = save_session_metrics(metrics, directory=str(tmp_path))
    assert os.path.exists(path)

    loaded = load_session_metrics(path)
    assert loaded.session_id == metrics.session_id
    assert loaded.participant_age == metrics.participant_age
    assert loaded.overlap.latency_mean_ms == metrics.overlap.latency_mean_ms
    assert loaded.gap.latency_mean_ms == metrics.gap.latency_mean_ms
    assert loaded.antisaccade.error_rate == metrics.antisaccade.error_rate
    assert np.isnan(loaded.antisaccade.corrected_error_rate)
    assert loaded.gap_effect_ms == metrics.gap_effect_ms
    assert loaded.main_sequence_v_max == metrics.main_sequence_v_max


def test_risk_profile_json_roundtrip(tmp_path):
    metrics = make_test_session_metrics()
    engine = RiskEngine()
    profile = engine.compute_risk_profile(metrics, participant_age=30)

    path = save_risk_profile(profile, directory=str(tmp_path))
    assert os.path.exists(path)

    loaded = load_risk_profile(path)
    assert loaded.session_id == profile.session_id
    assert loaded.participant_age == profile.participant_age
    assert loaded.risk_band == profile.risk_band
    assert len(loaded.metric_flags) == len(profile.metric_flags)
    assert loaded.n_borderline == profile.n_borderline
    assert loaded.n_elevated == profile.n_elevated


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_session_metrics("non_existent_file.json")


def test_raw_session_json_roundtrip(tmp_path):
    spec = TrialSpec(
        block_type=BlockType.OVERLAP,
        trial_number=1,
        block_number=1,
        target_direction="right",
        target_amplitude_deg=15.0,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    frame = GazeFrame(
        timestamp=1.0, frame_idx=1,
        gaze_x_deg=0.5, gaze_y_deg=-0.5,
        left_iris_x=10.0, left_iris_y=10.0,
        right_iris_x=20.0, right_iris_y=10.0,
        velocity_deg_s=0.0, blink=False, confidence=0.9, fps_actual=30.0,
        cal_x_deg=0.5, cal_y_deg=-0.5, calibration_applied=True
    )
    trace = GazeTrace(
        session_id="test",
        trial_id=1,
        timestamps_s=np.array([0.0, 0.033, 0.066]),
        x_deg=np.array([0.0, 0.5, 1.0]),
        y_deg=np.array([0.0, 0.0, 0.0]),
        x_deg_smooth=np.array([0.0, 0.5, 1.0]),
        y_deg_smooth=np.array([0.0, 0.0, 0.0]),
        v_x=np.array([0.0, 15.0, 30.0]),
        v_y=np.array([0.0, 0.0, 0.0]),
        v_mag=np.array([0.0, 15.0, 30.0]),
        blink_mask=np.array([False, False, False]),
        fps=30.0,
        n_blink_frames=0,
        is_usable=True
    )
    saccade = DetectedSaccade(
        onset_frame=1, offset_frame=2,
        onset_timestamp_s=0.033, offset_timestamp_s=0.066,
        latency_ms=200.0, duration_ms=33.0,
        peak_velocity_dps=350.0, amplitude_deg=15.0,
        direction="right", is_anticipatory=False,
        is_valid=True, rejection_reason=""
    )
    trial = TrialLog(
        spec=spec,
        t_trial_start=0.0,
        t_fixation_acquired=0.2,
        t_target_onset=0.5,
        t_trial_end=1.0,
        frames=[frame],
        n_frames=1,
        trace=trace,
        saccade=saccade,
        is_usable=True,
        aborted=False,
        abort_reason="",
        is_correct_antisaccade=None
    )
    block = BlockResult(
        block_type=BlockType.OVERLAP,
        block_number=1,
        trials=[trial],
        n_total=1,
        n_usable=1,
        n_aborted=0,
        t_block_start=0.0,
        t_block_end=1.0
    )

    path = save_raw_session([block], "session_raw_test", directory=str(tmp_path))
    assert os.path.exists(path)

    loaded = load_raw_session(path)
    assert len(loaded) == 1
    loaded_block = loaded[0]
    assert loaded_block.block_type == BlockType.OVERLAP
    assert len(loaded_block.trials) == 1

    loaded_trial = loaded_block.trials[0]
    assert loaded_trial.spec.target_direction == "right"
    assert loaded_trial.saccade.latency_ms == 200.0
    assert isinstance(loaded_trial.trace.x_deg_smooth, np.ndarray)
    assert np.allclose(loaded_trial.trace.x_deg_smooth, trace.x_deg_smooth)
    assert isinstance(loaded_trial.frames[0], GazeFrame)
    assert loaded_trial.frames[0].timestamp == 1.0


def test_raw_session_handles_none_trace_and_saccade(tmp_path):
    spec = TrialSpec(
        block_type=BlockType.OVERLAP,
        trial_number=1,
        block_number=1,
        target_direction="right",
        target_amplitude_deg=15.0,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    trial_unusable = TrialLog(
        spec=spec,
        t_trial_start=0.0,
        t_fixation_acquired=0.0,
        t_target_onset=0.5,
        t_trial_end=1.0,
        frames=[],
        n_frames=0,
        trace=None,
        saccade=None,
        is_usable=False,
        aborted=True,
        abort_reason="FIXATION_LOST",
        is_correct_antisaccade=None
    )
    block = BlockResult(
        block_type=BlockType.OVERLAP,
        block_number=1,
        trials=[trial_unusable],
        n_total=1,
        n_usable=0,
        n_aborted=1,
        t_block_start=0.0,
        t_block_end=1.0
    )
    path = save_raw_session([block], "session_unusable", directory=str(tmp_path))
    loaded = load_raw_session(path)
    assert loaded[0].trials[0].trace is None
    assert loaded[0].trials[0].saccade is None
    assert loaded[0].trials[0].is_usable is False

