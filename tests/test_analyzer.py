"""
Tests for savi/analyzer.py
11 tests covering math formulas, condition metrics, gap effect, MSP detection,
antisaccade error rate, and main sequence curve fitting.
"""
import pytest
import numpy as np

from savi.analyzer import Analyzer, ConditionMetrics
from savi.protocol import BlockType, BlockResult, TrialLog, TrialSpec
from savi.detector import DetectedSaccade
from savi.preprocessor import GazeTrace


def make_mock_trial(
    latency_ms=200.0,
    peak_velocity_dps=350.0,
    amplitude_deg=15.0,
    target_amplitude_deg=15.0,
    direction="right",
    target_direction="right",
    is_valid=True,
    is_usable=True,
    v_mag=None,
    block_type=BlockType.OVERLAP,
    trial_number=1
):
    spec = TrialSpec(
        block_type=block_type,
        trial_number=trial_number,
        block_number=1,
        target_direction=target_direction,
        target_amplitude_deg=target_amplitude_deg,
        gap_duration_ms=0.0,
        iti_duration_ms=1000.0
    )
    saccade = DetectedSaccade(
        onset_frame=10,
        offset_frame=20,
        onset_timestamp_s=1.0,
        offset_timestamp_s=1.05,
        latency_ms=latency_ms,
        duration_ms=50.0,
        peak_velocity_dps=peak_velocity_dps,
        amplitude_deg=amplitude_deg,
        direction=direction,
        is_anticipatory=False,
        is_valid=is_valid,
        rejection_reason="" if is_valid else "INVALID"
    )

    if v_mag is None:
        v_mag = np.zeros(30)
        if is_valid:
            v_mag[10:21] = peak_velocity_dps

    trace = GazeTrace(
        session_id="test",
        trial_id=trial_number,
        timestamps_s=np.linspace(0, 1, 30),
        x_deg=np.zeros(30),
        y_deg=np.zeros(30),
        x_deg_smooth=np.zeros(30),
        y_deg_smooth=np.zeros(30),
        v_x=np.zeros(30),
        v_y=np.zeros(30),
        v_mag=v_mag,
        blink_mask=np.zeros(30, dtype=bool),
        fps=30.0,
        n_blink_frames=0,
        is_usable=is_usable
    )

    return TrialLog(
        spec=spec,
        t_trial_start=0.0,
        t_fixation_acquired=0.2,
        t_target_onset=0.5,
        t_trial_end=1.0,
        frames=[],
        n_frames=30,
        trace=trace,
        saccade=saccade if is_valid else None,
        is_usable=is_usable,
        aborted=False,
        abort_reason="",
        is_correct_antisaccade=(direction != target_direction) if block_type == BlockType.ANTISACCADE else None
    )


def test_compute_gain_normal():
    analyzer = Analyzer()
    gain = analyzer._compute_gain(14.0, 15.0)
    assert 0.90 <= gain <= 1.0
    assert pytest.approx(gain, 0.001) == 14.0 / 15.0


def test_compute_gain_hypometric():
    analyzer = Analyzer()
    gain = analyzer._compute_gain(12.0, 15.0)
    assert gain < 0.85
    assert pytest.approx(gain, 0.001) == 0.80


def test_compute_gain_zero_target_returns_nan():
    analyzer = Analyzer()
    gain = analyzer._compute_gain(14.0, 0.0)
    assert np.isnan(gain)


def test_analyze_condition_empty_trials_returns_nan_metrics():
    analyzer = Analyzer()
    empty_block = BlockResult(
        block_type=BlockType.OVERLAP,
        block_number=1,
        trials=[],
        n_total=0,
        n_usable=0,
        n_aborted=0,
        t_block_start=0.0,
        t_block_end=1.0
    )
    metrics = analyzer._analyze_condition(empty_block, "overlap")
    assert metrics.n_total_trials == 0
    assert metrics.n_valid_trials == 0
    assert np.isnan(metrics.latency_mean_ms)
    assert np.isnan(metrics.latency_sd_ms)
    assert np.isnan(metrics.latency_median_ms)
    assert np.isnan(metrics.peak_velocity_mean_dps)
    assert np.isnan(metrics.peak_velocity_sd_dps)
    assert np.isnan(metrics.gain_mean)
    assert np.isnan(metrics.gain_sd)
    assert np.isnan(metrics.msp_rate)
    assert metrics.error_rate is None


def test_analyze_condition_latency_stats():
    analyzer = Analyzer()
    latencies = [200.0, 210.0, 190.0, 205.0, 195.0]
    trials = [
        make_mock_trial(latency_ms=lat, trial_number=i+1)
        for i, lat in enumerate(latencies)
    ]
    block = BlockResult(
        block_type=BlockType.OVERLAP,
        block_number=1,
        trials=trials,
        n_total=5,
        n_usable=5,
        n_aborted=0,
        t_block_start=0.0,
        t_block_end=1.0
    )
    metrics = analyzer._analyze_condition(block, "overlap")
    assert metrics.n_valid_trials == 5
    assert pytest.approx(metrics.latency_mean_ms, abs=0.01) == 200.0
    assert pytest.approx(metrics.latency_median_ms, abs=0.01) == 200.0
    assert pytest.approx(metrics.latency_sd_ms, abs=0.01) == 7.91


def test_gap_effect_normal():
    analyzer = Analyzer()
    overlap_metrics = ConditionMetrics(
        condition_name="overlap", n_total_trials=10, n_valid_trials=10,
        latency_mean_ms=200.0, latency_sd_ms=10.0, latency_median_ms=200.0,
        peak_velocity_mean_dps=350.0, peak_velocity_sd_dps=20.0,
        gain_mean=0.95, gain_sd=0.05, msp_rate=0.0,
        error_rate=None, corrected_error_rate=None
    )
    gap_metrics = ConditionMetrics(
        condition_name="gap", n_total_trials=10, n_valid_trials=10,
        latency_mean_ms=150.0, latency_sd_ms=10.0, latency_median_ms=150.0,
        peak_velocity_mean_dps=350.0, peak_velocity_sd_dps=20.0,
        gain_mean=0.95, gain_sd=0.05, msp_rate=0.0,
        error_rate=None, corrected_error_rate=None
    )
    gap_effect = analyzer._compute_gap_effect(overlap_metrics, gap_metrics)
    assert pytest.approx(gap_effect, 0.01) == 50.0


def test_gap_effect_nan_if_either_missing():
    analyzer = Analyzer()
    overlap_metrics = ConditionMetrics(
        condition_name="overlap", n_total_trials=0, n_valid_trials=0,
        latency_mean_ms=float('nan'), latency_sd_ms=float('nan'), latency_median_ms=float('nan'),
        peak_velocity_mean_dps=float('nan'), peak_velocity_sd_dps=float('nan'),
        gain_mean=float('nan'), gain_sd=float('nan'), msp_rate=float('nan'),
        error_rate=None, corrected_error_rate=None
    )
    gap_metrics = ConditionMetrics(
        condition_name="gap", n_total_trials=10, n_valid_trials=10,
        latency_mean_ms=150.0, latency_sd_ms=10.0, latency_median_ms=150.0,
        peak_velocity_mean_dps=350.0, peak_velocity_sd_dps=20.0,
        gain_mean=0.95, gain_sd=0.05, msp_rate=0.0,
        error_rate=None, corrected_error_rate=None
    )
    assert np.isnan(analyzer._compute_gap_effect(overlap_metrics, gap_metrics))


def test_msp_detection_single_peak():
    analyzer = Analyzer()
    v_mag = np.zeros(30)
    v_mag[10:16] = [0.0, 40.0, 100.0, 300.0, 80.0, 0.0]
    trial = make_mock_trial(v_mag=v_mag)
    assert analyzer._is_multiple_step_saccade(trial) is False


def test_msp_detection_double_peak():
    analyzer = Analyzer()
    v_mag = np.zeros(30)
    # Peak 1 at idx 12, Peak 2 at idx 16 (separated by >= 2 frames)
    v_mag[10:20] = [0.0, 40.0, 200.0, 40.0, 20.0, 50.0, 250.0, 60.0, 10.0, 0.0]
    trial = make_mock_trial(v_mag=v_mag)
    assert analyzer._is_multiple_step_saccade(trial) is True


def test_antisaccade_error_rate_all_correct():
    analyzer = Analyzer()
    # In antisaccade: target direction "left", correct saccade direction is "right"
    trials = [
        make_mock_trial(
            block_type=BlockType.ANTISACCADE,
            target_direction="left",
            direction="right",
            trial_number=i+1
        )
        for i in range(5)
    ]
    block = BlockResult(
        block_type=BlockType.ANTISACCADE,
        block_number=3,
        trials=trials,
        n_total=5,
        n_usable=5,
        n_aborted=0,
        t_block_start=0.0,
        t_block_end=1.0
    )
    error_rate, corrected = analyzer._compute_antisaccade_rates(block.trials)
    assert error_rate == 0.0
    assert np.isnan(corrected)


def test_main_sequence_fit_insufficient_data():
    analyzer = Analyzer()
    trials = [make_mock_trial(trial_number=1), make_mock_trial(trial_number=2)]
    block = BlockResult(
        block_type=BlockType.OVERLAP,
        block_number=1,
        trials=trials,
        n_total=2,
        n_usable=2,
        n_aborted=0,
        t_block_start=0.0,
        t_block_end=1.0
    )
    v_max, a0, r2 = analyzer._fit_main_sequence([block])
    assert np.isnan(v_max)
    assert np.isnan(a0)
    assert np.isnan(r2)
