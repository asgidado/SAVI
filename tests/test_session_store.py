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
    load_risk_profile
)
from savi.analyzer import SessionMetrics, ConditionMetrics
from savi.risk_engine import RiskProfile, MetricFlag, RiskEngine


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
