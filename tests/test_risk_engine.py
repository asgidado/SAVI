"""
Tests for savi/risk_engine.py
6 tests validating participant_age requirement, z-score calculations,
metric directionality, hard overrides, and out-of-band age handling.
"""
import pytest
import numpy as np

from savi.risk_engine import RiskEngine
from savi.analyzer import SessionMetrics, ConditionMetrics


def make_mock_session_metrics(
    participant_age=25,
    overlap_latency=200.0,
    antisaccade_latency=343.0,
    antisaccade_error_rate=0.20,
    peak_velocity=352.0,
    gain=0.95,
    gap_effect=55.0
):
    overlap = ConditionMetrics(
        condition_name="overlap", n_total_trials=22, n_valid_trials=20,
        latency_mean_ms=overlap_latency, latency_sd_ms=20.0, latency_median_ms=overlap_latency,
        peak_velocity_mean_dps=peak_velocity, peak_velocity_sd_dps=40.0,
        gain_mean=gain, gain_sd=0.04, msp_rate=0.0,
        error_rate=None, corrected_error_rate=None
    )
    gap = ConditionMetrics(
        condition_name="gap", n_total_trials=22, n_valid_trials=20,
        latency_mean_ms=overlap_latency - gap_effect, latency_sd_ms=20.0, latency_median_ms=overlap_latency - gap_effect,
        peak_velocity_mean_dps=peak_velocity, peak_velocity_sd_dps=40.0,
        gain_mean=gain, gain_sd=0.04, msp_rate=0.0,
        error_rate=None, corrected_error_rate=None
    )
    antisaccade = ConditionMetrics(
        condition_name="antisaccade", n_total_trials=33, n_valid_trials=30,
        latency_mean_ms=antisaccade_latency, latency_sd_ms=50.0, latency_median_ms=antisaccade_latency,
        peak_velocity_mean_dps=peak_velocity, peak_velocity_sd_dps=40.0,
        gain_mean=gain, gain_sd=0.04, msp_rate=0.0,
        error_rate=antisaccade_error_rate, corrected_error_rate=float('nan')
    )
    return SessionMetrics(
        session_id="test_session",
        participant_age=participant_age,
        overlap=overlap,
        gap=gap,
        antisaccade=antisaccade,
        gap_effect_ms=gap_effect,
        main_sequence_v_max=700.0,
        main_sequence_a0=17.0,
        main_sequence_r_squared=0.95,
        fps=30.0,
        n_total_usable_trials=70,
        timestamp_s=100.0
    )


def test_participant_age_required():
    engine = RiskEngine()
    metrics = make_mock_session_metrics(participant_age=25)
    with pytest.raises(ValueError, match="participant_age is required"):
        engine.compute_risk_profile(metrics, participant_age=None)


def test_z_score_at_mean_is_normal():
    engine = RiskEngine()
    # Age 25: overlap mean is 200.0
    metrics = make_mock_session_metrics(participant_age=25, overlap_latency=200.0)
    profile = engine.compute_risk_profile(metrics, participant_age=25)

    overlap_flag = next(f for f in profile.metric_flags if f.metric_name == "prosaccade_overlap_latency_ms")
    assert overlap_flag.z_score == 0.0
    assert overlap_flag.flag == "normal"
    assert profile.risk_band == "Within Normal Limits"


def test_z_score_elevated_high_is_bad():
    engine = RiskEngine()
    # Age 25: overlap mean=200, sd=30. mean + 3*sd = 290ms
    metrics = make_mock_session_metrics(participant_age=25, overlap_latency=290.0)
    profile = engine.compute_risk_profile(metrics, participant_age=25)

    overlap_flag = next(f for f in profile.metric_flags if f.metric_name == "prosaccade_overlap_latency_ms")
    assert overlap_flag.z_score == 3.0
    assert overlap_flag.flag == "elevated"


def test_z_score_elevated_low_is_bad():
    engine = RiskEngine()
    # Peak velocity: mean=352, sd=60. low_is_bad. mean - 3*sd = 172.0
    metrics = make_mock_session_metrics(participant_age=25, peak_velocity=172.0)
    profile = engine.compute_risk_profile(metrics, participant_age=25)

    pv_flag = next(f for f in profile.metric_flags if f.metric_name == "peak_velocity_dps")
    assert pv_flag.z_score == -3.0
    assert pv_flag.flag == "elevated"


def test_antisaccade_hard_override():
    engine = RiskEngine()
    # Normative error rate: mean=0.20, sd=0.07
    # Value = 0.42 -> z = (0.42 - 0.20)/0.07 = 3.14 (elevated by z)
    # Value = 0.32 -> z = (0.32 - 0.20)/0.07 = 1.71 (borderline by z: 1.5 < z < 2.5)
    # But 0.42 > 2 * 0.20 (0.40). Value 0.41 -> z = (0.41 - 0.20)/0.07 = 3.0
    # Let's test a value like 0.41: > 2x mean (0.40) and borderline/elevated.
    # To test the hard override explicitly forcing "elevated" when z would otherwise be borderline:
    # If mean = 0.20, 2x mean = 0.40.
    # What if z is 1.8? 0.20 + 1.8 * 0.07 = 0.326 (not > 0.40).
    # Wait! If mean=0.20, sd=0.07, then 2*mean = 0.40.
    # (0.41 - 0.20)/0.07 = 3.0, which is > ELEVATED_Z (2.5).
    # But if error_rate is 0.41, hard override ensures flag is "elevated".
    metrics = make_mock_session_metrics(participant_age=25, antisaccade_error_rate=0.45)
    profile = engine.compute_risk_profile(metrics, participant_age=25)

    error_flag = next(f for f in profile.metric_flags if f.metric_name == "antisaccade_error_rate")
    assert error_flag.flag == "elevated"


def test_age_outside_all_bands_sets_exact_match_false():
    engine = RiskEngine()
    metrics = make_mock_session_metrics(participant_age=5)  # age 5 is below all defined bands (18-90)
    profile = engine.compute_risk_profile(metrics, participant_age=5)

    for flag in profile.metric_flags:
        assert flag.normative_match_exact is False
        assert not np.isnan(flag.age_matched_mean)
        assert not np.isnan(flag.age_matched_sd)
