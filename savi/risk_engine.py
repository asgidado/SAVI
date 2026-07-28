"""
SAVI risk_engine.py
Age-matched normative deviation scoring and composite risk band calculation.
Source: savi_math_metrics_spec.md Formula 15; savi_architecture_spec.md Module 6.
"""
from dataclasses import dataclass
import numpy as np
import time
import logging

logger = logging.getLogger("savi.risk_engine")

# NORMATIVE_DB — hardcoded from peer-reviewed sources.
# Each entry cites its source inline — non-negotiable for paper
# reproducibility. Source: savi_math_metrics_spec.md Formula 15;
# savi_architecture_spec.md Module 6.
#
# Structure: metric_name -> {(age_min, age_max): {"mean", "sd", "source"}}
# Direction: "high_is_bad" | "low_is_bad" — determines interpretation.

NORMATIVE_DB = {
    "prosaccade_overlap_latency_ms": {
        (18, 40): {"mean": 200.0, "sd": 30.0, "source": "Imaoka et al. 2020"},
        (41, 60): {"mean": 215.0, "sd": 35.0, "source": "Hopf et al. 2018"},
        (61, 90): {"mean": 235.0, "sd": 40.0, "source": "Hopf et al. 2018"},
    },
    "antisaccade_latency_ms": {
        (18, 40): {"mean": 343.0, "sd": 76.0, "source": "Imaoka et al. 2020"},
        (41, 90): {"mean": 370.0, "sd": 85.0, "source": "Hopf et al. 2018"},
    },
    "antisaccade_error_rate": {
        (18, 90): {"mean": 0.20, "sd": 0.07, "source": "Fischer et al. 1997"},
    },
    "peak_velocity_dps": {
        (18, 90): {"mean": 352.0, "sd": 60.0, "source": "Hopf et al. 2018"},
    },
    "gain": {
        (18, 90): {"mean": 0.95, "sd": 0.05, "source": "Hopf et al. 2018"},
    },
    "gap_effect_ms": {
        (18, 90): {"mean": 55.0, "sd": 20.0, "source": "Fischer et al. 1997"},
    },
}

# Direction of abnormality per metric.
# Source: savi_math_metrics_spec.md Formula 15.
METRIC_DIRECTION = {
    "prosaccade_overlap_latency_ms": "high_is_bad",
    "antisaccade_latency_ms":        "high_is_bad",
    "antisaccade_error_rate":        "high_is_bad",
    "peak_velocity_dps":             "low_is_bad",
    "gain":                          "low_is_bad",
    "gap_effect_ms":                 "low_is_bad",
}

BORDERLINE_Z = 1.5
ELEVATED_Z   = 2.5


@dataclass
class MetricFlag:
    metric_name: str
    user_value: float
    unit: str
    age_matched_mean: float
    age_matched_sd: float
    z_score: float
    flag: str                    # "normal" | "borderline" | "elevated" | "insufficient_data"
    source_citation: str
    normative_match_exact: bool   # False if age fell outside all bands


@dataclass
class RiskProfile:
    session_id: str
    participant_age: int
    risk_band: str                # "Within Normal Limits" | "Borderline" | "Elevated"
    metric_flags: list            # list[MetricFlag]
    n_borderline: int
    n_elevated: int
    timestamp_s: float


class RiskEngine:
    """
    Compares SessionMetrics to age-matched normative data and produces
    a structured, per-metric deviation profile plus a composite risk band.

    participant_age is required at call time — this class never assumes
    or defaults an age band.
    """

    def compute_risk_profile(self, metrics, participant_age: int) -> RiskProfile:
        """
        Args:
            metrics: SessionMetrics from analyzer.py
            participant_age: REQUIRED. Raises ValueError if None.

        Returns:
            RiskProfile with per-metric z-scores, flags, and composite band.
        """
        if participant_age is None:
            raise ValueError(
                "participant_age is required — RiskEngine never assumes "
                "an age band."
            )

        flags = []

        # Prosaccade overlap latency
        flags.append(self._score_metric(
            "prosaccade_overlap_latency_ms",
            metrics.overlap.latency_mean_ms,
            "ms", participant_age
        ))

        # Antisaccade latency
        flags.append(self._score_metric(
            "antisaccade_latency_ms",
            metrics.antisaccade.latency_mean_ms,
            "ms", participant_age
        ))

        # Antisaccade error rate — with hard override
        error_flag = self._score_metric(
            "antisaccade_error_rate",
            metrics.antisaccade.error_rate,
            "proportion", participant_age
        )
        # Hard override: source savi_math_metrics_spec.md Formula 15
        if (metrics.antisaccade.error_rate is not None
                and not np.isnan(metrics.antisaccade.error_rate)
                and error_flag.age_matched_mean > 0
                and metrics.antisaccade.error_rate > 2 * error_flag.age_matched_mean):
            error_flag.flag = "elevated"
        flags.append(error_flag)

        # Peak velocity (use overlap condition as reference, 15° amplitude)
        flags.append(self._score_metric(
            "peak_velocity_dps",
            metrics.overlap.peak_velocity_mean_dps,
            "°/s", participant_age
        ))

        # Gain
        flags.append(self._score_metric(
            "gain",
            metrics.overlap.gain_mean,
            "ratio", participant_age
        ))

        # Gap effect
        flags.append(self._score_metric(
            "gap_effect_ms",
            metrics.gap_effect_ms,
            "ms", participant_age
        ))

        n_borderline = sum(1 for f in flags if f.flag == "borderline")
        n_elevated = sum(1 for f in flags if f.flag == "elevated")

        risk_band = self._compute_risk_band(n_elevated, n_borderline)

        return RiskProfile(
            session_id=metrics.session_id,
            participant_age=participant_age,
            risk_band=risk_band,
            metric_flags=flags,
            n_borderline=n_borderline,
            n_elevated=n_elevated,
            timestamp_s=time.perf_counter()
        )

    def _score_metric(
        self,
        metric_name: str,
        user_value: float | None,
        unit: str,
        participant_age: int
    ) -> MetricFlag:
        """
        Look up normative data for the given metric and age, compute
        z-score, and determine flag.

        If user_value is NaN or None (metric could not be computed), return
        a MetricFlag with flag="insufficient_data" sentinel in the flag field so
        the risk band computation and the UI can distinguish "scored
        as normal" from "could not be scored."
        """
        band_data, exact_match = self._lookup_normative_band(
            metric_name, participant_age
        )

        if user_value is None or np.isnan(user_value):
            return MetricFlag(
                metric_name=metric_name,
                user_value=float('nan') if user_value is None else user_value,
                unit=unit,
                age_matched_mean=band_data["mean"],
                age_matched_sd=band_data["sd"],
                z_score=float('nan'),
                flag="insufficient_data",
                source_citation=band_data["source"],
                normative_match_exact=exact_match
            )

        z = self._compute_z_score(user_value, band_data["mean"], band_data["sd"])

        direction = METRIC_DIRECTION[metric_name]
        # For "low_is_bad" metrics, a negative z (below mean) is the
        # abnormal direction. Flip sign for threshold comparison so
        # abs(effective_z) consistently represents "badness".
        if direction == "low_is_bad":
            effective_z = -z
        else:
            effective_z = z

        if abs(effective_z) > ELEVATED_Z:
            flag = "elevated"
        elif abs(effective_z) > BORDERLINE_Z:
            flag = "borderline"
        else:
            flag = "normal"

        return MetricFlag(
            metric_name=metric_name,
            user_value=user_value,
            unit=unit,
            age_matched_mean=band_data["mean"],
            age_matched_sd=band_data["sd"],
            z_score=z,
            flag=flag,
            source_citation=band_data["source"],
            normative_match_exact=exact_match
        )

    def _lookup_normative_band(
        self, metric_name: str, age: int
    ) -> tuple[dict, bool]:
        """
        Find the normative band matching the given age for a metric.

        Returns (band_data, exact_match).
        exact_match = False if age fell outside all defined bands and
        the nearest band was used as a fallback.
        """
        bands = NORMATIVE_DB[metric_name]

        for (age_min, age_max), data in bands.items():
            if age_min <= age <= age_max:
                return data, True

        # No exact band match — find nearest band by distance to range
        best_band = None
        best_distance = float('inf')
        for (age_min, age_max), data in bands.items():
            if age < age_min:
                distance = age_min - age
            else:
                distance = age - age_max
            if distance < best_distance:
                best_distance = distance
                best_band = data

        logger.warning(
            f"Age {age} outside all normative bands for {metric_name}. "
            f"Using nearest band as approximation "
            f"(source: {best_band['source']})."
        )
        return best_band, False

    def _compute_z_score(self, user_value: float, mean: float, sd: float) -> float:
        """
        Formula 15 — z-Score Normative Deviation.
        Source: savi_math_metrics_spec.md Formula 15.
        """
        if sd == 0:
            return float('nan')
        return (user_value - mean) / sd

    def _compute_risk_band(self, n_elevated: int, n_borderline: int) -> str:
        """
        Composite risk band logic.
        Source: savi_architecture_spec.md Module 6.
        """
        if n_elevated >= 2 or n_borderline >= 3:
            return "Elevated"
        elif n_elevated == 1 or n_borderline == 2:
            return "Borderline"
        else:
            return "Within Normal Limits"
