"""
SAVI analyzer.py
Extracts clinical saccade metrics from completed test battery block results.
Source: savi_math_metrics_spec.md
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import time
import logging

logger = logging.getLogger("savi.analyzer")

# Constants — Source: savi_math_metrics_spec.md
ONSET_THRESHOLD_DPS   = 30.0    # for MSP peak detection (Formula 13)
CORRECTION_WINDOW_MS  = 800.0   # Formula 12


@dataclass
class ConditionMetrics:
    """
    Aggregated metrics for one paradigm condition
    (overlap, gap, or antisaccade).
    """
    condition_name: str            # "overlap" | "gap" | "antisaccade"
    n_total_trials: int
    n_valid_trials: int            # trials with a valid detected saccade

    latency_mean_ms: float
    latency_sd_ms: float
    latency_median_ms: float

    peak_velocity_mean_dps: float
    peak_velocity_sd_dps: float

    gain_mean: float
    gain_sd: float

    msp_rate: float                 # NaN if no valid saccades

    # Antisaccade-only fields (NaN / None for overlap and gap)
    error_rate: float | None
    corrected_error_rate: float | None


@dataclass
class SessionMetrics:
    """
    Complete session-level metric summary.
    Produced by Analyzer.analyze_session().
    """
    session_id: str
    participant_age: int

    overlap: ConditionMetrics
    gap: ConditionMetrics
    antisaccade: ConditionMetrics

    gap_effect_ms: float            # overlap_latency_mean - gap_latency_mean

    main_sequence_v_max: float
    main_sequence_a0: float
    main_sequence_r_squared: float

    fps: float                       # 30.0
    n_total_usable_trials: int
    timestamp_s: float


@dataclass
class TrialOrderSeries:
    """
    Per-trial metric values in trial order, for one condition.
    Used to detect within-session drift (e.g. calibration degrading
    over a long battery) by plotting these against trial_numbers.
    """
    condition_name: str
    trial_numbers: list          # list[int], 1-indexed within block
    latencies_ms: list           # list[float], NaN for invalid trials
    peak_velocities_dps: list    # list[float], NaN for invalid trials
    gains: list                  # list[float], NaN for invalid trials
    is_valid: list                # list[bool] — was this trial usable+valid



class Analyzer:
    """
    Extracts all clinical saccade metrics from a completed test battery.
    Operates entirely on in-memory BlockResult/TrialLog objects produced
    by protocol.py. Does not touch the camera, tracker, or UI.
    """

    def analyze_session(
        self,
        block_results: list,     # list[BlockResult], length 3
        session_id: str,
        participant_age: int
    ) -> SessionMetrics:
        """
        Main entry point. Maps each BlockResult to its condition and
        computes all metrics.

        Raises:
            ValueError if participant_age is not provided, or if
            block_results does not contain exactly 3 blocks in the
            expected order (Overlap, Gap, Antisaccade).
        """
        if participant_age is None:
            raise ValueError(
                "participant_age is required — SAVI does not assume "
                "an age band. Callers must supply this explicitly."
            )

        if len(block_results) != 3:
            raise ValueError(
                f"Expected 3 block results (Overlap, Gap, Antisaccade), "
                f"got {len(block_results)}."
            )

        # Map blocks by type (do not assume order — check block_type)
        from savi.protocol import BlockType
        blocks_by_type = {br.block_type: br for br in block_results}

        for bt in (BlockType.OVERLAP, BlockType.GAP, BlockType.ANTISACCADE):
            if bt not in blocks_by_type:
                raise ValueError(f"Missing block result for {bt.value}")

        overlap_metrics = self._analyze_condition(
            blocks_by_type[BlockType.OVERLAP], "overlap"
        )
        gap_metrics = self._analyze_condition(
            blocks_by_type[BlockType.GAP], "gap"
        )
        antisaccade_metrics = self._analyze_condition(
            blocks_by_type[BlockType.ANTISACCADE], "antisaccade"
        )

        gap_effect = self._compute_gap_effect(overlap_metrics, gap_metrics)

        v_max, a0, r2 = self._fit_main_sequence(block_results)

        n_usable = sum(br.n_usable for br in block_results)

        return SessionMetrics(
            session_id=session_id,
            participant_age=participant_age,
            overlap=overlap_metrics,
            gap=gap_metrics,
            antisaccade=antisaccade_metrics,
            gap_effect_ms=gap_effect,
            main_sequence_v_max=v_max,
            main_sequence_a0=a0,
            main_sequence_r_squared=r2,
            fps=30.0,
            n_total_usable_trials=n_usable,
            timestamp_s=time.perf_counter()
        )

    def _analyze_condition(self, block_result, condition_name: str) -> ConditionMetrics:
        """
        Compute all per-condition metrics from a BlockResult's TrialLogs.

        Uses only trials where:
          trial.is_usable == True AND trial.saccade is not None
          AND trial.saccade.is_valid == True
        """
        valid_trials = [
            t for t in block_result.trials
            if t.is_usable and t.saccade is not None and t.saccade.is_valid
        ]

        n_total = len(block_result.trials)
        n_valid = len(valid_trials)

        if n_valid == 0:
            return ConditionMetrics(
                condition_name=condition_name,
                n_total_trials=n_total,
                n_valid_trials=0,
                latency_mean_ms=float('nan'),
                latency_sd_ms=float('nan'),
                latency_median_ms=float('nan'),
                peak_velocity_mean_dps=float('nan'),
                peak_velocity_sd_dps=float('nan'),
                gain_mean=float('nan'),
                gain_sd=float('nan'),
                msp_rate=float('nan'),
                error_rate=float('nan') if condition_name == "antisaccade" else None,
                corrected_error_rate=float('nan') if condition_name == "antisaccade" else None
            )

        # Formula 5 — Latency
        latencies = np.array([t.saccade.latency_ms for t in valid_trials])

        # Formula 7 — Peak velocity
        peak_velocities = np.array([t.saccade.peak_velocity_dps for t in valid_trials])

        # Formula 8/9 — Amplitude and gain
        gains = []
        for t in valid_trials:
            target_amp = t.spec.target_amplitude_deg
            actual_amp = t.saccade.amplitude_deg
            gain = self._compute_gain(actual_amp, target_amp)
            gains.append(gain)
        gains = np.array(gains)

        # Formula 13 — MSP rate
        msp_flags = [
            self._is_multiple_step_saccade(t) for t in valid_trials
        ]
        msp_rate = float(np.mean(msp_flags)) if msp_flags else float('nan')

        error_rate = None
        corrected_error_rate = None

        if condition_name == "antisaccade":
            error_rate, corrected_error_rate = self._compute_antisaccade_rates(
                block_result.trials
            )

        return ConditionMetrics(
            condition_name=condition_name,
            n_total_trials=n_total,
            n_valid_trials=n_valid,
            latency_mean_ms=float(np.mean(latencies)),
            latency_sd_ms=float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
            latency_median_ms=float(np.median(latencies)),
            peak_velocity_mean_dps=float(np.mean(peak_velocities)),
            peak_velocity_sd_dps=float(np.std(peak_velocities, ddof=1)) if len(peak_velocities) > 1 else 0.0,
            gain_mean=float(np.mean(gains)),
            gain_sd=float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
            msp_rate=msp_rate,
            error_rate=error_rate,
            corrected_error_rate=corrected_error_rate
        )

    def _compute_gain(self, amplitude_actual_deg: float, target_amplitude_deg: float) -> float:
        """
        Formula 9 — Saccade Gain
        Source: savi_math_metrics_spec.md Formula 9; Hopf et al. 2018.
        """
        if target_amplitude_deg == 0:
            return float('nan')
        return amplitude_actual_deg / abs(target_amplitude_deg)

    def _is_multiple_step_saccade(self, trial) -> bool:
        """
        Formula 13 — Multiple Step Pattern detection.
        Source: savi_math_metrics_spec.md Formula 13; Blekher et al. 2009.

        NOTE: at 30fps, MSP detection has lower confidence — a 150ms
        saccade spans only ~4.5 frames. Flag this limitation in the
        paper's methods section.
        """
        trace = trial.trace
        saccade = trial.saccade
        if trace is None or saccade is None:
            return False

        v_mag_window = trace.v_mag[saccade.onset_frame:saccade.offset_frame + 1]

        peaks, _ = find_peaks(
            v_mag_window,
            height=ONSET_THRESHOLD_DPS,
            distance=2
        )
        return len(peaks) >= 2

    def _compute_antisaccade_rates(self, trials: list) -> tuple[float, float]:
        """
        Formula 11 — Antisaccade Error Rate
        Formula 12 — Corrected Error Rate

        Source: savi_math_metrics_spec.md Formula 11/12; Fischer et al. 1997.
        """
        valid_trials = [
            t for t in trials
            if t.is_usable and t.saccade is not None
        ]

        if not valid_trials:
            return float('nan'), float('nan')

        n_errors = 0

        for t in valid_trials:
            saccade = t.saccade
            target_direction = t.spec.target_direction
            actual_direction = saccade.direction

            is_error = (actual_direction == target_direction)

            if is_error:
                n_errors += 1

        error_rate = n_errors / len(valid_trials)

        # Corrected error rate requires multi-saccade detection —
        # not available until detector.py supports secondary saccade
        # detection. Return NaN and document the gap.
        corrected_error_rate = float('nan')

        return error_rate, corrected_error_rate

    def _compute_gap_effect(self, overlap: ConditionMetrics, gap: ConditionMetrics) -> float:
        """
        Formula 14 — Gap Effect
        Source: savi_math_metrics_spec.md Formula 14; Fischer et al. 1997.
        """
        if np.isnan(overlap.latency_mean_ms) or np.isnan(gap.latency_mean_ms):
            return float('nan')
        return overlap.latency_mean_ms - gap.latency_mean_ms

    def _fit_main_sequence(self, block_results: list) -> tuple[float, float, float]:
        """
        Formula 10 — Main Sequence Exponential Model.
        Pools valid saccades across ALL conditions (overlap + gap +
        antisaccade) for the amplitude/velocity fit — this is a
        pipeline validation check, not a per-condition metric.

        Source: savi_math_metrics_spec.md Formula 10; Bahill et al. 1975.
        """
        amplitudes = []
        velocities = []

        for br in block_results:
            for t in br.trials:
                if t.is_usable and t.saccade is not None and t.saccade.is_valid:
                    amplitudes.append(t.saccade.amplitude_deg)
                    velocities.append(t.saccade.peak_velocity_dps)

        if len(amplitudes) < 4:
            logger.warning(
                f"Only {len(amplitudes)} valid saccades available — "
                "main sequence fit unreliable with < 4 points."
            )
            return float('nan'), float('nan'), float('nan')

        amplitudes = np.array(amplitudes)
        velocities = np.array(velocities)

        def main_sequence_model(A, v_max, A_0):
            return v_max * (1 - np.exp(-A / A_0))

        try:
            params, _ = curve_fit(
                main_sequence_model,
                amplitudes,
                velocities,
                p0=[700.0, 17.0],
                bounds=([300, 5], [1000, 40])
            )
            v_max_fit, a0_fit = params

            y_pred = main_sequence_model(amplitudes, v_max_fit, a0_fit)
            ss_res = np.sum((velocities - y_pred) ** 2)
            ss_tot = np.sum((velocities - np.mean(velocities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            if r_squared < 0.5:
                logger.warning(
                    f"Main sequence R²={r_squared:.2f} is below 0.5. "
                    "Velocity calculation may be corrupted. Check SG "
                    "window and calibration."
                )

            return float(v_max_fit), float(a0_fit), float(r_squared)

        except RuntimeError:
            logger.error("Main sequence curve fit failed to converge.")
            return float('nan'), float('nan'), float('nan')

    def extract_trial_order_series(self, block_result) -> TrialOrderSeries:
        """
        Extract per-trial metric values in trial order, without
        aggregation. Used for drift detection: plot latencies_ms or
        gains against trial_numbers and look for a slope. A flat
        trend indicates calibration held steady across the session.
        A visible upward slope in latency or downward slope in gain
        suggests accuracy degraded over the battery — investigate
        before trusting aggregate metrics from that session.

        Includes ALL trials (not just valid ones) with NaN placeholders
        for invalid/unusable trials, so the trial_numbers axis has no
        gaps — gaps would visually hide a drift pattern.
        """
        trial_numbers = []
        latencies = []
        peak_velocities = []
        gains = []
        valid_flags = []

        for t in block_result.trials:
            trial_numbers.append(t.spec.trial_number)

            is_valid_trial = (
                t.is_usable and t.saccade is not None and t.saccade.is_valid
            )
            valid_flags.append(is_valid_trial)

            if is_valid_trial:
                latencies.append(t.saccade.latency_ms)
                peak_velocities.append(t.saccade.peak_velocity_dps)
                gain = self._compute_gain(
                    t.saccade.amplitude_deg, t.spec.target_amplitude_deg
                )
                gains.append(gain)
            else:
                latencies.append(float('nan'))
                peak_velocities.append(float('nan'))
                gains.append(float('nan'))

        return TrialOrderSeries(
            condition_name=block_result.block_type.value,
            trial_numbers=trial_numbers,
            latencies_ms=latencies,
            peak_velocities_dps=peak_velocities,
            gains=gains,
            is_valid=valid_flags
        )

    def detect_drift(self, series: TrialOrderSeries) -> dict:
        """
        Quick statistical check for a trend in gain across trial order
        within a block. Uses simple linear regression (slope) on the
        valid (non-NaN) gain values against trial number.

        Returns:
            {
                "slope": float,       # gain change per trial
                "n_points": int,      # number of valid points used
                "flag": str           # "stable" | "drift_detected" | "insufficient_data"
            }

        This is a coarse screening check, not a formal statistical test.
        A |slope| > 0.005 per trial over a 20-trial block corresponds to
        a ~10% gain shift across the block — flagged as drift_detected.
        Adjust this threshold empirically once you have real session data.
        """
        valid_pairs = [
            (tn, g) for tn, g, ok in zip(
                series.trial_numbers, series.gains, series.is_valid
            ) if ok and not np.isnan(g)
        ]

        if len(valid_pairs) < 5:
            return {"slope": float('nan'), "n_points": len(valid_pairs),
                    "flag": "insufficient_data"}

        trial_nums = np.array([p[0] for p in valid_pairs])
        gain_vals = np.array([p[1] for p in valid_pairs])

        # Simple linear fit: gain = slope * trial_number + intercept
        slope, intercept = np.polyfit(trial_nums, gain_vals, 1)

        DRIFT_SLOPE_THRESHOLD = 0.005  # gain units per trial — empirical, tune later

        flag = "drift_detected" if abs(slope) > DRIFT_SLOPE_THRESHOLD else "stable"

        return {
            "slope": float(slope),
            "n_points": len(valid_pairs),
            "flag": flag
        }

