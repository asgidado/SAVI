from dataclasses import dataclass
import numpy as np

# I-VT Saccade Detection Constants
# Source: savi_math_metrics_spec.md Formula 6
# Thread 8; Nyström & Holmqvist 2010; Salvucci & Goldberg 2000
# Confirmed against current literature June 2026
ONSET_THRESHOLD_DPS  = 30.0    # °/s — standard I-VT onset
OFFSET_THRESHOLD_DPS = 20.0    # °/s — asymmetric: deceleration is slower
MIN_ONSET_FRAMES     = 3       # frames velocity must exceed threshold
MIN_DURATION_MS      = 10.0    # ms — minimum physiological saccade
MAX_DURATION_MS      = 150.0   # ms — beyond this is artifact or blink
MIN_AMPLITUDE_DEG    = 0.5     # degrees — reject microsaccades
ANTICIPATORY_MS      = 80.0    # ms — below this is not a reactive saccade
SEARCH_WINDOW_MS     = 600.0   # ms after target onset
VELOCITY_CEILING_DPS = 1000.0  # physiological ceiling — defensive re-check


@dataclass
class DetectedSaccade:
    onset_frame: int             # index into GazeTrace arrays
    offset_frame: int
    onset_timestamp_s: float
    offset_timestamp_s: float
    latency_ms: float            # (onset_timestamp_s - t_target_onset_s) * 1000
    duration_ms: float           # (offset_timestamp_s - onset_timestamp_s) * 1000
    peak_velocity_dps: float     # max(v_mag[onset_frame:offset_frame+1])
    amplitude_deg: float         # abs(x_deg_smooth[offset] - x_deg_smooth[onset])
    direction: str               # "left" or "right"
    is_anticipatory: bool        # True if latency_ms < ANTICIPATORY_MS
    is_valid: bool               # True if passes all rejection criteria
    rejection_reason: str        # "" if valid; first failing criterion if not


def detect_primary_saccade(
    trace,                    # GazeTrace
    t_target_onset_s: float,
    expected_direction: str   # "left" or "right"
) -> DetectedSaccade | None:
    """
    Detect the primary saccade in a trial using the I-VT algorithm.

    Returns DetectedSaccade (valid or not) if a candidate is found.
    Returns None if no velocity threshold crossing occurs in the window.

    Source: savi_math_metrics_spec.md Formula 6
    Thread 8; Nyström & Holmqvist 2010
    """
    # Pre-filter:
    v_mag = trace.v_mag.copy()
    v_mag = np.where(v_mag > VELOCITY_CEILING_DPS, 0.0, v_mag)

    # Define search window indices:
    search_start_s = t_target_onset_s
    search_end_s   = t_target_onset_s + (SEARCH_WINDOW_MS / 1000.0)

    in_window = np.where(
        (trace.timestamps_s >= search_start_s) &
        (trace.timestamps_s <= search_end_s)
    )[0]

    if len(in_window) == 0:
        return None

    # Onset detection:
    onset_frame = None
    for i in in_window:
        if v_mag[i] > ONSET_THRESHOLD_DPS:
            # check persistence
            end_check = min(i + MIN_ONSET_FRAMES, len(v_mag))
            if np.all(v_mag[i:end_check] > ONSET_THRESHOLD_DPS):
                onset_frame = i
                break

    if onset_frame is None:
        return None

    # Direction assignment:
    check_end = min(onset_frame + 3, len(trace.v_x))
    mean_vx = float(np.mean(trace.v_x[onset_frame:check_end]))
    direction = "right" if mean_vx > 0 else "left"

    # Offset detection:
    offset_frame = None
    consecutive = 0
    for i in range(onset_frame + 1, len(v_mag)):
        if v_mag[i] < OFFSET_THRESHOLD_DPS:
            consecutive += 1
            if consecutive >= 3:
                offset_frame = i
                break
        else:
            consecutive = 0

    if offset_frame is None:
        # No clean offset — use last frame of search window
        offset_frame = in_window[-1]
        no_offset = True
    else:
        no_offset = False

    # Compute metrics:
    # Formula 5 — Latency (Thread 8; Fischer et al. 1997)
    onset_ts  = trace.timestamps_s[onset_frame]
    offset_ts = trace.timestamps_s[offset_frame]
    latency_ms   = (onset_ts - t_target_onset_s) * 1000.0
    duration_ms  = (offset_ts - onset_ts) * 1000.0
    peak_vel     = float(np.max(v_mag[onset_frame:offset_frame + 1]))
    amplitude    = float(abs(
        trace.x_deg_smooth[offset_frame] - trace.x_deg_smooth[onset_frame]
    ))
    is_anticipatory = latency_ms < ANTICIPATORY_MS

    # Rejection criteria — evaluate all, record first failing reason:
    rejection_reason = ""
    is_valid = True

    if no_offset:
        is_valid = False
        rejection_reason = "no_offset"
    elif duration_ms < MIN_DURATION_MS:
        is_valid = False
        rejection_reason = "too_short"
    elif duration_ms > MAX_DURATION_MS:
        is_valid = False
        rejection_reason = "too_long"
    elif amplitude < MIN_AMPLITUDE_DEG:
        is_valid = False
        rejection_reason = "too_small"
    elif latency_ms < 0:
        is_valid = False
        rejection_reason = "negative_latency"

    return DetectedSaccade(
        onset_frame=onset_frame,
        offset_frame=offset_frame,
        onset_timestamp_s=onset_ts,
        offset_timestamp_s=offset_ts,
        latency_ms=latency_ms,
        duration_ms=duration_ms,
        peak_velocity_dps=peak_vel,
        amplitude_deg=amplitude,
        direction=direction,
        is_anticipatory=is_anticipatory,
        is_valid=is_valid,
        rejection_reason=rejection_reason
    )
