"""
SAVI session_store.py
JSON persistence for SessionMetrics and RiskProfile.
Replaces the SQLite approach originally scoped for v0.2.0 — deferred
to v0.4.0 when the full product schema is known.
"""
import json
import dataclasses
import os
import time
from savi.analyzer import SessionMetrics, ConditionMetrics
from savi.risk_engine import RiskProfile, MetricFlag


def save_session_metrics(metrics: SessionMetrics, directory: str = "data") -> str:
    """
    Save SessionMetrics to a JSON file.
    Filename: data/session_metrics_{session_id}_{timestamp}.json
    """
    os.makedirs(directory, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"session_metrics_{metrics.session_id}_{timestamp_str}.json"
    path = os.path.join(directory, filename)
    with open(path, 'w') as f:
        json.dump(dataclasses.asdict(metrics), f, indent=2)
    return path


def load_session_metrics(path: str) -> SessionMetrics:
    """Load SessionMetrics from a JSON file."""
    with open(path) as f:
        d = json.load(f)
    d["overlap"] = ConditionMetrics(**d["overlap"])
    d["gap"] = ConditionMetrics(**d["gap"])
    d["antisaccade"] = ConditionMetrics(**d["antisaccade"])
    return SessionMetrics(**d)


def save_risk_profile(profile: RiskProfile, directory: str = "data") -> str:
    """
    Save RiskProfile to a JSON file.
    Filename: data/risk_profile_{session_id}_{timestamp}.json
    """
    os.makedirs(directory, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"risk_profile_{profile.session_id}_{timestamp_str}.json"
    path = os.path.join(directory, filename)
    with open(path, 'w') as f:
        json.dump(dataclasses.asdict(profile), f, indent=2)
    return path


def load_risk_profile(path: str) -> RiskProfile:
    """Load RiskProfile from a JSON file."""
    with open(path) as f:
        d = json.load(f)
    d["metric_flags"] = [MetricFlag(**mf) for mf in d["metric_flags"]]
    return RiskProfile(**d)


def save_raw_session(
    block_results: list,     # list[BlockResult]
    session_id: str,
    directory: str = "data"
) -> str:
    """
    Serialize the full raw BlockResult structure — every TrialLog,
    every GazeFrame, every DetectedSaccade and GazeTrace — to JSON.

    This is the ground-truth record of a battery run. Unlike
    SessionMetrics (aggregated), this preserves per-trial and
    per-frame data so trial-order analysis and re-processing with
    different pipeline parameters remain possible after the fact.

    Filename: data/raw_session_{session_id}_{timestamp}.json
    """
    os.makedirs(directory, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"raw_session_{session_id}_{timestamp_str}.json"
    path = os.path.join(directory, filename)

    def serialize_block_result(br) -> dict:
        return {
            "block_type": br.block_type.value,
            "block_number": br.block_number,
            "n_total": br.n_total,
            "n_usable": br.n_usable,
            "n_aborted": br.n_aborted,
            "t_block_start": br.t_block_start,
            "t_block_end": br.t_block_end,
            "trials": [serialize_trial_log(t) for t in br.trials]
        }

    def serialize_trial_log(t) -> dict:
        return {
            "spec": {
                "block_type": t.spec.block_type.value,
                "trial_number": t.spec.trial_number,
                "block_number": t.spec.block_number,
                "target_direction": t.spec.target_direction,
                "target_amplitude_deg": t.spec.target_amplitude_deg,
                "gap_duration_ms": t.spec.gap_duration_ms,
                "iti_duration_ms": t.spec.iti_duration_ms,
            },
            "t_trial_start": t.t_trial_start,
            "t_fixation_acquired": t.t_fixation_acquired,
            "t_target_onset": t.t_target_onset,
            "t_trial_end": t.t_trial_end,
            "n_frames": t.n_frames,
            "frames": [serialize_gaze_frame(f) for f in t.frames],
            "trace": serialize_gaze_trace(t.trace) if t.trace else None,
            "saccade": serialize_saccade(t.saccade) if t.saccade else None,
            "is_usable": t.is_usable,
            "aborted": t.aborted,
            "abort_reason": t.abort_reason,
            "is_correct_antisaccade": t.is_correct_antisaccade,
        }

    def serialize_gaze_frame(f) -> dict:
        return dataclasses.asdict(f)

    def serialize_gaze_trace(trace) -> dict:
        return {
            "session_id": trace.session_id,
            "trial_id": trace.trial_id,
            "timestamps_s": trace.timestamps_s.tolist(),
            "x_deg": trace.x_deg.tolist(),
            "y_deg": trace.y_deg.tolist(),
            "x_deg_smooth": trace.x_deg_smooth.tolist(),
            "y_deg_smooth": trace.y_deg_smooth.tolist(),
            "v_x": trace.v_x.tolist(),
            "v_y": trace.v_y.tolist(),
            "v_mag": trace.v_mag.tolist(),
            "blink_mask": trace.blink_mask.tolist(),
            "fps": trace.fps,
            "n_blink_frames": trace.n_blink_frames,
            "is_usable": trace.is_usable,
        }

    def serialize_saccade(s) -> dict:
        return dataclasses.asdict(s)

    payload = {
        "session_id": session_id,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blocks": [serialize_block_result(br) for br in block_results]
    }

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

    return path


def load_raw_session(path: str):
    """
    Reconstruct list[BlockResult] from a saved raw session JSON file.

    Returns list[BlockResult] with all TrialLog, GazeTrace, DetectedSaccade,
    and GazeFrame objects fully reconstructed — usable directly by
    analyzer.py or any downstream re-processing.
    """
    import numpy as np
    from savi.protocol import (
        BlockResult, TrialLog, TrialSpec, BlockType
    )
    from savi.preprocessor import GazeTrace
    from savi.detector import DetectedSaccade
    from savi.tracker import GazeFrame

    with open(path) as f:
        payload = json.load(f)

    def deserialize_gaze_frame(d: dict) -> GazeFrame:
        return GazeFrame(**d)

    def deserialize_gaze_trace(d: dict | None) -> "GazeTrace | None":
        if d is None:
            return None
        return GazeTrace(
            session_id=d["session_id"],
            trial_id=d["trial_id"],
            timestamps_s=np.array(d["timestamps_s"]),
            x_deg=np.array(d["x_deg"]),
            y_deg=np.array(d["y_deg"]),
            x_deg_smooth=np.array(d["x_deg_smooth"]),
            y_deg_smooth=np.array(d["y_deg_smooth"]),
            v_x=np.array(d["v_x"]),
            v_y=np.array(d["v_y"]),
            v_mag=np.array(d["v_mag"]),
            blink_mask=np.array(d["blink_mask"], dtype=bool),
            fps=d["fps"],
            n_blink_frames=d["n_blink_frames"],
            is_usable=d["is_usable"],
        )

    def deserialize_saccade(d: dict | None) -> "DetectedSaccade | None":
        if d is None:
            return None
        return DetectedSaccade(**d)

    def deserialize_trial_log(d: dict) -> TrialLog:
        spec = TrialSpec(
            block_type=BlockType(d["spec"]["block_type"]),
            trial_number=d["spec"]["trial_number"],
            block_number=d["spec"]["block_number"],
            target_direction=d["spec"]["target_direction"],
            target_amplitude_deg=d["spec"]["target_amplitude_deg"],
            gap_duration_ms=d["spec"]["gap_duration_ms"],
            iti_duration_ms=d["spec"]["iti_duration_ms"],
        )
        return TrialLog(
            spec=spec,
            t_trial_start=d["t_trial_start"],
            t_fixation_acquired=d["t_fixation_acquired"],
            t_target_onset=d["t_target_onset"],
            t_trial_end=d["t_trial_end"],
            frames=[deserialize_gaze_frame(f) for f in d["frames"]],
            n_frames=d["n_frames"],
            trace=deserialize_gaze_trace(d["trace"]),
            saccade=deserialize_saccade(d["saccade"]),
            is_usable=d["is_usable"],
            aborted=d["aborted"],
            abort_reason=d["abort_reason"],
            is_correct_antisaccade=d["is_correct_antisaccade"],
        )

    def deserialize_block_result(d: dict) -> BlockResult:
        return BlockResult(
            block_type=BlockType(d["block_type"]),
            block_number=d["block_number"],
            trials=[deserialize_trial_log(t) for t in d["trials"]],
            n_total=d["n_total"],
            n_usable=d["n_usable"],
            n_aborted=d["n_aborted"],
            t_block_start=d["t_block_start"],
            t_block_end=d["t_block_end"],
        )

    return [deserialize_block_result(b) for b in payload["blocks"]]

