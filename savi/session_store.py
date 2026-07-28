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
