import numpy as np
from savi.tracker import GazeFrame
from savi.preprocessor import Preprocessor, GazeTrace
from savi.detector import detect_primary_saccade, ONSET_THRESHOLD_DPS

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


def test_sg_smoothing_preserves_length():
    from scipy.signal import savgol_filter
    data = np.random.rand(60)
    smoothed = savgol_filter(data, window_length=5, polyorder=2)
    assert len(smoothed) == len(data)


def test_sg_smoothing_reduces_noise():
    from scipy.signal import savgol_filter
    dt = 1.0 / 30.0
    t = np.arange(30) * dt
    clean = 5.0 * np.sin(2.0 * np.pi * 1.0 * t)
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, size=30)
    noisy = clean + noise
    
    smoothed = savgol_filter(noisy, window_length=5, polyorder=2)
    
    std_noisy = np.std(noisy - clean)
    std_smoothed = np.std(smoothed - clean)
    
    assert std_smoothed < std_noisy


def test_velocity_central_difference():
    dt = 1.0 / 30.0
    x = np.array([5.0 * (i * dt) for i in range(30)])
    v = np.gradient(x, dt)
    
    for val in v[1:29]:
        assert abs(val - 5.0) < 0.01


def test_blink_interpolation_short_gap():
    dt = 1.0 / 30.0
    frames = []
    for i in range(30):
        blink = (10 <= i <= 14)
        frames.append(make_frame(timestamp=i * dt, frame_idx=i, gaze_x=0.0, blink=blink))
        
    preprocessor = Preprocessor()
    trace = preprocessor.process_trial(frames, "s1", 1, t_target_onset_s=0.0)
    
    assert np.all(trace.blink_mask[10:15])
    assert not np.isnan(trace.x_deg_smooth).any()
    assert not np.isnan(trace.y_deg_smooth).any()


def test_usability_check_passes():
    dt = 1.0 / 30.0
    frames = []
    for i in range(60):
        frames.append(make_frame(timestamp=i * dt, frame_idx=i, gaze_x=0.0, blink=False))
        
    preprocessor = Preprocessor()
    trace = preprocessor.process_trial(frames, "s1", 1, t_target_onset_s=1.0)
    
    assert trace.is_usable is True


def test_detect_primary_saccade_known_input():
    fps = 100.0
    n = 100
    dt = 1.0 / fps
    t_target = 0.5  # target onset at 0.5s (index 50)
    
    timestamps = np.array([i * dt for i in range(n)])
    
    x_deg_smooth = np.zeros(n)
    # transition over two frames to produce 3-frame velocity spike
    x_deg_smooth[51] = 5.0
    x_deg_smooth[52:] = 10.0
    
    # Velocity: central difference of the position step
    v_x = np.gradient(x_deg_smooth, dt)
    v_y = np.zeros(n)
    v_mag = np.abs(v_x)
    
    trace = GazeTrace(
        session_id="test", trial_id=1,
        timestamps_s=timestamps,
        x_deg=x_deg_smooth.copy(), y_deg=np.zeros(n),
        x_deg_smooth=x_deg_smooth, y_deg_smooth=np.zeros(n),
        v_x=v_x, v_y=v_y, v_mag=v_mag,
        blink_mask=np.zeros(n, dtype=bool),
        fps=fps, n_blink_frames=0, is_usable=True
    )
    
    result = detect_primary_saccade(trace, t_target_onset_s=t_target, expected_direction="right")
    
    assert result is not None
    assert result.is_valid is True or result.rejection_reason in ("too_short", "too_small")
    assert result.direction == "right"
    assert result.latency_ms >= 0.0
    assert result.peak_velocity_dps > ONSET_THRESHOLD_DPS


def test_detect_returns_none_no_saccade():
    fps = 30.0
    n = 60
    dt = 1.0 / fps
    timestamps = np.array([i * dt for i in range(n)])
    
    trace = GazeTrace(
        session_id="test", trial_id=1,
        timestamps_s=timestamps,
        x_deg=np.zeros(n), y_deg=np.zeros(n),
        x_deg_smooth=np.zeros(n), y_deg_smooth=np.zeros(n),
        v_x=np.zeros(n), v_y=np.zeros(n), v_mag=np.zeros(n),
        blink_mask=np.zeros(n, dtype=bool),
        fps=fps, n_blink_frames=0, is_usable=True
    )
    
    result = detect_primary_saccade(trace, t_target_onset_s=0.5, expected_direction="right")
    assert result is None
