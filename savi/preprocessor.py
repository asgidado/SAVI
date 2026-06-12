import logging
from dataclasses import dataclass
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter
import numpy as np

logger = logging.getLogger("savi.preprocessor")

@dataclass
class GazeTrace:
    """
    Processed gaze trace produced by Preprocessor.process_trial().
    All arrays are shape (N,) where N = number of input GazeFrame objects.
    """
    session_id: str
    trial_id: int
    timestamps_s: np.ndarray      # float64 — time.perf_counter() values
    x_deg: np.ndarray             # calibrated horizontal gaze, degrees
    y_deg: np.ndarray             # calibrated vertical gaze, degrees
    x_deg_smooth: np.ndarray      # Savitzky-Golay smoothed x
    y_deg_smooth: np.ndarray      # Savitzky-Golay smoothed y
    v_x: np.ndarray               # horizontal velocity °/s, signed
    v_y: np.ndarray               # vertical velocity °/s, signed
    v_mag: np.ndarray             # velocity magnitude °/s, always >= 0
    blink_mask: np.ndarray        # bool — True = blink frame
    fps: float                    # 30.0
    n_blink_frames: int           # count of True values in blink_mask
    is_usable: bool               # False if >20% of saccade window is blinks


class Preprocessor:
    """
    Converts a list of GazeFrame objects into a processed GazeTrace.

    Pipeline order (strict — do not reorder):
      Step 1: Extract arrays from GazeFrame list
      Step 2: Build blink mask
      Step 3: Dilate blink mask
      Step 4: Blink interpolation
      Step 5: Savitzky-Golay smoothing
      Step 6: Velocity computation
      Step 7: Velocity ceiling clamp
      Step 8: Usability check
    """

    # Literature-derived constants — not configurable at runtime
    # Source: Thread 7 pipeline spec; savi_math_metrics_spec.md
    BLINK_CONFIDENCE_THRESHOLD    = 0.5
    BLINK_DILATION_FRAMES         = 2
    MAX_INTERPOLATION_DURATION_MS = 200.0
    BLINK_REJECTION_THRESHOLD     = 0.20
    SG_WINDOW                     = 5      # 30fps path — do not make adaptive
    SG_POLYORDER                  = 2
    VELOCITY_CEILING_DPS          = 1000.0
    # HEAD_CORRECTION_FACTOR = 0.8
    # ADR-0001: head pose correction deferred — iris-in-socket handles
    # the primary head movement sensitivity at the calibration layer.
    # This scalar factor is retained as a placeholder only.

    def process_trial(
        self,
        frames: list,           # list[GazeFrame]
        session_id: str,
        trial_id: int,
        t_target_onset_s: float
    ) -> GazeTrace:
        fps = 30.0  # Confirmed hardware constraint
        
        # Step 1: Extract arrays from GazeFrame list
        timestamps_s = np.array([f.timestamp for f in frames], dtype=np.float64)
        
        x_deg_list = []
        y_deg_list = []
        blink_raw_list = []
        confidence_list = []
        
        for f in frames:
            if not f.calibration_applied:
                logger.warning(
                    f"Frame {f.frame_idx} has no calibration applied — using raw gaze.\n"
                    "Spatial metrics will be inaccurate for this trial."
                )
                x_deg_list.append(f.gaze_x_deg)
                y_deg_list.append(f.gaze_y_deg)
            else:
                x_deg_list.append(f.cal_x_deg)
                y_deg_list.append(f.cal_y_deg)
                
            blink_raw_list.append(f.blink)
            confidence_list.append(f.confidence)
            
        x_deg = np.array(x_deg_list, dtype=np.float64)
        y_deg = np.array(y_deg_list, dtype=np.float64)
        blink_raw = np.array(blink_raw_list, dtype=bool)
        confidence = np.array(confidence_list, dtype=np.float64)
        
        # Step 2: Build blink mask
        # Formula: frame is a blink if tracker flagged it OR confidence is low
        # Source: savi_architecture_spec.md Module 4
        blink_mask = blink_raw | (confidence < self.BLINK_CONFIDENCE_THRESHOLD)
        
        # Step 3: Dilate blink mask
        struct = np.ones(2 * self.BLINK_DILATION_FRAMES + 1, dtype=bool)
        blink_mask = binary_dilation(blink_mask, structure=struct)
        
        # Step 4: Blink interpolation
        n = len(blink_mask)
        i = 0
        while i < n:
            if blink_mask[i]:
                start_idx = i
                while i < n and blink_mask[i]:
                    i += 1
                end_idx = i - 1
                
                duration_ms = (end_idx - start_idx + 1) / fps * 1000.0
                if duration_ms < self.MAX_INTERPOLATION_DURATION_MS:
                    left_anchor = start_idx - 1
                    right_anchor = end_idx + 1
                    
                    has_left = (left_anchor >= 0)
                    has_right = (right_anchor < n)
                    
                    if has_left and has_right:
                        x_gap = np.interp(
                            np.arange(start_idx, end_idx + 1),
                            [left_anchor, right_anchor],
                            [x_deg[left_anchor], x_deg[right_anchor]]
                        )
                        y_gap = np.interp(
                            np.arange(start_idx, end_idx + 1),
                            [left_anchor, right_anchor],
                            [y_deg[left_anchor], y_deg[right_anchor]]
                        )
                        x_deg[start_idx : end_idx + 1] = x_gap
                        y_deg[start_idx : end_idx + 1] = y_gap
                    elif has_left:
                        x_deg[start_idx : end_idx + 1] = x_deg[left_anchor]
                        y_deg[start_idx : end_idx + 1] = y_deg[left_anchor]
                    elif has_right:
                        x_deg[start_idx : end_idx + 1] = x_deg[right_anchor]
                        y_deg[start_idx : end_idx + 1] = y_deg[right_anchor]
            else:
                i += 1
                
        # Step 5: Savitzky-Golay smoothing
        # Formula 3 — Savitzky-Golay (Thread 7; Nyström & Holmqvist 2010)
        # CORRECT order: smooth position → compute velocity
        # INCORRECT: compute velocity → smooth velocity
        if len(x_deg) >= self.SG_WINDOW:
            x_deg_smooth = savgol_filter(
                x_deg, window_length=self.SG_WINDOW, polyorder=self.SG_POLYORDER
            )
            y_deg_smooth = savgol_filter(
                y_deg, window_length=self.SG_WINDOW, polyorder=self.SG_POLYORDER
            )
        else:
            x_deg_smooth = x_deg.copy()
            y_deg_smooth = y_deg.copy()
            
        # Step 6: Velocity computation
        # Formula 4 — Central difference velocity (Thread 8; Bahill et al.)
        # np.gradient() implements central difference for interior points.
        # Source confirmed June 2026 as field standard.
        dt = 1.0 / fps
        v_x = np.gradient(x_deg_smooth, dt)
        v_y = np.gradient(y_deg_smooth, dt)
        v_mag = np.sqrt(v_x**2 + v_y**2)
        
        # Step 7: Velocity ceiling clamp
        # Physiological ceiling: >1000°/s is an artifact, not a saccade
        # Source: Nyström & Holmqvist 2010; savi_math_metrics_spec.md Formula 4
        v_mag = np.where(v_mag > self.VELOCITY_CEILING_DPS, 0.0, v_mag)
        
        # Step 8: Usability check
        saccade_window_end_s = t_target_onset_s + 0.600
        in_window = (timestamps_s >= t_target_onset_s) & (timestamps_s <= saccade_window_end_s)
        n_window_frames = int(np.sum(in_window))

        if n_window_frames == 0:
            is_usable = True
        else:
            n_blink_in_window = int(np.sum(blink_mask & in_window))
            is_usable = (n_blink_in_window / n_window_frames) < self.BLINK_REJECTION_THRESHOLD
            
        return GazeTrace(
            session_id=session_id,
            trial_id=trial_id,
            timestamps_s=timestamps_s,
            x_deg=x_deg,
            y_deg=y_deg,
            x_deg_smooth=x_deg_smooth,
            y_deg_smooth=y_deg_smooth,
            v_x=v_x,
            v_y=v_y,
            v_mag=v_mag,
            blink_mask=blink_mask,
            fps=fps,
            n_blink_frames=int(np.sum(blink_mask)),
            is_usable=is_usable
        )
