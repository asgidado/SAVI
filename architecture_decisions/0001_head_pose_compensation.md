# ADR-0001: Head-Pose Compensated Calibration for Elderly-Friendly Gaze Assessment

## Context & Problem Statement

Standard webcam eye tracking maps the absolute pixel position of the iris in the camera frame directly to screen coordinates. Because this model assumes a fixed relationship between the eyes and the camera, it requires the user's head to remain absolutely still. 

Even minor head movements of a few millimeters shift the absolute iris position, introducing large errors ($3^\circ$ to $8^\circ$) in the calibrated gaze estimation. In clinical settings, particularly with elderly patients or subjects with motor/neck stiffness, requiring a physical chin rest or enforcing rigid head immobility leads to rapid fatigue, neck strain, and high test failure rates.

We need a way to perform gaze tracking calibration and validation that tolerates natural head drift and micro-movements, making the software accessible to clinical and elderly populations.

## Proposed Decision

We will transition from absolute iris coordinate mapping to **Head-Pose Compensated Calibration**. Instead of fitting a mapping from raw iris pixels $I(x, y)$ directly to screen degrees $D(x, y)$, the calibration algorithm will incorporate 3D head-pose translation and rotation metrics from MediaPipe's Face Mesh.

The mathematical model will expand the regression feature vector to combine relative iris-in-socket coordinates and head-pose variables:

1. **Iris-in-Socket coordinates**: Calculate the displacement of the iris center relative to the eye socket anchor landmarks (e.g. eye corners) instead of the absolute camera frame coordinates.
2. **3D Head-Pose Vector**: Compute the head's translation ($tx, ty, tz$) and rotation ($rx, ry, rz$) using the 3D landmark mesh (e.g., nose tip, chin, forehead, and outer face boundary).
3. **Regularized Polynomial Fusion**: Use a regularized polynomial model (Ridge Regression) that combines these coordinates:
   $$\text{Gaze}_x = f(\text{Iris}_{\text{rel\_x}}, tx, rx, \dots)$$

## Status

**Proposed** (Targeted for SAVI v0.0.3/v0.1.0)

## Consequences

### Positive
* **Clinical Compliance**: Eliminates the need for chin rests, head straps, or extreme straining, making the test comfortable for elderly, pediatric, and motor-impaired users.
* **Accuracy Under Motion**: Maintains sub-$2^\circ$ accuracy even if the user shifts their posture or leans slightly during the assessment.
* **Low Hardware Overhead**: Relies entirely on the existing MediaPipe 3D face mesh, requiring no secondary hardware (infrared sensors, depth cameras, etc.).

### Negative / Risks
* **Feature Vector Complexity**: Increasing the number of regression features requires regularized regression (Ridge/Lasso) to prevent overfitting on small calibration datasets (e.g., 5-point or 9-point grids).
* **Calibration Settling**: We must ensure that head-pose metrics are smoothed (e.g., low-pass or Kalman filter) to prevent high-frequency head tremor from injecting jitter into the gaze output.
