import math

import torch

from ..config import KalmanConfig
from ..rotation import align_yaw_branch


class LinearKalmanFilter:
  """Linear Kalman filter on the estimated pose with a constant-velocity (CV) or constant-acceleration (CA) model.

  Port of MagLevPoseEstimation `ConstantMotionModel` + `PoseMeasurement` (`filtering/ekf.py`): continuous white-noise
  process covariance, constant R, Joseph-form update, first measurement sets the pose with covariance R.
  Each measurement moves to the 90-degree yaw branch nearest the prediction; roll and pitch innovations wrap.
  F, Q, P0 and R are block diagonal there, so this runs one independent filter per axis. Internally SI (m, rad).
  Cycle as `FilteredPoseEstimator`: `predict` (the estimator starts from it), then `update` with the estimate.
  """

  _SI = torch.tensor([1e-3] * 3 + [math.pi / 180.0] * 3, dtype=torch.float64)  # mm, deg -> m, rad

  def __init__(self, config: KalmanConfig, dt_s: float) -> None:
    t = dt_s
    if config.motion_model == "cv":
      f = [[1, t], [0, 1]]
      q = [[t**3 / 3, t**2 / 2], [t**2 / 2, t]]
      rate_std = [config.initial_rate_std]
    elif config.motion_model == "ca":
      f = [[1, t, t**2 / 2], [0, 1, t], [0, 0, 1]]
      q = [[t**5 / 20, t**4 / 8, t**3 / 6], [t**4 / 8, t**3 / 3, t**2 / 2], [t**3 / 6, t**2 / 2, t]]
      rate_std = [config.initial_rate_std, config.initial_acceleration_std]
    else:
      raise ValueError(f"Unknown motion model {config.motion_model!r}")
    self._f = torch.tensor(f, dtype=torch.float64)
    self._q = torch.tensor(config.process_psd, dtype=torch.float64)[:, None, None] * torch.tensor(q, dtype=torch.float64)
    self._r = torch.tensor(config.measurement_std, dtype=torch.float64) ** 2
    self._rate_var = torch.tensor(rate_std, dtype=torch.float64).T ** 2  # (axis, order)

  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None:
    self._x = torch.zeros(6, len(self._f), dtype=torch.float64)
    self._x[:, 0] = torch.as_tensor(initial_pose_mmdeg, dtype=torch.float64) * self._SI
    self._started = False

  def predict(self) -> torch.Tensor:
    """Predicted pose (mm, deg); in the first cycle the reset pose, as the filter has no measurement yet."""
    if self._started:
      self._x = self._x @ self._f.T
      self._p = self._f @ self._p @ self._f.T + self._q
    return self._x[:, 0] / self._SI

  def update(self, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_yaw_deg = self._x[5, 0] / self._SI[5]
    z = align_yaw_branch(torch.as_tensor(pose_mmdeg, dtype=torch.float64), predicted_yaw_deg) * self._SI
    if not self._started:
      self._x[:, 0] = z
      self._p = torch.diag_embed(torch.cat([self._r[:, None], self._rate_var], -1))
      self._started = True
    else:
      innovation = z - self._x[:, 0]
      innovation[3:5] = torch.remainder(innovation[3:5] + math.pi, 2 * math.pi) - math.pi
      gain = self._p[:, :, 0] / (self._p[:, 0, 0] + self._r)[:, None]  # H = [1, 0, ...]
      self._x = self._x + gain * innovation[:, None]
      a = torch.eye(len(self._f), dtype=torch.float64) - gain[:, :, None] * torch.eye(len(self._f), dtype=torch.float64)[0]
      self._p = a @ self._p @ a.transpose(1, 2) + self._r[:, None, None] * gain[:, :, None] * gain[:, None, :]  # Joseph form
    return self._x[:, 0] / self._SI, self._x[:, 1] / self._SI
