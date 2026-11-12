import math

import torch

from .config import KalmanConfig, LowPassConfig


class FiniteDifferenceFilter:
  """Pass the pose through; velocity is the backward difference. Amplifies estimation noise."""

  def __init__(self, dt_s: float) -> None:
    self.dt_s = dt_s
    self._previous = None

  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None:
    self._previous = torch.as_tensor(initial_pose_mmdeg, dtype=torch.float64)

  def update(self, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pose = torch.as_tensor(pose_mmdeg, dtype=torch.float64)
    velocity = (pose - self._previous) / self.dt_s
    self._previous = pose
    return pose, velocity


class LowPassDifferenceFilter:
  """Pass the pose through; velocity is the backward difference smoothed by a first-order low-pass."""

  def __init__(self, config: LowPassConfig, dt_s: float) -> None:
    self._difference = FiniteDifferenceFilter(dt_s)
    time_constant_s = 1.0 / (2.0 * math.pi * config.cutoff_hz)
    self._alpha = dt_s / (time_constant_s + dt_s)

  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None:
    self._difference.reset(initial_pose_mmdeg)
    self._velocity = torch.zeros(6, dtype=torch.float64)

  def update(self, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pose, raw_velocity = self._difference.update(pose_mmdeg)
    self._velocity = self._velocity + self._alpha * (raw_velocity - self._velocity)
    return pose, self._velocity


class LinearKalmanFilter:
  """Linear Kalman filter on the estimated pose with a constant-velocity (CV) or constant-acceleration (CA) model.

  Port of MagLevPoseEstimation `ConstantMotionModel` + `PoseMeasurement` (`filtering/ekf.py`): continuous white-noise
  process covariance, constant R, Joseph-form update, first measurement sets the pose with covariance R.
  F, Q, P0 and R are block diagonal there, so this runs one independent filter per axis. Internally SI (m, rad).
  Not ported: the 90-degree yaw branch alignment of the measurement; the estimators here keep yaw continuous.
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

  def update(self, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.as_tensor(pose_mmdeg, dtype=torch.float64) * self._SI
    if not self._started:
      self._x[:, 0] = z
      self._p = torch.diag_embed(torch.cat([self._r[:, None], self._rate_var], -1))
      self._started = True
    else:
      x = self._x @ self._f.T
      p = self._f @ self._p @ self._f.T + self._q
      gain = p[:, :, 0] / (p[:, 0, 0] + self._r)[:, None]  # H = [1, 0, ...]
      self._x = x + gain * (z - x[:, 0])[:, None]
      a = torch.eye(len(self._f), dtype=torch.float64) - gain[:, :, None] * torch.eye(len(self._f), dtype=torch.float64)[0]
      self._p = a @ p @ a.transpose(1, 2) + self._r[:, None, None] * gain[:, :, None] * gain[:, None, :]  # Joseph form
    return self._x[:, 0] / self._SI, self._x[:, 1] / self._SI
