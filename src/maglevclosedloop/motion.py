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
  """Linear Kalman filter with a constant-velocity model, one independent 2-state filter per axis.

  State per axis: [pose, velocity]. Measurement: the estimated pose.
  Process noise: white acceleration with standard deviation `acceleration_std` (discrete model).
  """

  def __init__(self, config: KalmanConfig, dt_s: float) -> None:
    dt = dt_s
    self._f = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64)
    g = torch.tensor([dt * dt / 2.0, dt], dtype=torch.float64)
    acceleration_var = torch.tensor(config.acceleration_std, dtype=torch.float64) ** 2
    self._q = acceleration_var[:, None, None] * torch.outer(g, g)  # (axis, 2, 2)
    self._r = torch.tensor(config.measurement_std, dtype=torch.float64) ** 2
    self._initial_velocity_var = torch.tensor(config.initial_velocity_std, dtype=torch.float64) ** 2

  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None:
    self._x = torch.stack([torch.as_tensor(initial_pose_mmdeg, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)], -1)
    self._p = torch.diag_embed(torch.stack([self._r, self._initial_velocity_var], -1))

  def update(self, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = self._x @ self._f.T
    p = self._f @ self._p @ self._f.T + self._q
    gain = p[..., :, 0] / (p[..., 0, 0] + self._r)[:, None]  # H = [1, 0]
    x = x + gain * (torch.as_tensor(pose_mmdeg, dtype=torch.float64) - x[:, 0])[:, None]
    self._p = p - gain[..., :, None] * p[..., 0, None, :]
    self._x = x
    return x[:, 0].clone(), x[:, 1].clone()
