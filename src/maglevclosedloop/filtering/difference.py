import math

import torch

from ..config import LowPassConfig


class FiniteDifferenceFilter:
  """Pass the pose through; velocity is the backward difference. Amplifies estimation noise."""

  def __init__(self, dt_s: float) -> None:
    self.dt_s = dt_s
    self._previous = None

  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None:
    self._previous = torch.as_tensor(initial_pose_mmdeg, dtype=torch.float64)

  def predict(self) -> torch.Tensor:
    """No motion model: the last pose."""
    return self._previous

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

  def predict(self) -> torch.Tensor:
    return self._difference.predict()

  def update(self, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pose, raw_velocity = self._difference.update(pose_mmdeg)
    self._velocity = self._velocity + self._alpha * (raw_velocity - self._velocity)
    return pose, self._velocity

