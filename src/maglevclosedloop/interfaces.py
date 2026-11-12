"""Stage interfaces of the closed loop.

All poses are global `[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]` tensors of shape `(6,)`.
Hall data and coil currents keep the MagLevSim layout `(tile_count, 50, 3)` and `(tile_count, 24, 3)`.
"""

from typing import Protocol

import torch


class Reference(Protocol):
  def pose_mmdeg(self, time_s: float) -> torch.Tensor:
    """Desired pose at `time_s`."""


class CoilFieldCompensation(Protocol):
  def compensate(self, hall_counts: torch.Tensor, coil_currents_abc: torch.Tensor) -> torch.Tensor:
    """Return Hall data `(tile_count, 50, 3)` without the stator coil field."""


class PoseEstimator(Protocol):
  def estimate(self, hall_counts: torch.Tensor, prior_mmdeg: torch.Tensor, initial_mmdeg: torch.Tensor) -> torch.Tensor:
    """Pose `(6,)` from compensated Hall data `(tile_count, 50, 3)` in counts, order `[By, Bx, Bz]`.

    `prior_mmdeg`: the last output pose of the loop; it selects the sensor window.
    `initial_mmdeg`: the motion-filter prediction; start value and symmetry branch.
    """


class MotionFilter(Protocol):
  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None: ...

  def predict(self) -> torch.Tensor:
    """Predicted pose `(6,)` for the current cycle, before the estimate."""

  def update(self, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose `(6,)` and velocity `(6,)` in mm/s and deg/s from the estimated pose."""


class Controller(Protocol):
  def reset(self) -> None: ...

  def compute(
    self, pose_mmdeg: torch.Tensor, velocity: torch.Tensor, reference_mmdeg: torch.Tensor, achieved_wrench: torch.Tensor | None
  ) -> torch.Tensor:
    """Desired mover wrench `(6,)` as `[Fx, Fy, Fz, Tx, Ty, Tz]`.

    `achieved_wrench`: what the actuation realized of the previous wrench (None in the first cycle), for anti-windup.
    """


class Actuation(Protocol):
  def compute(self, wrench: torch.Tensor, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Coil command `(tile_count, 24, 3)` as `[d_incr, q_incr, electrical_angle_rad]`, and the wrench `(6,)` it achieves."""
