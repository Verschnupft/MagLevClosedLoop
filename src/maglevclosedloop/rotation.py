"""Euler-angle helpers in the MagLevSim convention R = Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in degrees."""

import torch
from maglevsim.physics import euler_to_rotation_matrix


def zyx_euler_deg(rotation: torch.Tensor) -> torch.Tensor:
  """Inverse of `euler_to_rotation_matrix`: `(..., 3, 3)` -> `(..., 3)` roll, pitch, yaw in deg."""
  roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
  pitch = torch.atan2(-rotation[..., 2, 0], torch.hypot(rotation[..., 0, 0], rotation[..., 1, 0]))
  yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
  return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=-1))


def wrap_deg(angle: torch.Tensor) -> torch.Tensor:
  return torch.remainder(angle + 180.0, 360.0) - 180.0


def align_yaw_branch(pose_mmdeg: torch.Tensor, reference_yaw_deg: torch.Tensor) -> torch.Tensor:
  """Move `pose` (6,) to the 90-degree symmetry branch R Rz(k 90) whose yaw is nearest `reference_yaw_deg`.

  Port of `align_gamma_branch` (MagLevPoseEstimation `neural/kalman.py`); roll and pitch change with the branch.
  """
  k = torch.round((reference_yaw_deg - pose_mmdeg[5]) / 90.0)
  twist = torch.stack([torch.zeros_like(k), torch.zeros_like(k), 90.0 * k])
  angles = zyx_euler_deg(euler_to_rotation_matrix(pose_mmdeg[3:]) @ euler_to_rotation_matrix(twist))
  out = pose_mmdeg.clone()
  out[3:5] = angles[:2]
  out[5] = pose_mmdeg[5] + 90.0 * k
  out[5] = out[5] + wrap_deg(angles[2] - out[5])
  return out
