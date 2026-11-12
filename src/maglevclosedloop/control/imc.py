"""Cascade gains by internal model control (IMC), as `ImcPidTuning` (PT1 case) in MagLevWhiteBox.

Plant per axis from the MagLevSim rigid body: m dv/dt = F - c v (rotation: I dw/dt = T - c_w w), so the
velocity plant is the PT1 element G(s) = K / (T s + 1) with K = 1/c and T = m/c.
IMC with closed velocity loop 1 / (lambda s + 1) gives the PI gains kp = T / (K lambda) = m / lambda and
ki = 1 / (K lambda) = c / lambda. The outer position loop sees the closed inner loop (gain 1) plus an
integrator, so a P gain 1 / lambda_position gives the closed position loop 1 / (lambda_position s + 1).
Gains use the controller units: N per mm/s and N m per deg/s.
"""

import math

import torch
from maglevsim.config import SimConfig

from ..config import CascadedPidConfig, ImcTuningConfig

# Controller units -> SI: mm -> m for translation, deg -> rad for rotation.
_PER_SI = (1e-3,) * 3 + (math.pi / 180.0,) * 3


def rigid_body_axes(plant: SimConfig) -> tuple[torch.Tensor, torch.Tensor]:
  """`(6,)` inertia (kg or kg m^2) and viscous damping (N s/m or N m s/rad) per axis."""
  inertia = torch.tensor([plant.mover_mass] * 3 + list(plant.inertia_diag), dtype=torch.float64)
  damping = torch.tensor(list(plant.linear_damping) + list(plant.angular_damping), dtype=torch.float64)
  return inertia, damping


def imc_cascade_gains(tuning: ImcTuningConfig, plant: SimConfig) -> CascadedPidConfig:
  inertia, damping = rigid_body_axes(plant)
  per_si = torch.tensor(_PER_SI, dtype=torch.float64)
  lambda_velocity = torch.tensor(tuning.lambda_velocity_s, dtype=torch.float64)
  lambda_position = lambda_velocity * tuning.position_to_velocity_ratio
  gravity = -plant.gravity_m_s2[2] * plant.mover_mass
  return CascadedPidConfig(
    position_kp_per_s=tuple((1.0 / lambda_position).tolist()),
    velocity_kp=tuple((inertia / lambda_velocity * per_si).tolist()),
    velocity_ki=tuple((damping / lambda_velocity * per_si).tolist()),
    velocity_kd=(0.0,) * 6,
    wrench_offset=(0.0, 0.0, gravity, 0.0, 0.0, 0.0),  # gravity feed-forward
  )
