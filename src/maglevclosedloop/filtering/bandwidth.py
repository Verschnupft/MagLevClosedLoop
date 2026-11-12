"""Process noise Q of the Kalman filter from the bandwidth of the velocity control loop.

The filter sits inside the control loop, so its phase lag must stay small at the loop bandwidth.
Rule: filter bandwidth = `bandwidth_factor` x velocity-loop bandwidth, per axis.

Velocity-loop bandwidth (P part, rigid body): omega_c = kp / m (translation) or kp / I (rotation).
Steady-state Kalman filter with white noise of order n (n = 2 for CV, 3 for CA) and measurement noise r:
its poles form a Butterworth pattern with natural frequency omega_f = (q / (r * dt))^(1 / (2n)),
so q = r * dt * omega_f^(2n). "Bandwidth" here means this natural frequency.
"""

import dataclasses
import math

import torch
from maglevsim.config import SimConfig

from ..config import CascadedPidConfig, KalmanConfig

MAX_BANDWIDTH_FRACTION_OF_NYQUIST = 0.25  # keeps omega_f * dt small, where the continuous-time rule holds


def velocity_loop_bandwidth_rad_s(controller: CascadedPidConfig, plant: SimConfig) -> torch.Tensor:
  """`(6,)` approximate crossover of the velocity loops; ignores I and D parts and the transport delay."""
  per_si_unit = torch.tensor([1000.0] * 3 + [180.0 / math.pi] * 3, dtype=torch.float64)  # kp per mm/s or per deg/s -> per SI unit
  inertia = torch.tensor([plant.mover_mass] * 3 + list(plant.inertia_diag), dtype=torch.float64)
  return torch.tensor(controller.velocity_kp, dtype=torch.float64) * per_si_unit / inertia


def process_psd_for_bandwidth(bandwidth_rad_s: torch.Tensor, config: KalmanConfig, dt_s: float) -> tuple[float, ...]:
  order = {"cv": 2, "ca": 3}[config.motion_model]
  bandwidth = torch.clamp(bandwidth_rad_s, max=MAX_BANDWIDTH_FRACTION_OF_NYQUIST * math.pi / dt_s)
  r = torch.tensor(config.measurement_std, dtype=torch.float64) ** 2
  return tuple((r * dt_s * bandwidth ** (2 * order)).tolist())


def kalman_config_for_controller(config: KalmanConfig, controller: CascadedPidConfig, plant: SimConfig) -> KalmanConfig:
  """Return `config` with Q set from the controller, or unchanged when `bandwidth_factor` is None."""
  if config.bandwidth_factor is None:
    return config
  bandwidth = config.bandwidth_factor * velocity_loop_bandwidth_rad_s(controller, plant)
  return dataclasses.replace(config, process_psd=process_psd_for_bandwidth(bandwidth, config, plant.dt))
