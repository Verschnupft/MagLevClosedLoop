"""Configuration. `ClosedLoopConfig` is the single entry point; `factory.py` distributes its values to the stages."""

import math
from dataclasses import dataclass, field

from maglevsim.config import GridConfig, SimConfig

Pose = tuple[float, float, float, float, float, float]
Axes = tuple[float, float, float, float, float, float]  # x, y, z, roll, pitch, yaw


@dataclass(frozen=True)
class GaussNewtonConfig:
  max_iterations: int = 5
  step_tolerance: float = 1e-10  # norm of the pose update in m and rad


@dataclass(frozen=True)
class KalmanConfig:
  """Constant-velocity Kalman filter. Units per axis: mm or deg. Untuned starting values."""

  measurement_std: Axes = (1e-3,) * 6  # std of the pose estimate
  acceleration_std: Axes = (1e4,) * 6  # per s^2; larger = faster, less smoothing
  initial_velocity_std: Axes = (1.0,) * 6  # per s


@dataclass(frozen=True)
class CascadedPidConfig:
  """Position P loop around a velocity PID loop per axis. Gains are the IMC-tuned values of MagLevWhiteBox."""

  position_kp_per_s: Axes = (40.0, 40.0, 40.2, 40.2, 40.2, 40.0)
  velocity_kp: Axes = (0.2598, 0.2598, 0.6049, 0.0380, 0.0380, 0.3458)
  velocity_ki: Axes = (0.3551, 0.3551, 605.9609, 7.6606, 7.6606, 23.0551)
  velocity_kd: Axes = (0.0, 0.0, 0.4581, 0.0073, 0.0073, 0.0017)
  derivative_filter_n: float = 10.0
  wrench_offset: Axes = (0.0, 0.0, 12.37, 0.0, 0.0, 0.0)  # feed-forward of the z velocity loop operating point


@dataclass(frozen=True)
class HalbachActuationConfig:
  """Wrench to d/q/phi via the four Halbach arrays. Values from MagLevWhiteBox."""

  halbach_offset_large_mm: float = 40.0
  halbach_offset_small_mm: float = 20.0
  halbach_dims_mm: tuple[float, float] = (40.0, 80.0)  # width, length
  coil_dims_mm: tuple[float, float] = (40.0, 120.0)  # width, length
  pole_pair_length_mm: float = 40.0
  coil_ref_offset_mm: float = 20.0
  phi_offset_rad: tuple[float, float] = (-2 / 3 * math.pi, -5 / 3 * math.pi)  # x-coils, y-coils
  force_k_xy_N_per_incr: float = -0.001  # lateral force per q increment
  force_k_z_poly: tuple[float, float, float] = (37.0470 / 4, -0.8496 / 4, 0.0067 / 4)  # k_z(z_m) in N/incr


@dataclass(frozen=True)
class ClosedLoopConfig:
  grid_rows: int = 1
  grid_cols: int = 1
  initial_pose_mmdeg: Pose = (120.0, 120.0, 4.0, 0.0, 0.0, 0.0)  # simulator start and estimator initial guess
  reference_schedule: tuple[tuple[float, Pose], ...] = (  # (start time in s, pose), piecewise constant
    (0.0, (120.0, 120.0, 4.0, 0.0, 0.0, 0.0)),
    (0.05, (122.0, 120.0, 4.0, 0.0, 0.0, 0.0)),
  )
  duration_s: float = 0.1
  estimator: str = "gauss_newton"  # "gauss_newton" or "oracle" (simulator ground truth)
  motion_filter: str = "kalman"  # pose and velocity for the controller: "kalman" or "difference"
  sensor_noise_std_counts: float = 0.0
  device: str | None = None  # simulator device; None picks CUDA when available
  seed: int = 0

  gauss_newton: GaussNewtonConfig = field(default_factory=GaussNewtonConfig)
  kalman: KalmanConfig = field(default_factory=KalmanConfig)
  controller: CascadedPidConfig = field(default_factory=CascadedPidConfig)
  actuation: HalbachActuationConfig = field(default_factory=HalbachActuationConfig)

  def plant(self) -> SimConfig:
    return SimConfig(
      grid=GridConfig(rows=self.grid_rows, cols=self.grid_cols),
      sensor_noise_std=self.sensor_noise_std_counts,
      device=self.device,
      default_hover_pose_mmdeg=self.initial_pose_mmdeg,
    )
