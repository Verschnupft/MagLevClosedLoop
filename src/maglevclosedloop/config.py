"""Configuration. `ClosedLoopConfig` is the single entry point; `factory.py` distributes its values to the stages."""

import math
from dataclasses import dataclass, field

from maglevsim.config import GridConfig, SimConfig

Pose = tuple[float, float, float, float, float, float]
Axes = tuple[float, float, float, float, float, float]  # x, y, z, roll, pitch, yaw


@dataclass(frozen=True)
class KalmanConfig:
  """Linear Kalman filter on the estimated pose, as `kf_ca_const_*` in MagLevPoseEstimation (`filtering/ekf.py`).

  SI units per axis (m or rad). Defaults: tuned for `cpp_parallel_taylor_1_element_offsets` on
  `data/filter_tuning_seed20261029/motion` with 4 counts Hall noise
  (`outputs/experiments/cache/filter_tuning/filter_tuning_seed20261029.json`).
  """

  motion_model: str = "ca"  # "cv" (constant velocity) or "ca" (constant acceleration)
  # Q from the controller (filtering/bandwidth.py): filter bandwidth = factor x velocity-loop bandwidth.
  # None uses process_psd below, which is tuned for open-loop estimation and lags too much in the closed loop.
  bandwidth_factor: float | None = 5.0
  process_psd: Axes = (0.0031622776601683794,) * 2 + (3.16227766016838e-05,) + (0.00963284861489642,) * 2 + (2.46740110027234,)
  measurement_std: Axes = (
    1.1056229422976008e-05, 1.101094220165291e-05, 7.978981556683469e-06,
    0.00022124804641882796, 0.0002189659578039589, 0.00026183501017451225,
  )
  initial_rate_std: Axes = (0.1, 0.1, 0.005, math.radians(5.0), math.radians(5.0), math.radians(90.0))
  initial_acceleration_std: Axes = (0.2, 0.2, 0.02, math.radians(20.0), math.radians(20.0), math.radians(180.0))


@dataclass(frozen=True)
class GaussNewtonConfig:
  max_iterations: int = 1  # 1: one first-order step per cycle, as cpp_parallel_taylor_1 in MagLevPoseEstimation
  step_tolerance: float = 1e-10  # norm of the pose update in m and rad; ends the iteration early
  kalman: KalmanConfig = field(default_factory=KalmanConfig)


@dataclass(frozen=True)
class NeuralConfig:
  checkpoint_path: str = "checkpoints/logmlp_seed42.pt"  # relative to the repository root, or absolute
  # Tuned for kf_ca_const_singleframe_log_mlp_700k, same tuning run as the KalmanConfig defaults.
  kalman: KalmanConfig = field(default_factory=lambda: KalmanConfig(
    process_psd=(0.0031622776601683794,) * 2 + (3.1622776601683795e-05,) + (0.009632848614896418,) * 2 + (2.4674011002723395,),
    measurement_std=(
      1.2666369508505687e-05, 1.2598174756307578e-05, 8.371159495135733e-06,
      0.00023530468227360503, 0.00023252118927479657, 0.00030102620063297424,
    ),
  ))


@dataclass(frozen=True)
class LowPassConfig:
  """First-order low-pass on the differenced velocity. Untuned starting value."""

  cutoff_hz: float = 200.0


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
class ImcTuningConfig:
  """IMC design of the cascade (control/imc.py): closed velocity loop 1 / (lambda s + 1) per axis.

  The position loop is `position_to_velocity_ratio` times slower (MagLevWhiteBox: 25 ms / 5 ms).
  """

  # From scripts/tune_controller.py (neural estimator, Kalman filter, 4 counts Hall noise, default scenario).
  lambda_velocity_s: Axes = (5e-3, 5e-3, 2e-3, 5e-3, 5e-3, 5e-3)
  position_to_velocity_ratio: float = 5.0


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
  estimator: str = "gauss_newton"  # "gauss_newton", "neural", or "oracle" (simulator ground truth)
  motion_filter: str = "kalman"  # pose and velocity for the controller: "kalman", "lowpass" or "difference"
  sensor_noise_std_counts: float = 4.0
  coil_field_in_hall: bool = False  # False: Hall data hold only the mover field (forces are unchanged)
  device: str | None = None  # simulator device; None picks CUDA when available
  seed: int = 0

  gauss_newton: GaussNewtonConfig = field(default_factory=GaussNewtonConfig)
  neural: NeuralConfig = field(default_factory=NeuralConfig)
  lowpass: LowPassConfig = field(default_factory=LowPassConfig)
  # Gains from IMC (controller_tuning); None uses the explicit gains in `controller` (MagLevWhiteBox values).
  controller_tuning: ImcTuningConfig | None = field(default_factory=ImcTuningConfig)
  controller: CascadedPidConfig = field(default_factory=CascadedPidConfig)
  actuation: HalbachActuationConfig = field(default_factory=HalbachActuationConfig)

  def plant(self) -> SimConfig:
    return SimConfig(
      grid=GridConfig(rows=self.grid_rows, cols=self.grid_cols),
      sensor_noise_std=self.sensor_noise_std_counts,
      scaling_const_stator=SimConfig.scaling_const_stator if self.coil_field_in_hall else 0.0,  # coil field in Hall data only
      device=self.device,
      default_hover_pose_mmdeg=self.initial_pose_mmdeg,
    )
