from collections.abc import Callable

import torch
from maglevsim.config import SimConfig
from maglevsim.geometry import SENSOR_ELEM_OFFSETS, build_nominal_halbach_mover
from maglevsim.physics import cuboid_field_exact, euler_to_rotation_matrix

from . import plant as plant_geometry
from .config import GaussNewtonConfig


class GaussNewtonPoseEstimator:
  """Iterative Gauss-Newton inverse of the exact MagLevSim mover-field model.

  The forward model equals the simulator: finite cuboid magnets and one position offset per Hall element.
  With noise-free, compensated Hall data the residual at the true pose is zero ("perfect" model).
  The previous estimate is the start value, which selects the branch of the 40 mm periodic field.
  All tiles' sensors are used, so the estimator works for any M x N grid.
  """

  def __init__(self, plant: SimConfig, config: GaussNewtonConfig) -> None:
    self.plant = plant
    self.config = config
    self._elements_m = plant_geometry.sensor_positions_m(plant)[:, None, :] + torch.from_numpy(SENSOR_ELEM_OFFSETS)[None]
    offsets, dims, pols = build_nominal_halbach_mover(plant.magnet_strength_t, plant.magnet_height_m)
    self._magnet_offsets = torch.from_numpy(offsets).double()
    self._magnet_dims = torch.from_numpy(dims).double()
    self._magnet_pols = torch.from_numpy(pols).double()
    self._pose = torch.zeros(6, dtype=torch.float64)  # world m and rad

  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None:
    pose = torch.as_tensor(initial_pose_mmdeg, dtype=torch.float64)
    self._pose = torch.cat([
      plant_geometry.global_xy_to_world_m(self.plant, pose[:2]),
      pose[2:3] / 1000.0,
      torch.deg2rad(pose[3:]),
    ])

  def predict_counts(self, pose: torch.Tensor) -> torch.Tensor:
    """Mover field `(sensors * 3,)` in counts, order `[By, Bx, Bz]`, for a world pose (m, rad)."""
    rotation = euler_to_rotation_matrix(torch.rad2deg(pose[3:]))
    local = (self._elements_m - pose[:3]) @ rotation  # (sensor, element, xyz)
    field_local = cuboid_field_exact(
      local[..., None, :] - self._magnet_offsets, self._magnet_dims, self._magnet_pols
    ).sum(-2)
    field = (field_local @ rotation.T).diagonal(dim1=-2, dim2=-1)  # element c measures world component c
    return (field[:, (1, 0, 2)] * self.plant.scaling_const).reshape(-1)

  def estimate(self, hall_counts: torch.Tensor) -> torch.Tensor:
    measured = hall_counts.detach().to("cpu", torch.float64).reshape(-1)
    pose = self._pose
    for _ in range(self.config.max_iterations):
      jacobian = torch.func.jacfwd(self.predict_counts)(pose)
      step = torch.linalg.lstsq(jacobian, measured - self.predict_counts(pose)).solution
      pose = pose + step
      if step.norm() < self.config.step_tolerance:
        break
    self._pose = pose
    return self.pose_mmdeg()

  def pose_mmdeg(self) -> torch.Tensor:
    half_mm = torch.tensor([self.plant.grid.array_width_m, self.plant.grid.array_height_m], dtype=torch.float64) * 500.0
    return torch.cat([self._pose[:2] * 1000.0 + half_mm, self._pose[2:3] * 1000.0, torch.rad2deg(self._pose[3:])])


class OracleEstimator:
  """Return the simulator ground truth. Reference for testing the other stages only."""

  def __init__(self, true_pose_mmdeg: Callable[[], torch.Tensor]) -> None:
    self._true_pose_mmdeg = true_pose_mmdeg

  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None:
    pass

  def estimate(self, hall_counts: torch.Tensor) -> torch.Tensor:
    return self._true_pose_mmdeg().detach().to("cpu", torch.float64)
