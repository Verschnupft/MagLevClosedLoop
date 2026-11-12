import torch
from maglevsim.config import SimConfig
from maglevsim.geometry import SENSOR_ELEM_OFFSETS, build_nominal_halbach_mover
from maglevsim.physics import cuboid_field_exact, euler_to_rotation_matrix

from .. import plant as plant_geometry
from ..config import GaussNewtonConfig


class GaussNewtonPoseEstimator:
  """Gauss-Newton inverse of the exact MagLevSim mover-field model on the 50-sensor window under the mover.

  As `cpp_parallel_taylor_1_element_offsets` in MagLevPoseEstimation: finite cuboid magnets, one position
  offset per Hall element, the window around the prior pose, and (default) one first-order step from the
  initial pose. With noise-free Hall data the residual at the true pose is zero ("perfect" model).
  The window follows the mover, so the estimator works for any M x N grid.
  """

  def __init__(self, plant: SimConfig, config: GaussNewtonConfig) -> None:
    self.plant = plant
    self.config = config
    self._elements_m = plant_geometry.sensor_positions_m(plant)[:, None, :] + torch.from_numpy(SENSOR_ELEM_OFFSETS)[None]
    offsets, dims, pols = build_nominal_halbach_mover(plant.magnet_strength_t, plant.magnet_height_m)
    self._magnet_offsets = torch.from_numpy(offsets).double()
    self._magnet_dims = torch.from_numpy(dims).double()
    self._magnet_pols = torch.from_numpy(pols).double()
    self._half_mm = torch.tensor([plant.grid.array_width_m, plant.grid.array_height_m], dtype=torch.float64) * 500.0

  def predict_counts(self, pose: torch.Tensor, elements_m: torch.Tensor | None = None) -> torch.Tensor:
    """Mover field `(sensors * 3,)` in counts, order `[By, Bx, Bz]`, for a world pose (m, rad); all sensors by default."""
    elements_m = self._elements_m if elements_m is None else elements_m
    rotation = euler_to_rotation_matrix(torch.rad2deg(pose[3:]))
    local = (elements_m - pose[:3]) @ rotation  # (sensor, element, xyz)
    field_local = cuboid_field_exact(
      local[..., None, :] - self._magnet_offsets, self._magnet_dims, self._magnet_pols
    ).sum(-2)
    field = (field_local @ rotation.T).diagonal(dim1=-2, dim2=-1)  # element c measures world component c
    return (field[:, (1, 0, 2)] * self.plant.scaling_const).reshape(-1)

  def estimate(self, hall_counts: torch.Tensor, prior_mmdeg: torch.Tensor, initial_mmdeg: torch.Tensor) -> torch.Tensor:
    window = plant_geometry.sensor_window_indices(
      self.plant, plant_geometry.sensor_window_offset_mm(self.plant, torch.as_tensor(prior_mmdeg, dtype=torch.float64)[:2])
    )
    measured = hall_counts.detach().to("cpu", torch.float64).reshape(-1, 3)[window].reshape(-1)
    elements_m = self._elements_m[window]
    pose = self._to_world(torch.as_tensor(initial_mmdeg, dtype=torch.float64))
    for _ in range(self.config.max_iterations):
      jacobian = torch.func.jacfwd(lambda p: self.predict_counts(p, elements_m))(pose)
      step = torch.linalg.lstsq(jacobian, measured - self.predict_counts(pose, elements_m)).solution
      pose = pose + step
      if step.norm() < self.config.step_tolerance:
        break
    return torch.cat([pose[:2] * 1000.0 + self._half_mm, pose[2:3] * 1000.0, torch.rad2deg(pose[3:])])

  def _to_world(self, pose_mmdeg: torch.Tensor) -> torch.Tensor:
    return torch.cat([plant_geometry.global_xy_to_world_m(self.plant, pose_mmdeg[:2]), pose_mmdeg[2:3] / 1000.0, torch.deg2rad(pose_mmdeg[3:])])
