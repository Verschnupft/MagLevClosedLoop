import torch
from maglevsim.config import SimConfig

from .. import plant as plant_geometry
from ..config import HalbachActuationConfig


class HalbachDqActuation:
  """Wrench -> eight Halbach array forces -> d/q increments and electrical angle per coil.

  Ported from MagLevWhiteBox (`ForceTransform`, `BaseWhiteBoxController`).
  Array order X1, Y2, X3, Y4; force order Fy1, Fz1, Fx2, Fz2, Fy3, Fz3, Fx4, Fz4.
  A coil drives an array when its centre lies under that array. Coils of all tiles are
  candidates, so the stage works for any M x N grid. Mover rotation is ignored for coil selection and angle.
  Also returns the wrench the limited currents achieve, for the anti-windup of the controller.
  """

  def __init__(self, plant: SimConfig, config: HalbachActuationConfig) -> None:
    self.config = config
    self.tile_count = plant.grid.tile_count
    self.coil_limit_incr = plant.coil_limit_incr
    world_centers_m, is_x_coil = plant_geometry.coil_centers_m(plant)
    half_mm = torch.tensor([plant.grid.array_width_m, plant.grid.array_height_m], dtype=torch.float64) * 500.0
    self._coil_centers_mm = world_centers_m * 1000.0 + half_mm  # global frame, like the pose
    self._is_x_coil = is_x_coil

    large, small = config.halbach_offset_large_mm, config.halbach_offset_small_mm
    self._array_offsets_mm = torch.tensor([(-small, large), (-large, -small), (small, -large), (large, small)], dtype=torch.float64)
    self._array_is_x = torch.tensor([True, False, True, False])

    tolerance_mm = 0.01
    along = (config.coil_dims_mm[1] + config.halbach_dims_mm[1]) / 2.0 - tolerance_mm
    across = (config.coil_dims_mm[0] + config.halbach_dims_mm[0]) / 2.0 - tolerance_mm
    x_array = torch.tensor([along, across], dtype=torch.float64)
    self._thresholds_mm = torch.stack([x_array, x_array.flip(0), x_array, x_array.flip(0)])

  def compute(self, wrench: torch.Tensor, pose_mmdeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pose = torch.as_tensor(pose_mmdeg, dtype=torch.float64)
    force_to_wrench = self._force_to_wrench(pose[2].item() / 1000.0)
    forces = torch.linalg.pinv(force_to_wrench) @ torch.as_tensor(wrench, dtype=torch.float64)
    lateral, vertical = forces[0::2], forces[1::2]  # per array

    delta_mm = (pose[:2] + self._array_offsets_mm)[:, None, :] - self._coil_centers_mm[None]  # (array, coil, xy)
    below = (delta_mm.abs() < self._thresholds_mm[:, None, :]).all(-1)
    below &= self._is_x_coil[None] == self._array_is_x[:, None]

    # All coils under one array carry the same d and q, so the coil limit applies per array.
    # Lift (d) first, lateral (q) gets the rest: |(d, q)| <= coil_limit_incr, as MagLevSim projects.
    k_z = sum(c * (pose[2] / 1000.0) ** p for c, p in zip(self.config.force_k_z_poly, (2, 1, 0), strict=True))
    k_xy = self.config.force_k_xy_N_per_incr
    d_array = torch.clamp(vertical / k_z, -self.coil_limit_incr, self.coil_limit_incr)
    q_room = (self.coil_limit_incr**2 - d_array.square()).clamp_min(0.0).sqrt()
    q_array = torch.maximum(torch.minimum(lateral / k_xy, q_room), -q_room)
    q = (below * q_array[:, None]).sum(0)
    d = (below * d_array[:, None]).sum(0)

    # shortcut: the per-tile power derating of MagLevSim is not mirrored; it acts only far above hover power.
    has_coil = below.any(dim=1)
    achieved_forces = torch.stack([q_array * k_xy * has_coil, d_array * k_z * has_coil], dim=-1).reshape(-1)
    achieved_wrench = force_to_wrench @ achieved_forces

    # X arrays commutate along y, Y arrays along x.
    commutation_mm = torch.where(self._array_is_x[:, None], delta_mm[..., 1], delta_mm[..., 0])
    offsets = torch.tensor(self.config.phi_offset_rad, dtype=torch.float64)[(~self._array_is_x).long()]
    phi = 2 * torch.pi / self.config.pole_pair_length_mm * (commutation_mm - self.config.coil_ref_offset_mm) + offsets[:, None]
    phi = torch.remainder((below * phi).sum(0), 2 * torch.pi)

    return torch.stack([d, q, phi], dim=-1).reshape(self.tile_count, 24, 3), achieved_wrench

  def _force_to_wrench(self, h_m: float) -> torch.Tensor:
    a = self.config.halbach_offset_large_mm / 1000.0
    b = self.config.halbach_offset_small_mm / 1000.0
    return torch.tensor([
      [0, 0, 1, 0, 0, 0, 1, 0],
      [1, 0, 0, 0, 1, 0, 0, 0],
      [0, 1, 0, 1, 0, 1, 0, 1],
      [h_m, a, 0, -b, h_m, -a, 0, b],
      [0, b, -h_m, a, 0, -b, -h_m, -a],
      [-b, 0, b, 0, b, 0, -b, 0],
    ], dtype=torch.float64)
