"""Nominal plant geometry taken from MagLevSim.

This is the only module that depends on MagLevSim internals (underscore helpers).
All values use the MagLevSim world frame: metres, origin at the centre of the tile array.
"""

import torch
from maglevsim.config import SimConfig
from maglevsim.geometry import _build_sensor_positions, _stator_unit_coords, _tile_centers


def tile_centers_m(plant: SimConfig) -> torch.Tensor:
  """`(tile_count, 3)` tile centres, MagLevSim tile order (top-left first, row-major)."""
  return torch.from_numpy(_tile_centers(plant.grid)[0]).double()


def sensor_positions_m(plant: SimConfig) -> torch.Tensor:
  """`(tile_count * 50, 3)` Hall sensor positions."""
  local = torch.from_numpy(_build_sensor_positions(plant.grid.tile_size_m, plant.sensor_z_m))
  return (tile_centers_m(plant)[:, None, :] + local[None]).reshape(-1, 3)


def coil_centers_m(plant: SimConfig) -> tuple[torch.Tensor, torch.Tensor]:
  """`(tile_count * 24, 2)` coil unit centres and `(tile_count * 24,)` mask of x-coils (drive X1/X3)."""
  units = _stator_unit_coords()
  local = torch.tensor([(x, y) for x, y, _ in units], dtype=torch.float64)
  is_x_coil = torch.tensor([not vertical for _, _, vertical in units])
  centers = tile_centers_m(plant)[:, None, :2] + local[None]
  return centers.reshape(-1, 2), is_x_coil.repeat(plant.grid.tile_count)


LATTICE_PITCH_MM = 24.0
WINDOW_CENTER_MM = 120.0


def sensor_window_offset_mm(plant: SimConfig, xy_mm: torch.Tensor) -> torch.Tensor:
  """Shift `(2,)` of the single-tile 50-sensor window whose centre is nearest to `xy_mm` and stays on the grid.

  The sensor lattice repeats under shifts (24 a, 24 b) mm with a + b even, so every such window
  has exactly the single-tile sensor layout. Ported from MagLevPoseEstimation `util/sensor_window.py`.
  """
  upper = torch.tensor([plant.grid.cols - 1, plant.grid.rows - 1], dtype=torch.float64) * 10
  target = torch.minimum(torch.clamp((xy_mm.double() - WINDOW_CENTER_MM) / LATTICE_PITCH_MM, min=0), upper)
  corners = torch.tensor([(0, 0), (1, 0), (0, 1), (1, 1)], dtype=torch.float64)
  candidates = torch.minimum(target.floor() + corners, upper)
  distance = (candidates - target).norm(dim=-1).masked_fill(candidates.sum(-1) % 2 == 1, float("inf"))
  return candidates[distance.argmin()] * LATTICE_PITCH_MM


def sensor_window_indices(plant: SimConfig, offset_mm: torch.Tensor) -> torch.Tensor:
  """`(50,)` indices into the flat sensor array `(tile_count * 50,)` for the window shifted by `offset_mm`."""
  half_mm = torch.tensor([plant.grid.array_width_m, plant.grid.array_height_m], dtype=torch.float64) * 500.0
  key = {tuple(p): i for i, p in enumerate(torch.round((sensor_positions_m(plant)[:, :2] * 1000 + half_mm) / 12).long().tolist())}
  local_mm = torch.from_numpy(_build_sensor_positions(plant.grid.tile_size_m, plant.sensor_z_m))[:, :2] * 1000 + plant.grid.tile_size_m * 500
  return torch.tensor([key[tuple(p)] for p in torch.round((local_mm + offset_mm) / 12).long().tolist()])


def global_xy_to_world_m(plant: SimConfig, xy_mm: torch.Tensor) -> torch.Tensor:
  """Global array coordinates (mm, origin at the lower-left array corner) to world metres."""
  half = torch.tensor([plant.grid.array_width_m, plant.grid.array_height_m], dtype=xy_mm.dtype) / 2.0
  return xy_mm / 1000.0 - half
