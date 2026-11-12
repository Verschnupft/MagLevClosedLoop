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


def global_xy_to_world_m(plant: SimConfig, xy_mm: torch.Tensor) -> torch.Tensor:
  """Global array coordinates (mm, origin at the lower-left array corner) to world metres."""
  half = torch.tensor([plant.grid.array_width_m, plant.grid.array_height_m], dtype=xy_mm.dtype) / 2.0
  return xy_mm / 1000.0 - half
