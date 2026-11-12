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
    magnets = tuple(torch.from_numpy(array).double() for array in (offsets, dims, pols))
    self._magnets = {"cpu": magnets}
    self._half_mm = torch.tensor([plant.grid.array_width_m, plant.grid.array_height_m], dtype=torch.float64) * 500.0

  def predict_counts(self, pose: torch.Tensor, elements_m: torch.Tensor | None = None) -> torch.Tensor:
    """Mover field `(sensors * 3,)` in counts, order `[By, Bx, Bz]`, for a world pose (m, rad); all sensors by default."""
    elements_m = self._elements_m if elements_m is None else elements_m
    offsets, dims, pols = self._magnets[pose.device.type]
    rotation = euler_to_rotation_matrix(torch.rad2deg(pose[3:]))
    local = (elements_m - pose[:3]) @ rotation  # (sensor, element, xyz)
    field_local = cuboid_field_exact(local[..., None, :] - offsets, dims, pols).sum(-2)
    field = (field_local @ rotation.T).diagonal(dim1=-2, dim2=-1)  # element c measures world component c
    return (torch.stack([field[:, 1], field[:, 0], field[:, 2]], -1) * self.plant.scaling_const).reshape(-1)  # no index tensor: capturable

  def estimate(self, hall_counts: torch.Tensor, prior_mmdeg: torch.Tensor, initial_mmdeg: torch.Tensor) -> torch.Tensor:
    window = plant_geometry.sensor_window_indices(
      self.plant, plant_geometry.sensor_window_offset_mm(self.plant, torch.as_tensor(prior_mmdeg, dtype=torch.float64)[:2])
    )
    measured = hall_counts.detach().to("cpu", torch.float64).reshape(-1, 3)[window].reshape(-1)
    elements_m = self._elements_m[window]
    pose = self._to_world(torch.as_tensor(initial_mmdeg, dtype=torch.float64))
    for _ in range(self.config.max_iterations):
      jacobian, field = self._linearize(pose, elements_m)
      step = torch.linalg.lstsq(jacobian, measured - field).solution
      pose = pose + step
      if step.norm() < self.config.step_tolerance:
        break
    return torch.cat([pose[:2] * 1000.0 + self._half_mm, pose[2:3] * 1000.0, torch.rad2deg(pose[3:])])

  def _linearize_eager(self, pose: torch.Tensor, elements_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobian `(150, 6)` and field `(150,)` at `pose` in one forward-mode pass."""
    return torch.func.jacfwd(lambda p: (self.predict_counts(p, elements_m),) * 2, has_aux=True)(pose)

  def _linearize(self, pose: torch.Tensor, elements_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """On CUDA a captured graph replays the linearization; the window always has 50 sensors, so shapes stay fixed."""
    if not self.config.cuda_graph or not torch.cuda.is_available():
      return self._linearize_eager(pose, elements_m)
    if not hasattr(self, "_graph"):
      device = torch.device("cuda")
      self._magnets["cuda"] = tuple(tensor.to(device) for tensor in self._magnets["cpu"])
      self._graph_pose, self._graph_elements = pose.to(device), elements_m.to(device)
      stream = torch.cuda.Stream()
      stream.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(stream):
        for _ in range(3):  # warm-up before capture
          self._linearize_eager(self._graph_pose, self._graph_elements)
      torch.cuda.current_stream().wait_stream(stream)
      self._graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(self._graph):
        self._graph_jacobian, self._graph_field = self._linearize_eager(self._graph_pose, self._graph_elements)
    self._graph_pose.copy_(pose)
    self._graph_elements.copy_(elements_m)
    self._graph.replay()
    return self._graph_jacobian.cpu(), self._graph_field.cpu()

  def _to_world(self, pose_mmdeg: torch.Tensor) -> torch.Tensor:
    return torch.cat([plant_geometry.global_xy_to_world_m(self.plant, pose_mmdeg[:2]), pose_mmdeg[2:3] / 1000.0, torch.deg2rad(pose_mmdeg[3:])])
