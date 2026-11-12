"""Neural pose estimation with networks trained in MagLevPoseEstimation.

A neural estimator has three exchangeable parts:
- `network`: any `torch.nn.Module` that maps standardized Hall data `(B, 150)` to standardized outputs `(B, K)`,
- the input/output normalization statistics,
- `decode`: maps one de-standardized output `(K,)` and the previous pose to a pose `(6,)`.
`NeuralPoseEstimator.from_checkpoint` fills all three from a checkpoint made by `scripts/slim_checkpoint.py`.
"""

from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import torch
from maglevsim.config import SimConfig
from maglevsim.physics import euler_to_rotation_matrix
from torch import nn

from . import plant as plant_geometry

Decode = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "elu": nn.ELU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


class _Block(nn.Module):
  def __init__(self, in_dim: int, out_dim: int, activation: str) -> None:
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)
    self.activation = _ACTIVATIONS[activation]()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.activation(self.linear(x))


class PoseMLP(nn.Module):
  """MLP encoder with a linear single or dual (translation/orientation) head.

  Same parameter names as `PoseModel` with an MLP encoder in MagLevPoseEstimation, so its weights load strictly.
  """

  def __init__(self, hidden_dims: list[int], output_dim: int, activation: str, dual_head: bool, input_dim: int = 150) -> None:
    super().__init__()
    dims = [input_dim, *hidden_dims]
    self.encoder = nn.Module()
    self.encoder.hidden_blocks = nn.ModuleList(_Block(i, o, activation) for i, o in pairwise(dims))
    self.head = nn.Module()
    if dual_head:
      self.head.translation = nn.Linear(dims[-1], 3)
      self.head.orientation = nn.Linear(dims[-1], output_dim - 3)
    else:
      self.head.output = nn.Linear(dims[-1], output_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for block in self.encoder.hidden_blocks:
      x = block(x)
    if hasattr(self.head, "translation"):
      return torch.cat([self.head.translation(x), self.head.orientation(x)], dim=-1)
    return self.head.output(x)


def build_network(model_config: dict) -> nn.Module:
  supported = (
    model_config["kind"] == "mlp" and model_config["head_kind"] == "linear"
    and model_config["dual_head"] and model_config["covariance_head"] == "none"
    and not (model_config["use_batch_norm"] or model_config["use_layer_norm"] or model_config["use_residual_connections"])
  )
  if not supported:
    # shortcut: only the plain MLP with dual head is ported; port further encoders from models.py when needed.
    raise NotImplementedError(f"Network not ported: {model_config['kind']}")
  return PoseMLP(model_config["hidden_dims"], model_config["output_dim"], model_config["activation"], dual_head=True)


def _zyx_euler_deg(rotation: torch.Tensor) -> torch.Tensor:
  """Inverse of `euler_to_rotation_matrix` (R = Rz Ry Rx): `(..., 3, 3)` -> `(..., 3)` roll, pitch, yaw in deg."""
  roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
  pitch = torch.atan2(-rotation[..., 2, 0], torch.hypot(rotation[..., 0, 0], rotation[..., 1, 0]))
  yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
  return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=-1))


def _swing(u: torch.Tensor) -> torch.Tensor:
  """Shortest-arc rotation from e_z to the unit vector `u`."""
  ux, uy, uz = u.unbind()
  w = 1.0 / (1.0 + uz)
  return torch.stack([
    torch.stack([1.0 - ux * ux * w, -ux * uy * w, ux]),
    torch.stack([-ux * uy * w, 1.0 - uy * uy * w, uy]),
    torch.stack([-ux, -uy, uz]),
  ])


def decode_unit_vector(uz_from_tilt: bool) -> Decode:
  """Decode `[x, y, z, ux, uy, uz, cos 4tau, sin 4tau]` (`PoseUnitVectorTransform`).

  `u` is the body z-axis, `tau` the twist about it. The 90-degree mover symmetry leaves four candidates;
  the one with yaw nearest the previous yaw wins, and yaw stays continuous.
  """

  def decode(encoded: torch.Tensor, previous_pose: torch.Tensor) -> torch.Tensor:
    u = encoded[3:6]
    if uz_from_tilt:  # the uz output is ignored; valid for tilt below 90 deg
      u = torch.cat([u[:2], (1.0 - u[:2].square().sum()).clamp_min(0.0).sqrt()[None]])
    u = u / u.norm().clamp_min(1e-8)
    tau_deg = torch.rad2deg(torch.atan2(encoded[7], encoded[6]) / 4) + 90.0 * torch.arange(4, dtype=encoded.dtype)
    twist = euler_to_rotation_matrix(torch.stack([torch.zeros(4), torch.zeros(4), tau_deg], -1).to(encoded))
    angles = _zyx_euler_deg(_swing(u) @ twist)
    yaw_step = torch.remainder(angles[:, 2] - previous_pose[5] + 180.0, 360.0) - 180.0
    best = yaw_step.abs().argmin()
    return torch.cat([encoded[:3], angles[best, :2], (previous_pose[5] + yaw_step[best])[None]])

  return decode


class NeuralPoseEstimator:
  """Run a network on the 50-sensor window under the mover; works for any M x N grid.

  Input: Hall counts in MagLevSim order `[By, Bx, Bz]`, as in training. Outputs are in the window frame;
  the window offset converts them to the global frame. The previous pose selects the window and the yaw branch.
  """

  def __init__(
    self,
    plant: SimConfig,
    network: nn.Module,
    normalization: dict[str, torch.Tensor],
    decode: Decode,
  ) -> None:
    self.plant = plant
    self.network = network.eval()
    self.normalization = normalization
    self.decode = decode

  @classmethod
  def from_checkpoint(cls, plant: SimConfig, path: str | Path) -> "NeuralPoseEstimator":
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    network = build_network(checkpoint["model"])
    network.load_state_dict(checkpoint["state_dict"])
    output_dim = checkpoint["model"]["output_dim"]
    if output_dim == 8:
      decode = decode_unit_vector(checkpoint["data"]["uz_from_tilt"])
    elif output_dim == 6:  # plain [x, y, z, roll, pitch, yaw] without symmetry handling
      decode = lambda encoded, previous_pose: encoded
    else:
      raise NotImplementedError(f"No decoding for output_dim {output_dim}")
    return cls(plant, network, checkpoint["normalization"], decode)

  def reset(self, initial_pose_mmdeg: torch.Tensor) -> None:
    self._pose = torch.as_tensor(initial_pose_mmdeg, dtype=torch.float64).clone()

  def estimate(self, hall_counts: torch.Tensor) -> torch.Tensor:
    offset = plant_geometry.sensor_window_offset_mm(self.plant, self._pose[:2])
    window = hall_counts.detach().cpu().reshape(-1, 3)[plant_geometry.sensor_window_indices(self.plant, offset)]
    n = self.normalization
    with torch.no_grad():
      output = self.network((window.reshape(1, -1).float() - n["input_mean"]) / n["input_std"])[0]
    encoded = (output * n["output_std"] + n["output_mean"]).double()
    previous = self._pose.clone()
    previous[:2] -= offset
    pose = self.decode(encoded, previous)
    pose[:2] += offset
    self._pose = pose
    return pose.clone()
