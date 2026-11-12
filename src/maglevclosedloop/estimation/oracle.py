from collections.abc import Callable

import torch


class OracleEstimator:
  """Return the simulator ground truth. Reference for testing the other stages only."""

  def __init__(self, true_pose_mmdeg: Callable[[], torch.Tensor]) -> None:
    self._true_pose_mmdeg = true_pose_mmdeg

  def estimate(self, hall_counts: torch.Tensor, prior_mmdeg: torch.Tensor, initial_mmdeg: torch.Tensor) -> torch.Tensor:
    return self._true_pose_mmdeg().detach().to("cpu", torch.float64)
