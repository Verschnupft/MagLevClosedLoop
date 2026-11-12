import torch


class NoCoilFieldCompensation:
  """Pass Hall data through unchanged (no stator field removed)."""

  def compensate(self, hall_counts: torch.Tensor, coil_currents_abc: torch.Tensor) -> torch.Tensor:
    return hall_counts
