import bisect

import torch


class PiecewiseConstantReference:
  """Setpoint steps: each `(start_time_s, pose)` holds until the next one starts."""

  def __init__(self, schedule) -> None:
    ordered = sorted(schedule, key=lambda item: item[0])
    self._times_s = [time_s for time_s, _ in ordered]
    self._poses = [torch.tensor(pose, dtype=torch.float64) for _, pose in ordered]

  def pose_mmdeg(self, time_s: float) -> torch.Tensor:
    return self._poses[max(bisect.bisect_right(self._times_s, time_s) - 1, 0)]
