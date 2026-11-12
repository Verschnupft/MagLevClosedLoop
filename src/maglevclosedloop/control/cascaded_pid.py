import torch

from ..config import CascadedPidConfig


class CascadedPidController:
  """Per-axis cascade: position P loop -> desired velocity -> velocity PID loop -> wrench.

  Ported from MagLevWhiteBox (`MimoController` of `SisoController`):
  forward-Euler integral, backward-Euler derivative on the measurement with a first-order filter,
  anti-windup by back-calculation with k_aw = ki / kp (tracking time = integral time).
  The velocity comes from the motion filter. Units follow the pose: mm, deg, s.
  """

  def __init__(self, config: CascadedPidConfig, dt_s: float) -> None:
    def axes(values) -> torch.Tensor:
      return torch.tensor(values, dtype=torch.float64)

    self.dt_s = dt_s
    self.position_kp = axes(config.position_kp_per_s)
    self.kp = axes(config.velocity_kp)
    self.ki = axes(config.velocity_ki)
    self.kd = axes(config.velocity_kd)
    self.n = config.derivative_filter_n
    self.wrench_offset = axes(config.wrench_offset)
    self.kaw = torch.nan_to_num(self.ki / self.kp, nan=0.0, posinf=0.0)  # 0 where kp is 0
    self.reset()

  def reset(self) -> None:
    self._previous_velocity = torch.zeros(6, dtype=torch.float64)
    self._previous_error = torch.zeros(6, dtype=torch.float64)
    self._integral = torch.zeros(6, dtype=torch.float64)
    self._derivative = torch.zeros(6, dtype=torch.float64)
    self._previous_wrench = None

  def compute(
    self, pose_mmdeg: torch.Tensor, velocity: torch.Tensor, reference_mmdeg: torch.Tensor, achieved_wrench: torch.Tensor | None
  ) -> torch.Tensor:
    pose = torch.as_tensor(pose_mmdeg, dtype=torch.float64)
    velocity = torch.as_tensor(velocity, dtype=torch.float64)
    reference = torch.as_tensor(reference_mmdeg, dtype=torch.float64)

    velocity_error = self.position_kp * (reference - pose) - velocity

    # Back-calculation: the part of the last wrench the actuation could not realize drains the integral.
    windup = torch.zeros(6, dtype=torch.float64)
    if achieved_wrench is not None and self._previous_wrench is not None:
      windup = torch.as_tensor(achieved_wrench, dtype=torch.float64) - self._previous_wrench
    self._integral = self._integral + self.dt_s * (self.ki * self._previous_error + self.kaw * windup)
    self._previous_error = velocity_error

    decay = 1.0 / (1.0 + self.dt_s * self.n)
    self._derivative = decay * self._derivative - decay * self.kd * self.n * (velocity - self._previous_velocity)
    self._previous_velocity = velocity

    self._previous_wrench = self.wrench_offset + self.kp * velocity_error + self._integral + self._derivative
    return self._previous_wrench
