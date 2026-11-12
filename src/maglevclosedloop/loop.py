from dataclasses import dataclass

import torch
from maglevsim import MagLevSim

from .interfaces import Actuation, CoilFieldCompensation, Controller, MotionFilter, PoseEstimator, Reference


@dataclass
class LoopLog:
  """Per-cycle signals, each `(cycles, ...)`. `measured_pose` is the true pose at Hall sampling time."""

  time_s: torch.Tensor
  reference: torch.Tensor
  measured_pose: torch.Tensor
  estimated_pose: torch.Tensor
  filtered_pose: torch.Tensor
  velocity: torch.Tensor
  wrench: torch.Tensor
  command: torch.Tensor


class ClosedLoop:
  """Wire the stages around the simulator. This is the only class that calls MagLevSim."""

  def __init__(
    self,
    sim: MagLevSim,
    reference: Reference,
    compensation: CoilFieldCompensation,
    estimator: PoseEstimator,
    motion_filter: MotionFilter,
    controller: Controller,
    actuation: Actuation,
  ) -> None:
    self.sim = sim
    self.reference = reference
    self.compensation = compensation
    self.estimator = estimator
    self.motion_filter = motion_filter
    self.controller = controller
    self.actuation = actuation

  def run(self, cycles: int, initial_pose_mmdeg) -> LoopLog:
    initial_pose = torch.as_tensor(initial_pose_mmdeg, dtype=torch.float64)
    self.sim.reset(initial_pose)
    self.motion_filter.reset(initial_pose)
    pose = initial_pose  # the first prior is the known start pose
    self.controller.reset()

    command = torch.zeros((self.sim.tile_count, 24, 3), dtype=torch.float64)
    achieved_wrench = None
    rows = {name: [] for name in LoopLog.__dataclass_fields__}
    for cycle in range(cycles):
      time_s = cycle * self.sim.config.dt
      reference = self.reference.pose_mmdeg(time_s)
      # The simulator samples Hall data at the pose before integrating this cycle.
      measured_pose = self.sim.actual_pose_global_mmdeg().detach().cpu().double()
      result = self.sim.step(reference, command_dqphi=command)

      hall = self.compensation.compensate(result.hall_sensor_measurements, result.currents_abc)
      # As FilteredPoseEstimator in MagLevPoseEstimation: predict, estimate from the prediction, update.
      estimated_pose = self.estimator.estimate(hall, pose, self.motion_filter.predict())
      pose, velocity = self.motion_filter.update(estimated_pose)
      wrench = self.controller.compute(pose, velocity, reference, achieved_wrench)
      command, achieved_wrench = self.actuation.compute(wrench, pose)

      values = (torch.tensor(time_s), reference, measured_pose, estimated_pose, pose, velocity, wrench, command)
      for name, value in zip(rows, values, strict=True):
        rows[name].append(value)
    return LoopLog(**{name: torch.stack(values) for name, values in rows.items()})
