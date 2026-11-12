from pathlib import Path

import torch
from maglevsim import MagLevSim, ZeroController

from .actuation import HalbachDqActuation
from .compensation import NoCoilFieldCompensation
from .config import ClosedLoopConfig
from .control import CascadedPidController
from .estimation import GaussNewtonPoseEstimator, OracleEstimator
from .loop import ClosedLoop
from .motion import FiniteDifferenceFilter, LinearKalmanFilter, LowPassDifferenceFilter
from .neural import NeuralPoseEstimator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
from .reference import PiecewiseConstantReference


def build_closed_loop(config: ClosedLoopConfig) -> ClosedLoop:
  """Default wiring. To exchange a stage, pass another implementation of its interface to `ClosedLoop`."""
  torch.manual_seed(config.seed)
  plant = config.plant()
  sim = MagLevSim(config=plant, controller=ZeroController())

  if config.estimator == "gauss_newton":
    estimator = GaussNewtonPoseEstimator(plant, config.gauss_newton)
  elif config.estimator == "neural":
    estimator = NeuralPoseEstimator.from_checkpoint(plant, REPOSITORY_ROOT / config.neural.checkpoint_path)
  elif config.estimator == "oracle":
    estimator = OracleEstimator(sim.actual_pose_global_mmdeg)
  else:
    raise ValueError(f"Unknown estimator {config.estimator!r}")

  if config.motion_filter == "kalman":
    motion_filter = LinearKalmanFilter(config.kalman, dt_s=plant.dt)
  elif config.motion_filter == "lowpass":
    motion_filter = LowPassDifferenceFilter(config.lowpass, dt_s=plant.dt)
  elif config.motion_filter == "difference":
    motion_filter = FiniteDifferenceFilter(dt_s=plant.dt)
  else:
    raise ValueError(f"Unknown motion filter {config.motion_filter!r}")

  return ClosedLoop(
    sim=sim,
    reference=PiecewiseConstantReference(config.reference_schedule),
    compensation=NoCoilFieldCompensation(),
    estimator=estimator,
    motion_filter=motion_filter,
    controller=CascadedPidController(config.controller, dt_s=plant.dt),
    actuation=HalbachDqActuation(plant, config.actuation),
  )
