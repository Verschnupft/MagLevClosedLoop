"""Choose the IMC time constants lambda per axis group by closed-loop simulation.

Usage: python scripts/tune_controller.py [estimator] [motion_filter]   (default: neural kalman)

Method: one pass of coordinate search over the groups xy, z, tilt, yaw. For each candidate lambda the
loop runs the default `ClosedLoopConfig` scenario (hover, then a 2 mm x step) with Hall noise. A candidate
is feasible when no requested coil command exceeds the MagLevSim limit; among feasible candidates the
smallest RMS tracking error of the group axes wins. The Kalman Q follows each candidate (bandwidth rule).
"""

import dataclasses
import sys

import torch

from maglevclosedloop.config import ClosedLoopConfig, ImcTuningConfig
from maglevclosedloop.control import CascadedPidController, imc_cascade_gains
from maglevclosedloop.factory import build_closed_loop
from maglevclosedloop.filtering import LinearKalmanFilter, kalman_config_for_controller

GROUPS = {"xy": (0, 1), "z": (2,), "tilt": (3, 4), "yaw": (5,)}
CANDIDATES_S = (1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3)


def evaluate(loop, config: ClosedLoopConfig, lambdas: tuple[float, ...], axes: tuple[int, ...]) -> tuple[float, float]:
  plant = config.plant()
  gains = imc_cascade_gains(dataclasses.replace(config.controller_tuning, lambda_velocity_s=lambdas), plant)
  loop.controller = CascadedPidController(gains, dt_s=plant.dt)
  if config.motion_filter == "kalman":
    kalman = config.neural.kalman if config.estimator == "neural" else config.gauss_newton.kalman
    loop.motion_filter = LinearKalmanFilter(kalman_config_for_controller(kalman, gains, plant), dt_s=plant.dt)
  torch.manual_seed(config.seed)
  log = loop.run(round(config.duration_s / plant.dt), config.initial_pose_mmdeg)
  rms = (log.measured_pose - log.reference)[:, axes].square().mean().sqrt().item()
  peak_command = log.command[..., :2].norm(dim=-1).max().item()
  return rms, peak_command


def main(estimator: str = "neural", motion_filter: str = "kalman") -> None:
  config = ClosedLoopConfig(estimator=estimator, motion_filter=motion_filter, controller_tuning=ImcTuningConfig())
  loop = build_closed_loop(config)
  limit = config.plant().coil_limit_incr
  lambdas = list(config.controller_tuning.lambda_velocity_s)
  for group, axes in GROUPS.items():
    results = []
    for candidate in CANDIDATES_S:
      trial = list(lambdas)
      for axis in axes:
        trial[axis] = candidate
      rms, peak = evaluate(loop, config, tuple(trial), axes)
      results.append((candidate, rms, peak))
      print(f"{group:5s} lambda {candidate * 1e3:5.1f} ms  rms {rms:.4f}  peak command {peak:9.0f}", flush=True)
    feasible = [r for r in results if r[2] <= limit]
    best = min(feasible, key=lambda r: r[1]) if feasible else min(results, key=lambda r: r[2])
    for axis in axes:
      lambdas[axis] = best[0]
    print(f"-> {group}: lambda {best[0] * 1e3:.1f} ms ({'feasible' if feasible else 'no feasible candidate, least saturation'})")
  print("lambda_velocity_s =", tuple(lambdas))


if __name__ == "__main__":
  main(*sys.argv[1:3])
