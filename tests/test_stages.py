import dataclasses
import math

import pytest
import torch
from maglevsim import MagLevSim, ZeroController

from maglevclosedloop.actuation import HalbachDqActuation
from maglevclosedloop.config import CascadedPidConfig, ClosedLoopConfig, ImcTuningConfig, KalmanConfig, LowPassConfig
from maglevclosedloop.control import imc_cascade_gains
from maglevclosedloop.estimation import GaussNewtonPoseEstimator, NeuralPoseEstimator
from maglevclosedloop.factory import REPOSITORY_ROOT
from maglevclosedloop.filtering import FiniteDifferenceFilter, LinearKalmanFilter, LowPassDifferenceFilter
from maglevclosedloop.filtering.bandwidth import process_psd_for_bandwidth

HOVER = (120.0, 120.0, 4.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("make_filter", [
  lambda dt: LinearKalmanFilter(KalmanConfig(), dt),
  lambda dt: LinearKalmanFilter(KalmanConfig(motion_model="cv"), dt),
  lambda dt: LowPassDifferenceFilter(LowPassConfig(), dt),
  lambda dt: FiniteDifferenceFilter(dt),
])
def test_motion_filter_tracks_a_constant_velocity_ramp(make_filter):
  dt = 250e-6
  velocity = torch.tensor([10.0, -5.0, 1.0, 0.5, -0.5, 20.0], dtype=torch.float64)
  start = torch.tensor(HOVER, dtype=torch.float64)
  motion_filter = make_filter(dt)
  motion_filter.reset(start)
  steps = 2000  # 0.5 s; the tuned CA filter smooths strongly and settles slowly in z and tilt
  for k in range(1, steps + 1):
    pose, estimated_velocity = motion_filter.update(start + velocity * k * dt)
  assert torch.allclose(estimated_velocity, velocity, rtol=1e-3)
  assert torch.allclose(pose, start + velocity * steps * dt, atol=1e-4)


@pytest.mark.parametrize("make_filter", [
  lambda dt, noise_std: LinearKalmanFilter(KalmanConfig(), dt),
  lambda dt, noise_std: LowPassDifferenceFilter(LowPassConfig(), dt),
])
def test_smoothing_filters_reduce_velocity_noise(make_filter):
  dt = 250e-6
  torch.manual_seed(0)
  start = torch.tensor(HOVER, dtype=torch.float64)
  noise_std = 1e-3
  smoothing = make_filter(dt, noise_std)
  difference = FiniteDifferenceFilter(dt)
  smoothing.reset(start)
  difference.reset(start)
  smoothed_velocity, difference_velocity = [], []
  for _ in range(2000):
    measurement = start + noise_std * torch.randn(6, dtype=torch.float64)
    smoothed_velocity.append(smoothing.update(measurement)[1])
    difference_velocity.append(difference.update(measurement)[1])
  assert torch.stack(smoothed_velocity[1000:]).std() < 0.5 * torch.stack(difference_velocity[1000:]).std()


def test_actuation_lift_only_uses_d_current_under_the_arrays():
  config = ClosedLoopConfig()
  actuation = HalbachDqActuation(config.plant(), config.actuation)
  command = actuation.compute(torch.tensor([0.0, 0.0, 12.0, 0.0, 0.0, 0.0]), torch.tensor(HOVER))
  assert command.shape == (1, 24, 3)
  active = command[..., 0] != 0
  assert 0 < active.sum() < 24
  assert torch.allclose(command[..., 1], torch.zeros(1, 24, dtype=torch.float64), atol=1e-9)  # pinv round-off
  assert torch.all(command[active][:, 0] > 0)


def test_actuation_on_a_grid_matches_the_single_tile_pattern():
  single = ClosedLoopConfig()
  grid = ClosedLoopConfig(grid_rows=2, grid_cols=3)
  wrench = torch.tensor([0.5, -0.3, 12.0, 0.01, -0.02, 0.005])
  reference = HalbachDqActuation(single.plant(), single.actuation).compute(wrench, torch.tensor(HOVER))
  # Tile (row 0, col 1) is the top-middle tile; its centre is at (360, 360) mm.
  command = HalbachDqActuation(grid.plant(), grid.actuation).compute(wrench, torch.tensor((360.0, 360.0, 4.0, 0, 0, 0)))
  assert torch.allclose(command[1], reference[0], atol=1e-9)
  assert torch.all(command[[0, 2, 3, 4, 5], :, :2] == 0)


PRIOR_ERROR = torch.tensor([1.0, -1.0, 0.3, 0.2, 0.2, -1.0])


def make_estimator(name: str, config: ClosedLoopConfig):
  if name == "gauss_newton":
    return GaussNewtonPoseEstimator(config.plant(), config.gauss_newton)
  return NeuralPoseEstimator.from_checkpoint(config.plant(), REPOSITORY_ROOT / config.neural.checkpoint_path)


TOLERANCE = {"gauss_newton": 1e-3, "neural": 0.1}  # mm and deg; the network is trained, not exact


@pytest.mark.parametrize("name", ["gauss_newton", "neural"])
def test_estimators_work_on_a_grid(name):
  pose = (300.0, 250.0, 3.5, -0.4, 0.2, 5.0)
  config = ClosedLoopConfig(grid_rows=2, grid_cols=3)
  forward_model = GaussNewtonPoseEstimator(config.plant(), config.gauss_newton)  # exact mover field, like MagLevSim
  forward_model.reset(torch.tensor(pose))
  hall = forward_model.predict_counts(forward_model._pose).reshape(6, 50, 3)
  estimator = make_estimator(name, config)
  estimator.reset(torch.tensor(pose) + PRIOR_ERROR)
  error = estimator.estimate(hall) - torch.tensor(pose, dtype=torch.float64)
  assert error.abs().max() < TOLERANCE[name]


def test_estimators_match_the_simulator():
  pose = (121.3, 118.7, 4.2, 0.3, -0.2, 1.5)
  config = ClosedLoopConfig(device="cpu", sensor_noise_std_counts=0.0)  # exact model comparison
  sim = MagLevSim(config=config.plant(), controller=ZeroController())
  sim.reset(pose)  # coils are off at reset, so the Hall data holds only the mover field
  for name, tolerance in TOLERANCE.items():
    estimator = make_estimator(name, config)
    estimator.reset(torch.tensor(pose) + PRIOR_ERROR)
    error = estimator.estimate(sim.state.hall_measurements) - torch.tensor(pose, dtype=torch.float64)
    assert error.abs().max() < tolerance, name


@pytest.mark.parametrize("motion_model, expected_gain", [("cv", 1.5**0.5), ("ca", 2.5**0.5)])
def test_kalman_filter_has_the_requested_bandwidth(motion_model, expected_gain):
  dt, bandwidth = 250e-6, 200.0  # rad/s; bandwidth * dt << 1, where the rule holds
  config = KalmanConfig(motion_model=motion_model, bandwidth_factor=None)
  config = dataclasses.replace(config, process_psd=process_psd_for_bandwidth(torch.full((6,), bandwidth), config, dt))
  kalman = LinearKalmanFilter(config, dt)
  kalman.reset(torch.zeros(6))
  steps = round(20 * 2 * math.pi / bandwidth / dt)  # 20 periods
  outputs = [kalman.update(torch.full((6,), math.sin(bandwidth * k * dt)))[0][0] for k in range(steps)]
  gain = torch.stack(outputs[steps // 2:]).abs().max()
  # Butterworth poles at the natural frequency plus the filter zeros: |H| = sqrt(3/2) (CV) or sqrt(5/2) (CA) there.
  assert abs(gain - expected_gain) < 0.05


def test_imc_reproduces_the_whitebox_velocity_gains():
  lambdas = (4.761e-3, 4.761e-3, 4.761e-3, 0.8875e-3, 0.8875e-3, 0.1940e-3)  # back-calculated from MagLevWhiteBox kp
  gains = imc_cascade_gains(ImcTuningConfig(lambda_velocity_s=lambdas), ClosedLoopConfig().plant())
  whitebox = CascadedPidConfig().velocity_kp
  for axis in (0, 1, 3, 4, 5):  # MagLevWhiteBox tuned z separately as a PID
    assert abs(gains.velocity_kp[axis] - whitebox[axis]) < 0.01 * whitebox[axis]
