import pytest
import torch
from maglevsim import MagLevSim, ZeroController

from maglevclosedloop.actuation import HalbachDqActuation
from maglevclosedloop.config import ClosedLoopConfig, KalmanConfig, LowPassConfig
from maglevclosedloop.estimation import GaussNewtonPoseEstimator
from maglevclosedloop.motion import FiniteDifferenceFilter, LinearKalmanFilter, LowPassDifferenceFilter

HOVER = (120.0, 120.0, 4.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("make_filter", [
  lambda dt: LinearKalmanFilter(KalmanConfig(), dt),
  lambda dt: LowPassDifferenceFilter(LowPassConfig(), dt),
  lambda dt: FiniteDifferenceFilter(dt),
])
def test_motion_filter_tracks_a_constant_velocity_ramp(make_filter):
  dt = 250e-6
  velocity = torch.tensor([10.0, -5.0, 1.0, 0.5, -0.5, 20.0], dtype=torch.float64)
  start = torch.tensor(HOVER, dtype=torch.float64)
  motion_filter = make_filter(dt)
  motion_filter.reset(start)
  for k in range(1, 200):
    pose, estimated_velocity = motion_filter.update(start + velocity * k * dt)
  assert torch.allclose(estimated_velocity, velocity, rtol=1e-3)
  assert torch.allclose(pose, start + velocity * 199 * dt, atol=1e-4)


@pytest.mark.parametrize("make_filter", [
  lambda dt, noise_std: LinearKalmanFilter(KalmanConfig(measurement_std=(noise_std,) * 6), dt),
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


def test_gauss_newton_inverts_its_forward_model_on_a_grid():
  pose = (300.0, 250.0, 3.5, -0.4, 0.2, 5.0)
  config = ClosedLoopConfig(grid_rows=2, grid_cols=3)
  estimator = GaussNewtonPoseEstimator(config.plant(), config.gauss_newton)
  estimator.reset(torch.tensor(pose))
  hall = estimator.predict_counts(estimator._pose).reshape(6, 50, 3)
  estimator.reset(torch.tensor(pose) + torch.tensor([1.0, -1.0, 0.3, 0.2, 0.2, -1.0]))
  assert torch.allclose(estimator.estimate(hall), torch.tensor(pose, dtype=torch.float64), atol=1e-6)


def test_gauss_newton_matches_the_simulator():
  pose = (121.3, 118.7, 4.2, 0.3, -0.2, 1.5)
  config = ClosedLoopConfig(device="cpu")
  sim = MagLevSim(config=config.plant(), controller=ZeroController())
  sim.reset(pose)  # coils are off at reset, so the Hall data holds only the mover field
  estimator = GaussNewtonPoseEstimator(config.plant(), config.gauss_newton)
  estimator.reset(torch.tensor(pose) + torch.tensor([1.0, -1.0, 0.3, 0.2, 0.2, -1.0]))
  error = estimator.estimate(sim.state.hall_measurements) - torch.tensor(pose, dtype=torch.float64)
  assert error.abs().max() < 1e-3
