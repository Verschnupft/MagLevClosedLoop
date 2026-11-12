"""Start the closed loop: edit `CONFIG`, then run `python run.py`."""

import time

from maglevclosedloop.config import ClosedLoopConfig
from maglevclosedloop.factory import build_closed_loop

CONFIG = ClosedLoopConfig(
  grid_rows=1,
  grid_cols=1,
  initial_pose_mmdeg=(120.0, 120.0, 4.0, 0.0, 0.0, 0.0),
  reference_schedule=(
    (0.0, (120.0, 120.0, 4.0, 0.0, 0.0, 0.0)),
    (0.05, (122.0, 120.0, 4.0, 0.0, 0.0, 0.0)),
  ),
  duration_s=0.1,
  estimator="gauss_newton",  # or "oracle": simulator ground truth as pose feedback
  motion_filter="kalman",  # or "lowpass": low-pass-filtered difference, or "difference": raw backward difference
)

AXES = ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")


def main() -> None:
  loop = build_closed_loop(CONFIG)
  cycles = round(CONFIG.duration_s / loop.sim.config.dt)
  started = time.perf_counter()
  log = loop.run(cycles, CONFIG.initial_pose_mmdeg)
  print(f"{cycles} cycles in {time.perf_counter() - started:.1f} s")

  tracking = (log.measured_pose - log.reference)[-1]
  estimation = (log.estimated_pose - log.measured_pose).abs().max(0).values
  print(f"{'axis':>10} {'final pose':>12} {'tracking err':>13} {'max est err':>12}")
  for i, axis in enumerate(AXES):
    print(f"{axis:>10} {log.measured_pose[-1, i]:12.4f} {tracking[i]:13.4f} {estimation[i]:12.2e}")


if __name__ == "__main__":
  main()
