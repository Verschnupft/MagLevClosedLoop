# Architecture

For a short view with only the interfaces, see [architecture_overview.md](architecture_overview.md).

## Goal

The repository closes one control loop around the MagLevSim planar motor with the least code possible.
Every stage of the loop is exchangeable: Hall-field compensation, pose estimation, motion filtering, control, and actuation.
A stage depends only on its interface, never on another concrete stage.

## Package layout

```
src/maglevclosedloop/
  config.py       ClosedLoopConfig and all stage configs
  factory.py      default wiring: config -> ClosedLoop
  loop.py         ClosedLoop: the only place that steps MagLevSim
  interfaces.py   stage Protocols
  plant.py        nominal geometry from MagLevSim (sensors, coils, sensor window)
  reference/      piecewise_constant.py
  compensation/   passthrough.py
  estimation/     gauss_newton.py, neural.py, oracle.py
  filtering/      kalman.py, difference.py (difference, low-pass), bandwidth.py (Q from the controller)
  control/        cascaded_pid.py, imc.py (gains from the plant and lambda)
  actuation/      halbach_dq.py
```

One subpackage per stage; each exports its implementations in `__init__.py`.

## Components

```mermaid
flowchart LR
  subgraph Plant["MagLevSim (submodule)"]
    SIM[MagLevSim]
  end

  subgraph Loop["maglevclosedloop"]
    REF[Reference]
    CMP[CoilFieldCompensation]
    EST[PoseEstimator]
    MOT[MotionFilter]
    CTL[Controller]
    ACT[Actuation]
  end

  SIM -- "Hall counts (T,50,3)<br/>coil currents (T,24,3)" --> CMP
  CMP -- "mover-only Hall counts (T,50,3)" --> EST
  EST -- "estimated pose (6,) mm/deg" --> MOT
  MOT -- "pose (6,), velocity (6,)" --> CTL
  REF -- "reference pose (6,) mm/deg" --> CTL
  CTL -- "wrench (6,) N, N·m" --> ACT
  MOT -- "pose (6,)" --> ACT
  ACT -- "d/q/phi command (T,24,3)" --> SIM
```

`T` is the tile count (`grid_rows * grid_cols`).

| Stage | Interface method | Default implementation | Origin |
|---|---|---|---|
| Reference | `pose_mmdeg(time_s)` | `PiecewiseConstantReference` | new |
| Compensation | `compensate(hall, currents)` | `NoCoilFieldCompensation` (pass-through) | new |
| Pose estimation | `estimate(hall, prior, initial)` | `GaussNewtonPoseEstimator` or `NeuralPoseEstimator` | MagLevPoseEstimation (finite cuboid model with Hall element offsets; trained networks) |
| Motion filter | `update(pose)` → pose, velocity | `LinearKalmanFilter` (constant acceleration), `LowPassDifferenceFilter` or `FiniteDifferenceFilter` | MagLevPoseEstimation (`kf_ca_const_*`); low-pass and difference new |
| Control | `compute(pose, velocity, reference)` | `CascadedPidController`, gains by IMC (`control/imc.py`) | MagLevWhiteBox (`MimoController`, `ImcPidTuning`) |
| Actuation | `compute(wrench, pose)` | `HalbachDqActuation` | MagLevWhiteBox (`ForceTransform`) |

With `coil_field_in_hall=False` (default) the plant uses `SimConfig.scaling_const_stator = 0`. The Hall data then hold only the mover field; the coil forces stay unchanged. The compensation stage is a pass-through until the coil field is switched on.

Current limits: MagLevSim limits each coil to |(d, q)| <= 19000 increments (d and q scaled together) and derates a tile above its power limit. The actuation applies the coil limit itself with lift priority (d first, q gets the rest) and returns the wrench the limited currents achieve. The controller uses it for anti-windup by back-calculation (k_aw = ki / kp). The tile power derating is not mirrored; it acts only far above hover power.

`OracleEstimator` returns the simulator ground truth (`estimator="oracle"`). Use it to test the other stages without estimation error.

The motion filter supplies the controller with pose and velocity (`motion_filter` switch).
`FiniteDifferenceFilter` passes the pose through and differentiates it; this amplifies estimation noise.
`LowPassDifferenceFilter` smooths that difference with a first-order low-pass (`cutoff_hz`); the pose stays unfiltered.
`LinearKalmanFilter` estimates pose and velocity per axis with a constant-acceleration (or constant-velocity) model and smooths the noise.

### Kalman filter parameters

The filter is the same as `kf_ca_const_*` in MagLevPoseEstimation (`filtering/ekf.py`): constant acceleration, continuous white-noise process covariance Q, constant measurement covariance R, Joseph-form update. All values are SI (m, rad).

R describes the estimator, so each estimator config holds its own `KalmanConfig`:

| Estimator | Config | Tuned entry in `filter_tuning_seed20261029.json` |
|---|---|---|
| Gauss-Newton (and oracle) | `GaussNewtonConfig.kalman` (= `KalmanConfig` defaults) | `kf_ca_const_cpp_parallel_taylor_1_element_offsets` |
| Neural (log MLP) | `NeuralConfig.kalman` | `kf_ca_const_singleframe_log_mlp_700k` |

The values were fitted in MagLevPoseEstimation on `data/filter_tuning_seed20261029/motion` with 4 counts Hall noise (notebooks `experiments/05_hold_*` and `06_motion_*`): R from the estimator residuals, Q per axis group by minimal rollout RMSE.
The tuned Q minimizes the open-loop estimation error. In the closed loop its phase lag is too large: the filter bandwidth is about 6–11 Hz, the velocity loops reach about 33 Hz (x, y), 78 Hz (z), 180 Hz (roll, pitch), and 820 Hz (yaw). With the oracle estimator and the tuned Q the mover falls, even in pure hover.

Therefore Q follows the controller by default (`filtering/bandwidth.py`, `KalmanConfig.bandwidth_factor`):

1. Velocity-loop bandwidth per axis: omega_c = kp / m (translation) or kp / I (rotation), from `CascadedPidConfig.velocity_kp` and the MagLevSim mass and inertia.
2. Filter bandwidth omega_f = `bandwidth_factor` x omega_c, at most a quarter of the Nyquist frequency.
3. Q = r · dt · omega_f^(2n), n = 2 (CV) or 3 (CA). The steady-state filter then has Butterworth poles at omega_f (checked in the tests).

R stays the tuned estimator value. `bandwidth_factor=None` uses the tuned `process_psd` instead.

Coupling as `FilteredPoseEstimator` in MagLevPoseEstimation, per cycle: the filter predicts; the estimator works on the sensor window around the last filtered pose and starts from the prediction (Gauss-Newton start value, network yaw branch); the filter moves the measurement to the yaw branch of the prediction, wraps the roll and pitch innovations, and updates. The first measurement sets the pose with covariance R.

| Method in `notebooks/experiments` (hold, motion) | `ClosedLoopConfig` |
|---|---|
| `cpp_parallel_taylor_1_element_offsets` | `estimator="gauss_newton"`, `motion_filter="lowpass"` or `"difference"` (only for the velocity) |
| `kf_ca_const_cpp_parallel_taylor_1_element_offsets` | `estimator="gauss_newton"`, `motion_filter="kalman"` |
| `singleframe_log_mlp_700k` | `estimator="neural"`, `motion_filter="lowpass"` or `"difference"` |
| `kf_ca_const_singleframe_log_mlp_700k` | `estimator="neural"`, `motion_filter="kalman"` |

Without the Kalman filter the prediction is the last pose, as in the notebook rollouts.

Remaining difference to the notebooks: Q follows the controller (bandwidth rule) instead of the open-loop tuning. The Gauss-Newton step runs in PyTorch, not in the C++ code of `cpp_parallel_taylor_1`.

### Controller tuning

The cascade uses the IMC method of MagLevWhiteBox (`ImcPidTuning`, PT1 case), with the plant taken from the MagLevSim parameters instead of a system identification (`control/imc.py`):

| Step | Rule |
|---|---|
| Velocity plant per axis | m dv/dt = F − c v, so G(s) = K / (T s + 1) with K = 1/c, T = m/c (rotation: I, c_w) |
| Velocity loop (PI) | kp = m / lambda, ki = c / lambda: closed loop 1 / (lambda s + 1) |
| Position loop (P) | kp = 1 / lambda_position, lambda_position = 5 lambda (MagLevWhiteBox ratio) |
| Feed-forward | m g on Fz |

With the MagLevWhiteBox lambdas (4.8 ms xy, 0.9 ms tilt, 0.19 ms yaw) the formulas give its gains (test). Those lambdas suit the noise-free plant; with Hall noise the yaw loop (lambda below the 0.25 ms sample time) and the tilt loop amplify the velocity noise until the coils saturate.

`scripts/tune_controller.py` therefore chooses lambda per axis group (xy, z, tilt, yaw) by closed-loop simulation: neural estimator, Kalman filter (Q from the bandwidth rule), 4 counts Hall noise, default scenario (hover, then 2 mm x step). A candidate is feasible when no requested coil command exceeds the MagLevSim limit (19000 increments); the smallest RMS tracking error among feasible candidates wins. Result: lambda = 5 ms (xy, tilt, yaw) and 2 ms (z). For xy the coil limit is the binding constraint.

`controller_tuning=None` uses the explicit `CascadedPidConfig` gains (MagLevWhiteBox values) instead.

## Neural pose estimation

`NeuralPoseEstimator` runs a network trained in MagLevPoseEstimation. It has three exchangeable parts:

```mermaid
flowchart LR
  H["Hall counts (T,50,3)"] --> W["sensor window<br/>50 sensors under the mover<br/>(plant.py)"]
  W --> N["input standardization"] --> NET["network<br/>(any nn.Module)"] --> D["output de-standardization"] --> DEC["decode<br/>(any callable)"]
  DEC --> P["pose (6,) + window offset"]
  PRIOR["prior: last loop pose"] --> W
  INIT["initial: filter prediction"] --> DEC
```

| Part | Source | Default |
|---|---|---|
| Network architecture | checkpoint `model` (kind, hidden dims, activation, head) | `PoseMLP`; other kinds raise `NotImplementedError` until ported |
| Weights | checkpoint `state_dict` | loaded strictly |
| Input/output normalization | checkpoint `normalization` | per-component Hall statistics, per-output statistics |
| Output decoding | checkpoint `model.output_dim` and `data.uz_from_tilt` | 8 outputs: body z-axis and 4-fold twist, yaw branch from the previous pose |
| Sensor window | fixed: MagLevSim sensor lattice | window centre nearest the previous pose; works on any M x N grid |
| Hall units and axis order | fixed: training convention | counts, `[By, Bx, Bz]`, no coil field |

`NeuralPoseEstimator.from_checkpoint` reads everything from the checkpoint. To customize, construct `NeuralPoseEstimator(plant, network, normalization, decode)` directly with another network or decoder.

Checkpoints live in `checkpoints/`. `scripts/slim_checkpoint.py` reduces a training checkpoint from MagLevPoseEstimation to the inference part and keeps the run name for provenance.
`checkpoints/logmlp_seed42.pt` is run `final_logmlp_mlp_2048-2048-1024_lr1e-4_cosine1000_seed42_1m` (best model, epoch 1000).

A neural estimator inside a Kalman filter is the combination `estimator="neural"` and `motion_filter="kalman"`; the filter then uses `NeuralConfig.kalman`.

## Interfaces

The interfaces are `typing.Protocol` classes in `interfaces.py`.
A new stage needs no base class; it needs only the matching method.

```mermaid
classDiagram
  class Reference {
    <<Protocol>>
    pose_mmdeg(time_s) Tensor
  }
  class CoilFieldCompensation {
    <<Protocol>>
    compensate(hall_counts, coil_currents_abc) Tensor
  }
  class PoseEstimator {
    <<Protocol>>
    estimate(hall_counts, prior_mmdeg, initial_mmdeg) Tensor
  }
  class MotionFilter {
    <<Protocol>>
    reset(initial_pose_mmdeg)
    predict() Tensor
    update(pose_mmdeg) pose, velocity
  }
  class Controller {
    <<Protocol>>
    reset()
    compute(pose_mmdeg, velocity, reference_mmdeg) Tensor
  }
  class Actuation {
    <<Protocol>>
    compute(wrench, pose_mmdeg) Tensor
  }
  class ClosedLoop {
    sim: MagLevSim
    run(cycles, initial_pose_mmdeg) LoopLog
  }

  Reference <|.. PiecewiseConstantReference
  CoilFieldCompensation <|.. NoCoilFieldCompensation
  PoseEstimator <|.. GaussNewtonPoseEstimator
  PoseEstimator <|.. OracleEstimator
  PoseEstimator <|.. NeuralPoseEstimator
  MotionFilter <|.. LinearKalmanFilter
  MotionFilter <|.. LowPassDifferenceFilter
  MotionFilter <|.. FiniteDifferenceFilter
  Controller <|.. CascadedPidController
  Actuation <|.. HalbachDqActuation

  ClosedLoop o-- Reference
  ClosedLoop o-- CoilFieldCompensation
  ClosedLoop o-- PoseEstimator
  ClosedLoop o-- MotionFilter
  ClosedLoop o-- Controller
  ClosedLoop o-- Actuation
```

## One control cycle (250 µs)

```mermaid
sequenceDiagram
  participant L as ClosedLoop
  participant S as MagLevSim
  participant C as Compensation
  participant E as PoseEstimator
  participant M as MotionFilter
  participant K as Controller
  participant A as Actuation

  L->>S: step(reference, command[k-1])
  Note over S: samples Hall data at the current pose,<br/>then integrates the rigid body
  S-->>L: Hall counts, coil currents
  L->>C: compensate(hall, currents)
  C-->>L: mover-only Hall counts
  L->>M: predict()
  M-->>L: prediction
  L->>E: estimate(hall, last pose, prediction)
  E-->>L: pose estimate
  L->>M: update(pose estimate)
  M-->>L: pose, velocity
  L->>K: compute(pose, velocity, reference)
  K-->>L: wrench
  L->>A: compute(wrench, pose)
  A-->>L: command[k]
```

The command of cycle `k` takes effect in cycle `k + 1`. This is one sample of transport delay, as on real hardware.

## Configuration

```mermaid
flowchart TD
  CFG["ClosedLoopConfig<br/>(run.py)"]
  CFG -- "plant()" --> PLANT["maglevsim.SimConfig<br/>grid, noise, device, initial pose"]
  CFG --> GN[GaussNewtonConfig]
  CFG --> NC[NeuralConfig]
  GN --> KF1[KalmanConfig]
  NC --> KF2[KalmanConfig]
  CFG --> LP[LowPassConfig]
  CFG --> IMC[ImcTuningConfig] --> PID[CascadedPidConfig]
  CFG --> HAL[HalbachActuationConfig]
  FAC["factory.build_closed_loop()"]
  CFG --> FAC
  PLANT --> FAC
  FAC --> LOOP[ClosedLoop]
```

`ClosedLoopConfig` holds every value a user changes: grid size, initial pose, reference schedule, duration, estimator, and motion filter.
`build_closed_loop` distributes these values to the stage configs.
The initial pose is set once and used for both the simulator reset and the estimator start value.

## Frames and units

| Quantity | Frame and unit |
|---|---|
| Pose at interfaces | global array frame, origin at the lower-left array corner: `[x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]` |
| Estimator internals | MagLevSim world frame, origin at the array centre: metres, radians |
| Hall data | counts (20000 per tesla), axis order `[By, Bx, Bz]` |
| Wrench | `[Fx, Fy, Fz]` in N, `[Tx, Ty, Tz]` in N·m |
| Command | per coil `[d_incr, q_incr, electrical_angle_rad]` |

Units change only inside a stage. `plant.py` is the only module that uses MagLevSim internals (geometry helpers).

## Grid size (M x N)

No stage is limited to one tile:

- The Gauss-Newton estimator uses the 50-sensor window under the mover, as the network does.
- The neural estimator uses the 50-sensor window under the mover, which can span tiles.
- The actuation checks the coils of all tiles against the four Halbach arrays.
- The motion filter and the controller work on the global pose only.

The simulator start-up time grows with the tile count because MagLevSim precomputes the coil-to-sensor sensitivity matrix.

## Exchange a stage

1. Write a class with the interface method (see `interfaces.py`).
2. Construct `ClosedLoop` with it, or change the wiring in `factory.build_closed_loop`.
3. Add its parameters as a frozen dataclass in `config.py` when it has any.
