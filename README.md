# MagLevClosedLoop

Minimal closed-loop control of the [MagLevSim](https://github.com/PhilippH18/MagLevSim) magnetic-levitation planar motor.

The loop is built from exchangeable components: pose estimation, controller, and actuation (wrench to coil currents).
See [docs/architecture.md](docs/architecture.md).

## Setup

```bash
git clone --recurse-submodules https://github.com/Verschnupft/MagLevClosedLoop.git
cd MagLevClosedLoop
uv sync
```

MagLevSim is a private repository; cloning the submodule needs read access.

## Run

Edit `CONFIG` in [run.py](run.py) (grid size, initial pose, setpoints, duration, estimator), then:

```bash
.venv/Scripts/python run.py
```

## Tests

```bash
.venv/Scripts/python -m pytest
```
