"""Reduce a MagLevPoseEstimation training checkpoint to what inference needs.

Usage: python scripts/slim_checkpoint.py <run_dir>/checkpoints/best_model.pt checkpoints/<name>.pt

Keeps the model config, the input/output normalization, the weights, and the provenance.
Drops optimizer, scheduler, and training history.
"""

import sys

import torch


def main(source: str, target: str) -> None:
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  config = checkpoint["config"]
  torch.save(
    {
      "model": config["model"],
      "data": {key: config["data"][key] for key in ("input_normalization", "uz_from_tilt")},
      "normalization": {key: value.float() for key, value in checkpoint["transform_state_dict"].items()},
      "state_dict": {key: value.float() for key, value in checkpoint["model_state_dict"].items()},
      "source": {"experiment_name": config["experiment_name"], "epoch": checkpoint["epoch"]},
    },
    target,
  )


if __name__ == "__main__":
  main(*sys.argv[1:3])
