"""Coil-field compensation: remove the stator field from the Hall data (`interfaces.CoilFieldCompensation`)."""

from .passthrough import NoCoilFieldCompensation

__all__ = ["NoCoilFieldCompensation"]
