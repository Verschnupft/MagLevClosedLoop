# MagLevSim: inconsistent Euler-angle conventions

Status: local fix in `external/MagLevSim`, **not committed**. For discussion before anything goes upstream (PhilippH18/MagLevSim, commit `fa295ee`).

## Problem

MagLevSim stores the mover orientation as a rotation matrix `R` (body to world). Euler angles appear only at the interface:

| Function | Used in | Convention |
|---|---|---|
| `physics.euler_to_rotation_matrix` | `MagLevSim.reset` (initial pose) | `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)` (Z-Y-X) |
| `physics.rotation_matrix_to_euler` | `MagLevSim.actual_pose_global_mmdeg` (reported pose, controller input) | exact inverse of `R = Rx(roll) @ Ry(pitch) @ Rz(yaw)` (X-Y-Z) |

The two functions use opposite rotation orders. A pose set with `reset` is reported back with different angles.
The dynamics, the Hall model and the force model use `R` directly, so the physics is correct.
Only the reported angles are wrong.

The error is second order: it grows with the product of two angles.

| Set with `reset` (roll, pitch, yaw) in deg | Reported before the fix | Reported after the fix |
|---|---|---|
| 0.3, -0.2, 1.5 | 0.3051, -0.1921, 1.5010 | 0.3, -0.2, 1.5 |
| 5, 5, 45 | 0.0135, 7.0666, 44.7807 | 5, 5, 45 |

## Fix

`rotation_matrix_to_euler` becomes the inverse of `euler_to_rotation_matrix`. The construction stays unchanged, because Z-Y-X (yaw-pitch-roll) is the common convention and `reset` already uses it.

```diff
 def rotation_matrix_to_euler(orientation: torch.Tensor) -> torch.Tensor:
-    r02 = orientation[..., 0, 2]
-    r12 = orientation[..., 1, 2]
-    r22 = orientation[..., 2, 2]
+    # Inverse of euler_to_rotation_matrix: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
     r00 = orientation[..., 0, 0]
-    r01 = orientation[..., 0, 1]
-    pitch = torch.atan2(r02, torch.sqrt(torch.clamp(r00.square() + r01.square(), min=1e-12)))
-    roll = torch.atan2(-r12, r22)
-    yaw = torch.atan2(-r01, r00)
+    r10 = orientation[..., 1, 0]
+    r20 = orientation[..., 2, 0]
+    r21 = orientation[..., 2, 1]
+    r22 = orientation[..., 2, 2]
+    pitch = torch.atan2(-r20, torch.sqrt(torch.clamp(r21.square() + r22.square(), min=1e-12)))
+    roll = torch.atan2(r21, r22)
+    yaw = torch.atan2(r10, r00)
     return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=-1))
```

Alternative: keep `rotation_matrix_to_euler` and change `euler_to_rotation_matrix` to X-Y-Z.
That is the right choice only if the hardware (TwinCAT XPlanar) or the training data of the bundled neural controller defines the angles in X-Y-Z order. This is not verified.

## Verification

- Round trip `rotation_matrix_to_euler(euler_to_rotation_matrix(a))` over 10 000 random angles (roll and pitch within ±30°, yaw within ±179°): maximum error 3e-14° in float64, 8e-6° in float32.
- MagLevSim smoke test (`tests/test_hover_controller.py`): passes.
- MagLevClosedLoop tests: 8 passed.
- Hover with the bundled neural controller, 400 cycles (0.1 s) at `(120, 120, 4, 0, 0, 0)`, old vs fixed conversion: final pose and maximum errors are equal within 1e-4 (roll differs by 3e-4°). Near hover the angles are small, so the second-order error has almost no effect.

## Open points for discussion

1. Which convention is correct for the real system (TwinCAT XPlanar A/B/C angles)? This decides whether to fix the inverse (as here) or the construction.
2. MagLevPoseEstimation: if datasets combine Hall data from `reset(pose)` with labels from `actual_pose_global_mmdeg`, the labels are inconsistent for tilted and rotated poses (e.g. 5°, 5°, 45° gives 7° pitch). Labels taken from the commanded pose are not affected. Not checked.
3. Report upstream to the MagLevSim owner, or keep a local patch? This repository never commits inside the submodule.
