# ENVIRONMENT_SPEC.md — Menagerie Panda Reach

> Implements: `References/SelfEx-WM_Notes.tex` **§8 Phase 2: First Menagerie Task — Panda Reach**
> (§8.1 Task, §8.2 Observation, §8.3 Action, §8.4 Control interval).
> Frozen at execution-order step 11 (§18); after that, any change invalidates prior runs.

**Status: §8 values frozen 2026-08-25.** Derived from
`third_party/mujoco_menagerie/franka_emika_panda/scene.xml` at the commit pinned in `UPSTREAM.md`.

Every constant here has exactly one source of truth: the `wrapper:` block of
`menagerie_integration/menagerie_panda_reach.yaml`. This document explains and freezes them; the YAML
supplies them to the code. They must never disagree.

---

## 1. Robot and scene

| Item | Value |
|---|---|
| Robot | Franka Emika Panda |
| Menagerie path | `third_party/mujoco_menagerie/franka_emika_panda/` |
| Base model | `scene.xml` (includes `panda.xml` + hand, floor, light) |
| Task MJCF | `menagerie_tasks/panda_reach.xml` |
| Scene contents | Menagerie `scene.xml` `<include>`, `target_site`, fixed camera |
| `nq` / `nv` | 9 / 9 — 7 arm joints + 2 finger joints |
| End-effector reference | computed: `data.xpos[hand] + data.xmat[hand] @ (0, 0, 0.1034)` |
| Actuator set | all 8 — `actuator1..7` (arm) + `actuator8` (gripper) |
| `nu` (action dim) | 8 |

Menagerie's Panda defines **no sites at all** (`nsite == 0`), so `p_ee` does not exist in the stock
model — and MJCF cannot reopen a body that arrived via `<include>`, so a site cannot be attached to
the `hand` body from our task file either. The environment therefore computes `p_ee` from the hand
body's pose, offsetting by the Franka TCP (0.1034 m along the hand frame's +z, between the fingers).
Exact, no model surgery, Menagerie untouched.

`actuator1..7` are **position** actuators whose `ctrlrange` equals the joint limit in radians, so a
control is an absolute joint target. `actuator8` is the gripper on a `[0, 255]` scale — a different
unit entirely, which §8.3's affine map handles without special-casing.

The gripper is included because §8.3 says to use the robot's native actuator interface with
`shape=(num_actuators,)`. For free-space reaching its motion is pure exploration noise, but excluding
it would give Panda Push (§15) a different action space and break comparability between the two tasks.

The target is a **visual `site` only**: massless, contact-free, and not part of the physics. It exists
so the goal is visible in `rgb_array` frames. Nothing about the task dynamics depends on it.

## 2. Task (§8.1)

At reset, sample a Cartesian target `g = (g_x, g_y, g_z)` uniformly from a fixed reachable box.
With `p_ee(s_t)` the end-effector position:

    d_t = || p_ee(s_t) - g ||_2

Bounded dense task reward, with `alpha` **frozen before training**:

    r_task_t = exp(-alpha * d_t^2)

Binary success at a fixed tolerance:

    success_t = 1[ d_t < epsilon ]

| Constant | Value | Notes |
|---|---|---|
| `target_box_low` | `(0.30, -0.30, 0.20)` | metres, world frame |
| `target_box_high` | `(0.60,  0.30, 0.60)` | metres, world frame |
| `alpha` | `10.0` | **frozen before training**, per §8.1 |
| `success_tol` (`epsilon`) | `0.05` | 5 cm, per §8.1 |
| Initial arm configuration | `home` keyframe | `qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]` |
| Initial-state jitter | `U(-0.05, +0.05)` rad on the 7 arm joints | fingers held at `0.04`; `qvel` zeroed |

The box sits in front of the base, well inside the Panda's ~0.85 m reach and above the floor, so every
corner is comfortably reachable. `tests/test_target_sampling.py` asserts samples stay inside it;
`scripts/stress_rollout.py` reports the achieved target-distance distribution as a reachability check.

`alpha = 10` gives `r(5 cm) = 0.98`, `r(30 cm) = 0.41`, `r(60 cm) = 0.03` — an informative gradient
across the entire box, so the task actor gets signal from anywhere in the workspace rather than only
near the goal.

Joint jitter exists so different seeds produce different initial states, not just different targets
(§10 "Seed separation"). It is small enough not to disturb the `home` pose's kinematic character.

The target box must be validated as genuinely reachable — see `tests/test_target_sampling.py` and the
`scripts/stress_rollout.py` target-distance distribution.

**The task reward exists only so the task actor can be evaluated.** The exploration actor optimizes the
Plan2Explore intrinsic objective, never this reward.

## 3. Observation (§8.2)

    o_t = [ q_t, q̇_t, p_ee,t, g ]

Exposed as a flat single-key dict of NumPy arrays, which is what SheepRL's MLP encoder consumes:

```python
{
    "state": np.concatenate([
        qpos,
        qvel,
        ee_position,
        target_position,
    ]).astype(np.float32)
}
```

| Item | Value |
|---|---|
| `observation_space` | `gymnasium.spaces.Dict({"state": Box(-inf, inf, (24,), float32)})` |
| `obs_dim` | `24` = `nq(9) + nv(9) + p_ee(3) + g(3)` |
| SheepRL encoder keys | `algo.mlp_keys.encoder=[state]`, `algo.cnn_keys.encoder=[]` |
| Observation mode | `vector` (manifest field `observation_mode`) |
| Normalization | none in the env — SheepRL handles its own scaling |

Single flat key, deliberately: it keeps the observation contract identical between this task and Panda
Push (§15), where only `obs_dim` changes.

## 4. Action (§8.3)

Native actuator interface, normalized:

    a_t ∈ [-1, 1]^m
    u_t = u_min + (a_t + 1)/2 * (u_max - u_min)

| Item | Value |
|---|---|
| `action_space` | `gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)` |
| `u_min`, `u_max` | read from `model.actuator_ctrlrange`, never hard-coded |
| Unbounded actuators | none — all 8 have `ctrllimited == True`; the env asserts this at construction |
| Clipping | actions clipped to `[-1, 1]` before denormalization |
| Action mode | `normalized_native_actuator` (manifest field `action_mode`) |

Plan2Explore is never told about robot-specific action units. The environment does the normalization.

## 5. Control interval (§8.4)

    N_substeps = Δt_control / Δt_sim         (must be an integer)

| Item | Value |
|---|---|
| `Δt_control` | `0.05` s |
| `Δt_sim` (`model.opt.timestep`) | `0.002` s — from the compiled model |
| `n_substeps` | `25` — exact, asserted integer at construction |
| `render_fps` | `20` (= `1 / Δt_control`) |
| `action_repeat` (SheepRL) | `1` — the env already integrates the control interval |

Both `Δt_control` and `Δt_sim` are recorded in every run manifest (§17).

## 6. Episode structure

| Item | Value |
|---|---|
| `max_episode_steps` | `100` control steps |
| Episode duration | `5.0` s = `100 * 0.05` |
| `terminated` | always `False` — reaching is not an absorbing task; success does not end the episode |
| `truncated` | `True` when `step_count >= max_episode_steps` |

`terminated` is held at `False` on purpose: an early-terminating episode would leak task information
into the exploration objective through the DreamerV3 continue-predictor.

## 7. `info` dict

Emitted every step, for the §13 metrics:

| Key | Type | Purpose |
|---|---|---|
| `distance` | `float` | `d_t`, final-distance metric (§13.1) |
| `success` | `bool` | `1[d_t < epsilon]`, success rate (§13.1) |
| `ee_position` | `(3,) float32` | workspace-coverage voxelization (§13.2) |
| `target_position` | `(3,) float32` | target-distance distribution |
| `ctrl_saturation` | `float` | fraction of actuators at a control limit (§13.2) |

## 8. Determinism and seeding

| Item | Value |
|---|---|
| Seeding | `super().reset(seed=seed)`, then all sampling via `self.np_random` |
| Reset determinism | same seed ⇒ identical initial state *and* target (`tests/test_reset_determinism.py`) |
| Seed separation | different seeds ⇒ different targets/initial states (`tests/test_seed_separation.py`) |
| MuJoCo state reset | `mujoco.mj_resetData` before applying any keyframe or jitter |

No global RNG (`np.random.*`, `random.*`) is used anywhere in the environment.

## 9. Rendering

| Item | Value |
|---|---|
| `render_modes` | `["rgb_array", "human"]` |
| `render_fps` | `20` |
| Frame size | `(480, 640)`, returned as `uint8` `(H, W, 3)` |
| Camera | `track_cam`, a fixed camera declared in `panda_reach.xml` |

Rendering is off during training (`capture_video: False`); it exists for evaluation videos and for
`tests/test_render.py`.

## 10. Out of scope for Panda Reach

Contact, objects, and the table are deliberately absent. Free-space reaching isolates environment
correctness, action scaling, state estimation, Plan2Explore learning, exploration coverage, and task
actor learning before contact-rich manipulation is introduced in Panda Push (§15).
