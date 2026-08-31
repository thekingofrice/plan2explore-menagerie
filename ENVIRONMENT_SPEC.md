# ENVIRONMENT_SPEC.md - Menagerie x SheepRL/Plan2Explore DreamerV3

> Implements: `References/SelfEx-WM_Notes.tex` **§8 Phase 2: First Menagerie Task - Panda Reach**
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
| Model loaded | `third_party/mujoco_menagerie/franka_emika_panda/scene.xml`, **unmodified** |
| Task MJCF | none yet - see "Why no task MJCF" below |
| `nq` / `nv` | 9 / 9 - 7 arm joints + 2 finger joints |
| End-effector reference | computed: `data.xpos[hand] + data.xmat[hand] @ (0, 0, 0.1034)` |
| Actuator set | all 8 - `actuator1..7` (arm) + `actuator8` (gripper) |
| `nu` (action dim) | 8 |

Menagerie's Panda defines **no sites at all** (`nsite == 0`), so `p_ee` does not exist in the stock
model. The environment computes it from the hand body's pose, offsetting by the Franka TCP (0.1034 m
along the hand frame's +z, between the fingers). Exact, no model surgery, Menagerie untouched.

### Why no task MJCF

§5's layout lists `menagerie_tasks/panda_reach.xml`, and one was written, but nothing in §8 needs it:

- `p_ee` is computed from the `hand` body pose - no site required.
- `g` is a number the environment samples, writes into the observation, and measures `d_t` against.
  It has no mass, no geometry, and no effect on the physics, so it need not exist in the model.

A target `site` and a fixed camera are therefore **purely cosmetic** - they only make `g` visible in
rendered frames. Deferred until §10's render test needs a viewpoint, at which point `mujoco.MjSpec`
can attach them to the loaded model from Python.

The task file was dropped rather than fixed because making it work required copying Menagerie's
`scene.xml` contents into this repository. MuJoCo compounds the base directory across nested
cross-directory `<include>`s: `scene.xml` contains a bare `<include file="panda.xml"/>`, and
including `scene.xml` from `menagerie_tasks/` made MuJoCo resolve that to
`franka_emika_panda/third_party/mujoco_menagerie/franka_emika_panda/panda.xml` - the path twice -
which does not exist. Inlining the upstream scene to avoid the nesting would have created a silent
fork of a checkout that §6 pins by SHA precisely so it has one source of truth. Loading `scene.xml`
directly as the top-level model makes its own relative includes resolve correctly and copies nothing.

`actuator1..7` are **position** actuators whose `ctrlrange` equals the joint limit in radians, so a
control is an absolute joint target. `actuator8` is the gripper on a `[0, 255]` scale - a different
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
| Initial arm configuration | `home` keyframe | restores `qpos`, `qvel` **and** `ctrl` together |
| Initial-state jitter | `U(-0.05, +0.05)` rad on the 7 arm joints | clipped to `jnt_range`; fingers held at `0.04`; `qvel` zeroed |

The keyframe is restored with `mj_resetDataKeyframe`, which sets `ctrl` as well as `qpos`. That
matters with position actuators: resetting `qpos` while leaving `ctrl` at zero would command every
joint toward position 0 and produce a lurch on the first step of every episode.

The box sits in front of the base, well inside the Panda's ~0.85 m reach and above the floor, so every
corner is comfortably reachable. `tests/test_target_sampling.py` asserts samples stay inside it.

### Measured start-state distribution (2000 seeds, 2026-08-25)

| Quantity | Value |
|---|---|
| `p_ee` at `home` | `(0.5458, 0.0006, 0.5254)` |
| `d0` min / median / max | `0.017` / `0.272` / `0.498` m |
| Episodes starting inside the 5 cm tolerance | **1.05 %** |
| Episodes starting within 10 cm | 5.15 % |
| Median start reward | `0.478` |

`p_ee(home)` lies **inside** the target box on all three axes, so `g` can be drawn essentially at the
start position. This puts a **~1 % floor under the success rate**: a policy that does nothing at all
scores about 1 %. Accepted deliberately rather than patched, so that sampling stays exactly
uniform-in-box as specified. §13's success rate must be read against a 1 % no-op baseline, not 0 %.
§14's random-action baseline measures the same floor directly.

`alpha = 10` is **frozen** and confirmed against this measured distribution: `r(0.017) = 0.997`,
`r(0.27) = 0.478`, `r(0.50) = 0.082`. The reward spans its full range across the distances the task
actually produces, rather than saturating near 1 or collapsing to 0 - so the task actor gets an
informative gradient from anywhere in the box.

Joint jitter exists so different seeds produce different initial states, not just different targets
(§10 "Seed separation"). It is small enough not to disturb the `home` pose's kinematic character.

The target box must be validated as genuinely reachable - see `tests/test_target_sampling.py` and the
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
| Normalization | none in the env - SheepRL handles its own scaling |

Single flat key, deliberately: it keeps the observation contract identical between this task and Panda
Push (§15), where only `obs_dim` changes.

`q_t` is the full `nq = 9` vector - 7 arm joints plus **both** finger joints - not the 7 arm joints
alone. §8.2 says only "at minimum", and neither Menagerie nor Plan2Explore fixes the convention
(Plan2Explore is DMC-only and pixel-based; SheepRL's DMC wrapper flattens whatever a task declares).
The deciding argument is consistency with §8.3: the gripper **is** actuated. A joint the policy can
move but the world model cannot see makes its effect look like irreducible noise - exactly the
aleatoric uncertainty Plan2Explore's ensemble must not mistake for epistemic uncertainty, and an
inexhaustible source of intrinsic reward for an exploration actor that learns to wiggle the gripper.
If it is actuated, it is observed.

Menagerie couples the fingers with an equality constraint (`finger_joint2` mirrors `finger_joint1`),
so one observation dimension is exactly redundant. Harmless - a constant linear dependence the
encoder learns to ignore - and it keeps `q_t` literally equal to `data.qpos`, with no index masking.

## 4. Action (§8.3)

Native actuator interface, normalized:

    a_t ∈ [-1, 1]^m
    u_t = u_min + (a_t + 1)/2 * (u_max - u_min)

| Item | Value |
|---|---|
| `action_space` | `gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)` |
| `u_min`, `u_max` | read from `model.actuator_ctrlrange`, never hard-coded |
| Unbounded actuators | none - all 8 have `ctrllimited == True`; the env asserts this at construction |
| Clipping | actions clipped to `[-1, 1]` before denormalization |
| Action mode | `normalized_native_actuator` (manifest field `action_mode`) |

Plan2Explore is never told about robot-specific action units. The environment does the normalization.

## 5. Control interval (§8.4)

    N_substeps = Δt_control / Δt_sim         (must be an integer)

| Item | Value |
|---|---|
| `Δt_control` | `0.05` s |
| `Δt_sim` (`model.opt.timestep`) | `0.002` s - from the compiled model |
| `n_substeps` | `25` - exact, asserted integer at construction |
| `render_fps` | `20` (= `1 / Δt_control`) |
| `action_repeat` (SheepRL) | `1` - the env already integrates the control interval |

Both `Δt_control` and `Δt_sim` are recorded in every run manifest (§17).

## 6. Episode structure

| Item | Value |
|---|---|
| `max_episode_steps` | `100` control steps |
| Episode duration | `5.0` s = `100 * 0.05` |
| `terminated` | always `False` - reaching is not an absorbing task; success does not end the episode |
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
| Camera | MuJoCo's free camera for now; a fixed camera is added at §10 (see "Why no task MJCF") |

Rendering is off during training (`capture_video: False`); it exists for evaluation videos and for
`tests/test_render.py`.

## 10. Known issues

**Intermittent SIGFPE in `mj_step` (2026-08-25, unresolved).** One `pytest tests/` run died with
`Floating point exception (core dumped)` inside `mujoco.mj_step`, called from `env.step` during the
§10 "Action bounds" row. A native SIGFPE kills the process outright, so there is no Python traceback.
Running that test standalone was clean, and a subsequent full-suite run passed with all ten rows
green. **It disappeared rather than being fixed.**

Not reproduced often enough to diagnose. Candidates, in order of suspicion: floating-point traps
enabled somewhere in the import path, turning a normally harmless underflow inside MuJoCo into a
fatal signal; the leaked `MjModel`/`MjData` pair from the "API test" row, which is the only env in
the suite never closed; or genuine solver divergence under sustained maximum control authority.

Relevant because §12's budgets step this environment 10^5-10^6 times. A crash rare enough to hide in
a test suite is near-certain over a Gate C run. If a training run dies with no Python traceback, this
is the first thing to check - not a Plan2Explore or SheepRL bug.

## 11. Out of scope for Panda Reach

Contact, objects, and the table are deliberately absent. Free-space reaching isolates environment
correctness, action scaling, state estimation, Plan2Explore learning, exploration coverage, and task
actor learning before contact-rich manipulation is introduced in Panda Push (§15).

---

# Panda Push (§15)

Everything below describes `menagerie_integration/menagerie_panda_push.py`. Where a value is not
restated here it is identical to Panda Reach's above.

## 12. Scene and model

| Item | Value |
|---|---|
| Task MJCF | `menagerie_tasks/panda_push.xml` |
| Model loaded | that file, **via the symlink beside Menagerie's `scene.xml`** |
| `nq` / `nv` / `njnt` | `16` / `15` / `10` |
| `nu` (action dim) | `8` - **unchanged from Reach** |
| Table geom | static box, half-extents `(0.30, 0.40, 0.11)` at `(0.5, 0, 0.11)`; top surface `z = 0.22` |
| Cube geom | box, half-extent `0.025` (5 cm cube), mass `0.05` kg, `friction="1 0.005 0.0001"` |
| Cube joint | `cube_free`, a free joint - `qposadr 9`, `dofadr 9` |

The MJCF must be loaded from inside the Menagerie directory. `panda_push.xml` contains
`<include file="scene.xml"/>`, and MuJoCo resolves that relative to the file being loaded; only as a
sibling of `scene.xml` does the include work, for the reason §1's "Why no task MJCF" records.
`scripts/install_env_wrapper.sh` places the link and excludes it from that clone's `git status`. No
tracked Menagerie file changes, so the §6 SHA pin still verifies.

Adding the cube leaves `nu` at 8, so **Push and Reach share an action space** and stay comparable -
which is why §1 keeps the gripper actuator in Reach despite it being pure noise there.

`nq != njnt` for the first time: a free joint carries 7 `qpos` entries and 6 dofs but counts as one
joint. Any code slicing `qpos` by joint index is wrong here, which is why §13.2's joint metrics log
only the 9 actuated joints and `metrics.py` splits the diagnostics row on `nu`.

The wrapper reads the cube's half-extent and the table top's height **off the compiled model** rather
than repeating them in Python, so the MJCF is the single source of truth for both.

## 13. Task (§15)

At reset, sample a planar target `g = (g_x, g_y)` uniformly from a fixed region of the table top,
at the cube's resting height. With `p_cube(s_t)` the cube's centre:

    d_t = || p_cube(s_t) - g ||_2
    r_task_t = exp(-beta * d_t^2)
    success_t = 1[ d_t < epsilon ]

| Constant | Value | Notes |
|---|---|---|
| `target_box_low` | `(0.35, -0.20)` | metres, world frame; planar |
| `target_box_high` | `(0.65,  0.20)` | |
| target `z` | `0.245` | derived: table top `0.22` + cube half-extent `0.025` |
| `cube_init_xy` | `(0.45, 0.0)` | |
| `cube_jitter` | `U(-0.03, +0.03)` m on x and y | |
| `beta` | `30.0` | **frozen 2026-08-31**, per the measurement below |
| `success_tol` (`epsilon`) | `0.05` | 5 cm, as Reach |

The distance is cube-to-goal, not end-effector-to-goal. That single change is the substance of §15:
the arm must make contact and transport the cube, where Reach only had to arrive.

### Measured start-state distribution (2000 seeds, 2026-08-31)

Cube start and goal are both drawn in the plane at the same `z`, so `d_0` is a planar distance and
the distribution follows from the sampling alone - no physics involved.

| Quantity | Value |
|---|---|
| `d_0` p0 / p25 / median / p75 / p100 | `0.006` / `0.100` / `0.146` / `0.188` / `0.304` m |
| `d_0` mean | `0.144` m |
| Episodes starting inside the 5 cm tolerance | **6.05 %** |
| Episodes starting within 10 cm | 24.85 % |

**The no-op floor is 6 %, six times Reach's 1.05 %.** A policy that never touches the cube scores
about 6 % on §13.1's success rate, because the cube starts near the centre of a region only
0.30 x 0.40 m. §13's success rate must be read against that floor, and §14's random-action baseline
measures it directly. It is a property of the geometry, not of `beta`: widening the target region,
shrinking `cube_jitter`, or excluding a disc around the cube's start would each reduce it.

### `beta = 30` is frozen, and Reach's `alpha = 10` does not transfer

`alpha = 10` was chosen for Reach only after confirming the reward spanned its range across the
distances that task produces. Carrying the same constant to Push does **not** hold, because Push's
distances are roughly half Reach's and distance enters squared:

| | Reach, `alpha = 10` | Push, `beta = 10` | Push, `beta = 30` |
|---|---|---|---|
| median `d_0` | 0.272 m | 0.146 m | 0.146 m |
| `r` at median | 0.478 | 0.807 | 0.526 |
| `r` at the worst start | 0.082 | 0.398 | 0.063 |
| spread (max - min) | 0.915 | 0.602 | 0.936 |

At `beta = 10` the reward is compressed into its top 60 %: an untouched cube already scores 0.81 at
the median start. `beta = 30` reproduces `alpha = 10`'s character for this distribution almost
exactly, and is what the code and the YAML now carry.

The wrapper's own sampling was checked against this model over 2000 seeds: median `d_0` 0.1441 m
against the analytic 0.1463, and a 6.70 % floor against 6.05 % — within one standard error of a 6 %
proportion at that sample size (±0.53 %).

## 14. Observation (§15)

    o_t = [ q_t, q̇_t, p_ee,t, p_cube,t, ṗ_cube,t, g ]

| Block | Dim | Source |
|---|---|---|
| `q_t` | 9 | `data.qpos[:9]` - arm + fingers, **not** the cube's free joint |
| `q̇_t` | 9 | `data.qvel[:9]` |
| `p_ee,t` | 3 | hand body pose + TCP offset, as Reach |
| `p_cube,t` | 3 | `data.xpos[cube]` |
| `ṗ_cube,t` | 3 | `data.qvel[dofadr:dofadr+3]` - a free joint's first three dofs are world-frame linear velocity |
| `g` | 3 | planar target, `z` fixed |
| **total** | **30** | |

`q_t` deliberately excludes the cube's free-joint coordinates: they would duplicate `p_cube` and add
a quaternion §15 does not ask for. It also keeps `state[:9]` meaning joint positions, which
`scripts/buffer_metrics.py` relies on for both tasks.

## 15. Episode structure

| Item | Value |
|---|---|
| `max_episode_steps` | `200` control steps |
| Episode duration | `10.0` s = `200 * 0.05` |
| `terminated` | always `False`, as Reach |

Double Reach's horizon: pushing needs approach, contact and transport where reaching needs only
approach. Not yet validated against a measured time-to-success - if 10 s proves too short for any
policy to finish, the success rate floors at the 6 % no-op rate and the task is uninformative.

## 16. `info` dict

As Reach's §7, with `distance` now cube-to-goal, plus:

| Key | Type | Purpose |
|---|---|---|
| `cube_position` | `(3,) float32` | cube trajectory, and any later cube-coverage metric |

`ctrl_saturation` is absent: §13.2's saturation is computed from the `diag_*.f32` stream, which
records the normalized action directly.

## 17. Out of scope for Panda Push

Coverage stays **end-effector** coverage (§13.2), unchanged from Reach. Whether the cube reached
interesting regions is an evaluation question about a trained task actor, not an exploration metric,
and is deferred.
