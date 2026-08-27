# UPSTREAM.md

> Implements: `References/SelfEx-WM_Notes.tex` **§6 Phase 0: Freeze the Software**, and supplies the
> `sheeprl_commit` / `menagerie_commit` / toolchain fields of **§17 Reproducibility Manifest**.

This file is the pin. Once experiments begin, nothing may rely on a floating `main` branch.

Regenerate with:

    bash scripts/record_upstream.sh

That script rewrites everything below the `<!-- BEGIN RECORDED -->` marker. Do not hand-edit below it.

---

## Pinned commits

<!-- BEGIN RECORDED -->

Recorded 2026-08-25 on `trlab-ubuntu`.

| Repository | URL | Commit SHA | Recorded (UTC) |
|---|---|---|---|
| SheepRL | https://github.com/Eclectic-Sheep/sheeprl | `33b636681fd8b5340b284f2528db8821ab8dcd0b` | 2026-08-25 |
| MuJoCo Menagerie | https://github.com/google-deepmind/mujoco_menagerie | `da76818e269b82289eba39808e2fb91d679d6994` | 2026-08-25 |

Verified present at these commits:

- `sheeprl/sheeprl/algos/p2e_dv3/` — `agent.py`, `evaluate.py`, `p2e_dv3_exploration.py`,
  `p2e_dv3_finetuning.py`, `utils.py`
- `third_party/mujoco_menagerie/franka_emika_panda/` — `panda.xml`, `scene.xml`, `hand.xml`,
  `mjx_panda_nohand.xml`, `assets/`

## Toolchain

Host: `trlab-ubuntu`.

| Item | Value |
|---|---|
| `python_version` | `3.11.10` |
| OS / kernel | `Linux 6.8.0-136-generic #136~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC x86_64` (Ubuntu 22.04) |
| GPU | `NVIDIA GeForce RTX 4090`, 24564 MiB |
| NVIDIA driver | `580.173.02` |
| Driver CUDA capability | `13.0` |
| `torch_version` | `<pending — Phase 1>` |
| `torch.version.cuda` | `<pending — Phase 1>` |
| `torch.cuda.is_available()` | `<pending — Phase 1>` |
| `mujoco_version` | `<pending — Phase 1>` |
| `gymnasium_version` | `<pending — Phase 1>` |
| `lightning_version` | `<pending — Phase 1>` |

The GPU is shared — `nvidia-smi` showed ~3.4 GB already in use by other users' processes at record
time, leaving ~21 GB. DreamerV3 at the default XL size on vector observations fits comfortably, but
run budgets should not assume an empty card.

## Environment pins that are load-bearing

These are not implied by any upstream package's metadata. Without them a fresh venv fails.

`requirements-lock.txt` must be generated with **`pip freeze --all`**. Plain `pip freeze` omits
`setuptools`, `pip` and `wheel`, which would silently drop the `setuptools<81` pin below and hand the
next person the same `pkg_resources` failure the lock file exists to prevent.

| Pin | Why |
|---|---|
| `setuptools<81` | setuptools 84 removed `pkg_resources`. `lightning-utilities 0.9.0` (pulled in by `lightning 2.3.0`) still does `import pkg_resources` in `core/imports.py`, so `import lightning` — and therefore all of SheepRL — raises `ModuleNotFoundError: No module named 'pkg_resources'`. Pinning below 81 restores the module. |
| `export MUJOCO_GL=egl` | Headless over SSH. MuJoCo defaults to GLFW, which needs an X11 `DISPLAY` and dies with `GLFWError: X11: The DISPLAY environment variable is missing` then `mujoco.FatalError: gladLoadGL error`. EGL renders offscreen on the GPU with no display server. |
| `env.sync_env=True` | With `MUJOCO_GL=egl`, `AsyncVectorEnv`'s forked workers inherit an EGL context created in the parent and die immediately (`ConnectionResetError: [Errno 104] Connection reset by peer`, with the real exception swallowed). Sync mode has no workers. Only required when rendering is in the step path. |
| `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | The stock DreamerV3 XL config (`recurrent_state_size: 4096`, `dense_units: 1024`, `mlp_layers: 5`, 8-member ensemble, `precision: 32-true`) needs ~23.4 GiB of the RTX 4090's 23.5 GiB, on a card shared with other users. Without this it OOMs in the exploration actor's imagination rollout. |

The `setuptools<81` pin makes `lightning_utilities` emit a `UserWarning` about `pkg_resources` being
deprecated on every run. It is expected and harmless; the warning itself recommends this exact pin.
The alternative fix — upgrading `lightning-utilities` past the versions that use `pkg_resources` —
was rejected because it moves a Lightning component underneath a frozen DreamerV3 implementation
mid-experiment. Revisit only between baselines, never within one.

None of the above touches `sheeprl/algos/`, so §3's deviation log does not apply. They are
environment and runtime settings only.

## Phase 1 outcome (§7)

`exp=p2e_dv3_exploration env=gym env.id=HalfCheetah-v4` reached and entered the training loop with
the GPU active: environment constructed, EGL rendering working, world model and 8-member ensemble
built, replay collecting past `learning_starts=1024`, no NaNs, no simulator instability, no crash.

**Not verified:** the run was stopped before `metric.log_every=5000` policy steps, so no scalars were
ever written and §7's "world-model and ensemble losses train / intrinsic reward is nonconstant /
actor-critic losses are finite" criteria were never actually observed. Treated as passed on the basis
that the run proceeded without error.

**Open concern for §12.** The default `exp` config runs HalfCheetah from **pixels**
(`cnn_keys.encoder: [rgb]`), so every environment step renders a 64x64 frame through EGL, and
`sync_env=True` steps the 4 envs serially. The run did not reach 5000 policy steps in ~40 minutes.
That rate makes §12.2's 10^5-step pilot and §12.3's 5x10^5-10^6-step baseline expensive, and the
budgets must be sized against a measured step rate before Gate B is launched.

This cost is specific to pixel observations. Panda Reach is a **vector**-observation task per §8.2,
with no CNN keys and no rendering in the step path, so it should be substantially faster and may not
need `sync_env=True` at all.

## Phase 5 outcome (§11)

`env=menagerie_panda_reach` resolves, `_target_` imports, and Plan2Explore trains on the Panda with no
crash. Confirmed 2026-08-26. Two sets of overrides were required; both are configuration, so
`sheeprl/algos/` remains untouched and §3's deviation log does not apply.

### Observation keys

The stock `exp=p2e_dv3_exploration` declares `cnn_keys.encoder=[rgb]` and `cnn_keys.decoder=[rgb]`,
which are defaults for pixel tasks. Panda Reach emits a single `"state"` key (§8.2), so SheepRL
raised `ValueError: The user specified keys ['rgb'] are not a subset of the environment ... keys`.
Required:

    algo.cnn_keys.encoder=[]  algo.cnn_keys.decoder=[]
    algo.mlp_keys.encoder=[state]  algo.mlp_keys.decoder=[state]

Adding an `rgb` observation instead — to "adapt the environment to the baseline" per §3 — was
considered and rejected: §8.2 specifies a vector-observation experiment and §17's manifest field is
`observation_mode: vector`, so it would contradict the section being implemented. It would also
reintroduce per-step EGL rendering, the suspected cause of the Phase 1 slowness above. A pixel
version of Panda Reach is a legitimate later experiment with its own manifest, not a way around a
config override.

### Model size: DreamerV3-S, not XL

SheepRL's default is DreamerV3-**XL** (`recurrent_state_size 4096`, `dense_units 1024`,
`mlp_layers 5`), sized for 64x64 pixel observations. Panda Reach's observation is a 24-dimensional
vector, and Hafner's own paper uses the S/M configurations for proprioceptive control. Reduced to S:

    algo.dense_units=512  algo.mlp_layers=2
    algo.world_model.recurrent_model.recurrent_state_size=512
    algo.world_model.transition_model.hidden_size=512
    algo.world_model.representation_model.hidden_size=512

`dense_units` and `mlp_layers` propagate by interpolation to the encoder, decoder, reward model,
discount model, actor, critic and ensembles.

`algo.ensembles.n` stays at **8**. Disagreement across those members *is* Plan2Explore's intrinsic
reward, so shrinking it would change the exploration signal rather than the compute cost.
`per_rank_batch_size` (16) and `per_rank_sequence_length` (64) stay at their defaults, so the
training recipe is stock and only the network size differs.

Per §4.1 the result is reported as **Plan2Explore (DreamerV3-S, SheepRL)**.

### GPU contention

The RTX 4090 is shared and persistently carries ~11.6 GB of other users' processes, leaving ~12 GB.
XL needed 12.11 GB and OOM'd in the imagination rollout even with `per_rank_batch_size=8` — that
memory is dominated by `batch x seq_len x horizon` latent states, which the batch size alone cannot
fix. S was adopted for correctness of task sizing, not only to fit.

## Measured throughput (§12 sizing)

2400 policy steps in ~12 minutes, seed 0, DreamerV3-S, `num_envs: 4`. That average splits in two:
the first ~1024 steps are pure collection at roughly 30 steps/s (GPU idle, `learning_starts: 1024`),
and the remainder run at **≈ 2 steps/s** once `replay_ratio: 1` starts a gradient update per
environment step.

Sequential wall-clock at 2 steps/s:

| Gate | Steps | Time |
|---|---|---|
| §12.2 B — 3 seeds x 10^5 | 3x10^5 | ~42 h |
| §12.3 C — 5 seeds x 5x10^5 | 2.5x10^6 | ~14 days |
| §12.3 C — 5 seeds x 10^6 | 5x10^6 | ~29 days |

Gate C as specified is not feasible sequentially. Seeds are independent experiments, so running them
concurrently is free scientifically and is the first lever to try — DreamerV3-S is small enough that
several should fit in the ~12 GB left by other users. `algo.replay_ratio` is the only other large
lever, and it changes how much the world model learns per unit of experience, so it would have to be
frozen and recorded like `alpha` and the model size.

`algo.total_steps` defaults to **5,000,000** — far beyond even §12.3's 10^6 — so every gate must set
it explicitly.

## Gate A (§12.1): abbreviated by decision

Run as a confirmation that the pipeline executes end to end, not as an evaluation of §12.1's pass
criteria. Agreed between Minh and their advisor, 2026-08-26.

Confirmed: the run starts, trains, and completes without NaNs or simulator instability; GPU use is
confirmed; TensorBoard scalars are written; `Rewards/intrinsic` and the loss curves were inspected
visually and look healthy.

Not done: any quantitative check of §12.1's criteria — intrinsic-reward variance, ensemble-loss
descent and actor/critic finiteness were judged by eye rather than measured. §13's metrics are the
instrument for that, and Gate B is the first run they will be applied to.

<!-- END RECORDED -->

---

## Notes on the pins

**Python 3.11.** SheepRL declares `requires-python = ">=3.8,<3.12"`. The virtualenv built by
`scripts/phase1_install.sh` is therefore Python 3.11.

**Gymnasium 0.29.** SheepRL pins `gymnasium==0.29.*`. That release line's MuJoCo environments stop at
`v4`, so the §7 upstream smoke test runs `HalfCheetah-v4`. Upgrading Gymnasium would be a change
outside the §3 allowed-modification list and is not permitted.

**Menagerie is models only.** It supplies the Franka Emika Panda MJCF under
`third_party/mujoco_menagerie/franka_emika_panda/`. It supplies no reinforcement-learning task; the
task is defined by this repository (`menagerie_tasks/panda_reach.xml` and
`menagerie_integration/menagerie_panda.py`).

## Upstream deviations log

Per §3, the Plan2Explore implementation is frozen. Any change to
`sheeprl/sheeprl/algos/p2e_dv3/` or `sheeprl/sheeprl/algos/dreamer_v3/` is permitted **only** for a
genuine upstream compatibility bug that makes execution impossible, must be recorded here, and must
not change the scientific algorithm.

| Date | File | Reason execution was impossible | Diff | Scientific impact |
|---|---|---|---|---|
| _(none)_ | | | | |

`scripts/verify_algorithm_untouched.sh` fails the run if this table is empty but the algorithm
directories differ from the pinned SHA.