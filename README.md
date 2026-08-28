# plan2explore-menagerie

Standalone Plan2Explore (DreamerV3, SheepRL) baseline on MuJoCo Menagerie robot arms.

    Menagerie robot + standalone Gymnasium task + unmodified SheepRL Plan2Explore

This repository is **self-contained**. It imports no external research codebase, methodology,
representation, dataset, or evaluation code. Its two upstream dependencies — SheepRL and MuJoCo
Menagerie — are pinned by commit SHA in [`UPSTREAM.md`](UPSTREAM.md) and cloned by
`SYNTAX.md`.

## Algorithm-purity rule

Plan2Explore is **frozen upstream code**. The following are never edited:

    sheeprl/sheeprl/algos/p2e_dv3/
    sheeprl/sheeprl/algos/dreamer_v3/

Modifications are restricted to the environment side:

    sheeprl/sheeprl/envs/            <- MenageriePandaReach wrapper (installed, see below)
    sheeprl/sheeprl/configs/env/     <- menagerie_panda_reach.yaml (installed, see below)
    menagerie_tasks/                 <- task MJCF
    scripts/                         <- run + evaluation scripts

    adapt the environment to the baseline, not the baseline to the experiment.

Verify with `git -C sheeprl status --porcelain` (nothing under `algos/`) and
`git -C sheeprl rev-parse HEAD` against the pin in `UPSTREAM.md`.

### Where the two Option-A files actually live

The notes prescribe **Option A**: the env wrapper and its config sit inside the SheepRL tree at
`sheeprl/sheeprl/envs/menagerie_panda.py` and `sheeprl/sheeprl/configs/env/menagerie_panda_reach.yaml`.

The SheepRL checkout is a pinned clone and is `.gitignore`d, so the canonical, version-controlled
copies of those two files live in `menagerie_integration/`. `scripts/install_env_wrapper.sh` symlinks
them into the checkout (copying if the filesystem refuses symlinks). At runtime the layout is exactly
Option A, and `_target_` resolves to `sheeprl.envs.menagerie_panda.MenageriePandaReach`.

## Layout

    README.md                 UPSTREAM.md  ENVIRONMENT_SPEC.md  requirements-lock.txt
    sheeprl/                  pinned SheepRL clone            (gitignored)
    third_party/
      mujoco_menagerie/       pinned Menagerie clone          (gitignored)
    menagerie_integration/    canonical source of the Option-A wrapper + env config
    menagerie_tasks/          panda_reach.xml, later panda_push.xml
    scripts/                  phase scripts, gates, evaluation, verification
    tests/                    Phase 4 environment tests (notes 10)
    results/
      manifests/              one reproducibility manifest per run (notes 17)
      summaries/              aggregated multi-seed metrics

Replay buffers, checkpoints, videos and TensorBoard logs stay outside Git.

## Provenance map - which file referrences which section

Every file in this repository implements a specific section of
`../References/SelfEx-WM_Notes.tex` and carries that reference in its own header.

| File | Notes section |
|---|---|
| `README.md`, `.gitignore` | §5 Repository Layout |
| `SYNTAX.md` | §18 Minimal Execution Order; §19 Definition of Success — every command, with flags explained |
| `UPSTREAM.md` | §6 Phase 0: Freeze the Software — pins, toolchain, and every run-affecting decision |
| `ENVIRONMENT_SPEC.md` | §8 Phase 2: First Menagerie Task — Panda Reach (§8.1–§8.4), all frozen constants |
| `requirements-lock.txt` | §6, §19 — `pip freeze --all` of the Python 3.11 venv |
| `menagerie_integration/menagerie_panda.py` | §8.1–§8.4 task/obs/action/control; §9 Phase 3 Gymnasium env; §13 trajectory logging |
| `menagerie_integration/menagerie_panda_reach.yaml` | §11.1 Phase 5 Option A env config |
| `scripts/install_env_wrapper.sh` | §11.1 Phase 5 Option A — links both files into the SheepRL checkout |
| `scripts/common.sh` | shared plumbing: paths, pins, `check_pins()` |
| `tests/test_environment.py` | §10 Phase 4 — all ten rows of the table, one test each |
| `scripts/metrics.py` | §13.1 task, §13.2 exploration, §13.3 world-model metrics |
| `scripts/random_baseline.py` | §14 Phase 8: Random Baseline — drives the uniform-random policy |
| `scripts/baseline_metrics.py` | §14 Phase 8 — metrics for a baseline run (no world model, so no §13.3) |

Phase 0 and Phase 1 have no scripts by choice: they are one-time, environment-dependent setup, and
`SYNTAX.md` gives the commands directly.

## Quickstart (Linux + CUDA)

**See [`SYNTAX.md`](SYNTAX.md)** for every command with its flags explained. In outline:

1. Clone and pin SheepRL and Menagerie to the SHAs in `UPSTREAM.md` (§6)
2. `python3.11 -m venv .venv311`, install from `requirements-lock.txt` (§7)
3. `bash scripts/install_env_wrapper.sh` — Option A links into the SheepRL tree (§11.1)
4. `pytest tests/` — the §10 environment tests
5. Train, then `python scripts/metrics.py run ...` per seed (§12, §13)
6. `python scripts/random_baseline.py ...` and `scripts/baseline_metrics.py` (§14)

## Notes on versions

SheepRL declares `requires-python = ">=3.8,<3.12"`, so the virtualenv is built with **Python 3.11**.
It also pins `gymnasium==0.29.*`, whose MuJoCo environments stop at `v4`; the Phase 1 upstream smoke
test therefore uses `HalfCheetah-v4`. Bumping Gymnasium is outside the allowed-modification list.

## Execution order (notes 18)

 1. Clone and pin SheepRL.
 2. Clone and pin MuJoCo Menagerie.
 3. Install SheepRL with MuJoCo support.
 4. Run Plan2Explore-DreamerV3 on an existing Gymnasium MuJoCo task.
 5. Confirm GPU use, replay, intrinsic reward, ensemble learning, and logging.
 6. Implement Panda Reach as a standalone Gymnasium environment.
 7. Pass environment API and determinism tests.
 8. Run a long random-action environment stress test.
 9. Run a tiny Plan2Explore Panda smoke test.
10. Run a 3-seed Panda Reach pilot.
11. Freeze the environment and training specification.
12. Run the 5-seed Panda Reach baseline.
13. Add Panda Push.
14. Only then expand to UR5e, iiwa14, xArm7, and Sawyer if useful.

## References

- Sekar et al., *Planning to Explore via Self-Supervised World Models*, ICML 2020 —
  <https://arxiv.org/abs/2005.05960>
- Original Plan2Explore — <https://github.com/ramanans1/plan2explore>
- SheepRL — <https://github.com/Eclectic-Sheep/sheeprl>
- MuJoCo Menagerie — <https://github.com/google-deepmind/mujoco_menagerie>
