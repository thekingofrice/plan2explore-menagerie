# plan2explore-menagerie

Standalone Plan2Explore (DreamerV3, SheepRL) baseline on MuJoCo Menagerie robot arms.

    Menagerie robot + standalone Gymnasium task + unmodified SheepRL Plan2Explore

This repository is **self-contained**. It imports no external research codebase, methodology,
representation, dataset, or evaluation code. Its two upstream dependencies — SheepRL and MuJoCo
Menagerie — are pinned by commit SHA in [`UPSTREAM.md`](UPSTREAM.md) and cloned by
`scripts/phase0_freeze.sh`.

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

`scripts/verify_algorithm_untouched.sh` enforces this and must pass before any run is reported.

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
| `UPSTREAM.md` | §6 Phase 0: Freeze the Software; §17 Reproducibility Manifest |
| `scripts/phase0_freeze.sh` | §6 Phase 0: Freeze the Software |
| `scripts/record_upstream.sh` | §6 Phase 0 (commit SHAs + toolchain versions) |
| `scripts/phase1_install.sh` | §7 Phase 1: Install and Validate SheepRL Before Menagerie |
| `scripts/smoke_upstream.sh` | §7 Phase 1 (upstream smoke test); §12.1 Gate A |
| `ENVIRONMENT_SPEC.md` | §8 Phase 2: First Menagerie Task — Panda Reach (§8.1–§8.4) |
| `menagerie_integration/menagerie_panda.py` | §8.1–§8.4 task/obs/action/control; §9 Phase 3 Gymnasium env |
| `menagerie_integration/menagerie_panda_reach.yaml` | §11.1 Phase 5 Option A |
| `scripts/install_env_wrapper.sh` | §11.1 Phase 5 Option A |
| `scripts/verify_algorithm_untouched.sh` | §3 Algorithm-Purity Rule |
| `tests/test_env_api.py` | §10 Phase 4 — "API test" |
| `tests/test_reset_determinism.py` | §10 Phase 4 — "Reset determinism" |
| `tests/test_action_bounds.py` | §10 Phase 4 — "Action bounds" |
| `tests/test_zero_action.py` | §10 Phase 4 — "Zero action" |
| `tests/test_finite_state.py` | §10 Phase 4 — "Finite state" |
| `tests/test_joint_safety.py` | §10 Phase 4 — "Joint safety" |
| `tests/test_target_sampling.py` | §10 Phase 4 — "Target sampling" |
| `tests/test_reward.py` | §10 Phase 4 — "Reward monotonicity" |
| `tests/test_render.py` | §10 Phase 4 — "Render test" |
| `tests/test_seed_separation.py` | §10 Phase 4 — "Seed separation" |
| `scripts/stress_rollout.py` | §10 Phase 4 (several-thousand-step random-control stress test) |
| `scripts/smoke_panda.sh` | §12.1 Gate A on the Panda task |
| `scripts/train_panda_reach.sh` | §12.2 Gate B pilot; §12.3 Gate C baseline |
| `scripts/eval_panda_reach.sh` | §13.1 Task metrics (task actor evaluation) |
| `scripts/metrics/task_metrics.py` | §13.1 Task metrics |
| `scripts/metrics/coverage.py` | §13.2 Exploration metrics (workspace coverage) |
| `scripts/metrics/world_model_metrics.py` | §13.3 World-model metrics |
| `scripts/random_baseline.py` | §14 Phase 8: Random Baseline |
| `scripts/write_manifest.py` | §17 Reproducibility Manifest |
| `menagerie_tasks/` *(deferred)* | §8 / §15 task MJCF — see ENVIRONMENT_SPEC.md "Why no task MJCF" |

## Quickstart (Linux + CUDA)

    bash scripts/phase0_freeze.sh          # clone + pin SheepRL and Menagerie, record SHAs
    bash scripts/phase1_install.sh         # python3.11 venv, pip install -e ".[mujoco,dev,test]"
    bash scripts/install_env_wrapper.sh    # Option A: link wrapper + config into the SheepRL tree
    bash scripts/smoke_upstream.sh         # Gate A on stock Gymnasium MuJoCo, before any robot code
    pytest tests/                          # Phase 4 environment tests
    bash scripts/smoke_panda.sh            # tiny Plan2Explore run on MenageriePandaReach

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
