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

_Not yet recorded. Run `bash scripts/phase0_freeze.sh`, then `bash scripts/record_upstream.sh`._

| Repository | URL | Commit SHA | Recorded (UTC) |
|---|---|---|---|
| SheepRL | https://github.com/Eclectic-Sheep/sheeprl | `<pending>` | `<pending>` |
| MuJoCo Menagerie | https://github.com/google-deepmind/mujoco_menagerie | `<pending>` | `<pending>` |

## Toolchain

| Item | Value |
|---|---|
| `python_version` | `<pending>` |
| `torch_version` | `<pending>` |
| `torch.version.cuda` | `<pending>` |
| `torch.cuda.is_available()` | `<pending>` |
| `mujoco_version` | `<pending>` |
| `gymnasium_version` | `<pending>` |
| `lightning_version` | `<pending>` |
| GPU (`nvidia-smi`) | `<pending>` |
| NVIDIA driver | `<pending>` |
| OS / kernel | `<pending>` |

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