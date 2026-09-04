#!/usr/bin/env python
"""Resume a killed run (§12) under PyTorch >= 2.6.

torch.load's `weights_only` default became True in 2.6. SheepRL checkpoints hold a pickled
EnvIndependentReplayBuffer (buffer.checkpoint defaults to True), which is not an allowed global, so
fabric.load raises UnpicklingError before the run starts. Full unpickling is safe here -- the file is
this project's own checkpoint -- and allowlisting globals one at a time does not terminate, since the
buffer pickle also carries SequentialReplayBuffer, numpy memmaps and TensorDicts.

sheeprl/algos/ is untouched on disk, so §3 holds and verify_algorithm_untouched.sh still passes.

Overrides pass straight through, but on a resume sheeprl/cli.py reloads the checkpoint's config and
merges it over the command line, popping five keys first. Those five are the only ones that change
anything:

    root_dir  run_name  algo.total_steps  algo.learning_starts  checkpoint.resume_from

`env=` and `exp=` are still needed for Hydra to compose a config, but the checkpoint's values replace
them. That is why nothing here names a robot or a task, and why §16's arms need no change.

Three of the five are filled in from the checkpoint when resuming; anything you pass wins:

    root_dir, run_name       read off the checkpoint's path, so a typo cannot fork a new run tree
    algo.learning_starts=0   sheeprl adds start_iter to it on every resume, so any nonzero value
                             injects a second random-action prefill mid-run, silently

    python scripts/resume.py \\
      exp=p2e_dv3_exploration \\
      env=<the run's env config> \\
      checkpoint.resume_from=<absolute path to ckpt_<step>_0.ckpt> \\
      algo.total_steps=<the original, absolute budget>

Finetuning is the other caller and gets none of the three: it opens a new run tree from an
exploration checkpoint, and its buffer starts empty, so it needs its own prefill (SYNTAX.md §6d).

Move the trajectory directory aside first -- filenames carry the pid, so the resume cannot reopen the
old files, and `env.wrapper.trajectory_log` is not one of the five, so the resume writes into the
original directory whatever you pass. Both sets in one directory corrupt coverage_curve (SYNTAX.md
§6b).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SHEEPRL_DIR = REPO_ROOT / "sheeprl"


def patch_torch_load() -> None:
    """Restore torch.load's pre-2.6 default so the checkpointed replay buffer unpickles."""
    original = torch.load

    def load(*args, **kwargs):
        # setdefault, not override: a caller that deliberately asks for weights_only=True keeps it.
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load


def run_with_postmortem(run) -> None:
    """Run, then drop into pdb on the frame that raised instead of losing it with the traceback.

    A try/except rather than sys.excepthook, because Hydra prints and re-raises and anything in
    between could replace the hook. Costs nothing until something raises -- unlike `python -m pdb`,
    whose line tracing would slow every step of a run that takes hours to reach its failure.
    """
    import pdb
    import traceback

    try:
        run()
    except Exception:
        traceback.print_exc()
        pdb.post_mortem(sys.exc_info()[2])
        raise  # keep the nonzero exit; pdb is for looking, not for swallowing


def find_override(overrides: list[str], name: str) -> str | None:
    """The value of `name=...` in the override list, or None if the caller did not pass it."""
    match = [o for o in overrides if o.startswith(f"{name}=")]
    return match[0].split("=", 1)[1] if match else None


def derive_run_identity(ckpt: Path) -> tuple[str, str]:
    """root_dir and run_name of the run a checkpoint belongs to, read off its path.

        logs/runs/<root_dir>/<run_name>/version_N/checkpoint/ckpt_<step>_0.ckpt

    root_dir is everything between `runs` and the run directory, so a one- or two-component value
    both work and no robot or task is named. Raises ValueError if the path is not that shape --
    guessing here would silently start a new run instead of continuing this one.
    """
    if ckpt.parent.name != "checkpoint":
        raise ValueError(f"{ckpt.parent.name!r} is not a checkpoint directory")
    version = ckpt.parent.parent
    if not version.name.startswith("version_"):
        raise ValueError(f"{version.name!r} is not a version_N directory")

    run_dir = version.parent
    parts = run_dir.parts
    if "runs" not in parts:
        raise ValueError(f"no `runs` component in {run_dir}")

    # rindex: `runs` could appear earlier in an absolute path, and the log tree's own is the last.
    anchor = len(parts) - 1 - parts[::-1].index("runs")
    root_dir = "/".join(parts[anchor + 1 : -1])
    if not root_dir:
        raise ValueError(f"nothing between `runs` and the run directory in {run_dir}")
    return root_dir, parts[-1]


def main() -> int:
    overrides = sys.argv[1:]
    if not overrides:
        print(__doc__)
        return 2

    if not SHEEPRL_DIR.is_dir():
        print(f"SheepRL not cloned at {SHEEPRL_DIR} -- see SYNTAX.md step 1", file=sys.stderr)
        return 1

    # Two callers, two key names: resuming an interrupted run uses checkpoint.resume_from, while
    # p2e_dv3_finetuning takes the exploration run's checkpoint as checkpoint.exploration_ckpt_path.
    # Both need the same torch.load patch, so both are accepted here.
    key, ckpt = "", ""
    for name in ("checkpoint.resume_from", "checkpoint.exploration_ckpt_path"):
        value = find_override(overrides, name)
        if value is not None:
            key, ckpt = name, value
            break

    if not key:
        print(
            "no checkpoint override found. Pass checkpoint.resume_from=<ckpt> to continue an "
            "interrupted run, or checkpoint.exploration_ckpt_path=<ckpt> to finetune from one.",
            file=sys.stderr,
        )
        return 1

    if not ckpt or not Path(ckpt).is_file():
        # An empty or wrong path is falsy to SheepRL's `if cfg.checkpoint.resume_from:`, so it would
        # silently start a FRESH run from step 0. An unset shell variable produces exactly this.
        print(
            f"{key} does not point at a file: {ckpt!r}\n"
            "Refusing: SheepRL treats an unset value as 'start from scratch', which would discard "
            "the run you are trying to continue from.",
            file=sys.stderr,
        )
        return 1

    # Hydra writes logs relative to the working directory, and training runs are launched from
    # sheeprl/ (SYNTAX.md step 5). Anything else puts this run's logs somewhere new.
    os.chdir(SHEEPRL_DIR)
    sys.path.insert(0, str(SHEEPRL_DIR))

    patch_torch_load()
    print(f"{key} = {ckpt}\n")

    from sheeprl.cli import run

    sys.argv = ["sheeprl.py", *overrides]
    if os.environ.get("RESUME_POSTMORTEM"):
        print("RESUME_POSTMORTEM set: an unhandled exception will open pdb on the raising frame\n")
        run_with_postmortem(run)
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
