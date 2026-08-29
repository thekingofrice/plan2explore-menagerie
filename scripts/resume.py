#!/usr/bin/env python
"""Resume a killed run (§12) under PyTorch >= 2.6.

torch.load's `weights_only` default became True in 2.6. SheepRL checkpoints hold a pickled
EnvIndependentReplayBuffer (buffer.checkpoint defaults to True), which is not an allowed global, so
fabric.load raises UnpicklingError before the run starts. Full unpickling is safe here -- the file is
this project's own checkpoint -- and allowlisting globals one at a time does not terminate, since the
buffer pickle also carries SequentialReplayBuffer, numpy memmaps and TensorDicts.

sheeprl/algos/ is untouched on disk, so §3 holds and verify_algorithm_untouched.sh still passes.

Overrides pass straight through. Only these five matter -- every other key is force-restored from the
checkpoint's config.yaml, and anything else you type is silently discarded:

    python scripts/resume.py \\
      exp=p2e_dv3_exploration \\
      env=menagerie_panda_reach \\
      checkpoint.resume_from=<absolute path to ckpt_<step>_0.ckpt> \\
      root_dir=p2e_dv3_exploration/MenageriePandaReach \\
      run_name=<the original run directory name> \\
      algo.total_steps=500000 \\
      algo.learning_starts=0

Move the trajectory directory aside first -- filenames carry the pid, so the resume cannot reopen the
old files, and both sets in one directory corrupt coverage_curve (SYNTAX.md §6b).
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


def main() -> int:
    overrides = sys.argv[1:]
    if not overrides:
        print(__doc__)
        return 2

    if not SHEEPRL_DIR.is_dir():
        print(f"SheepRL not cloned at {SHEEPRL_DIR} -- see SYNTAX.md step 1", file=sys.stderr)
        return 1

    resume = [o for o in overrides if o.startswith("checkpoint.resume_from=")]
    ckpt = resume[0].split("=", 1)[1] if resume else ""
    if not ckpt or not Path(ckpt).is_file():
        # An empty or wrong path is falsy to `if cfg.checkpoint.resume_from:`, so SheepRL would
        # silently start a FRESH run from step 0 instead of resuming. Fail loudly instead.
        print(
            f"checkpoint.resume_from does not point at a file: {ckpt!r}\n"
            "Refusing: SheepRL treats an unset value as 'start from scratch', which would discard "
            "the run you are trying to continue.",
            file=sys.stderr,
        )
        return 1

    # Hydra writes logs relative to the working directory, and training runs are launched from
    # sheeprl/ (SYNTAX.md step 5). Anything else puts this run's logs somewhere new.
    os.chdir(SHEEPRL_DIR)
    sys.path.insert(0, str(SHEEPRL_DIR))

    patch_torch_load()
    print(f"resuming from {ckpt}\n")

    from sheeprl.cli import run

    sys.argv = ["sheeprl.py", *overrides]
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
