#!/usr/bin/env python
"""§13.1 and §13.3 for a p2e_dv3_finetuning run.

Implements: References/SelfEx-WM_Notes.tex §13.1 (task metrics) and §13.3 (world-model metrics) for
the few-shot phase. Separate from scripts/metrics.py because finetuning declares different
TensorBoard keys, and because two of §13's metrics stop being meaningful here:

    §13.2 coverage        the acting policy is the TASK actor, so trajectories are task-directed.
                          That is not exploration coverage and must never be pooled with Gate C's
                          C_workspace. Not computed here at all.
    §13.2 intrinsic       the ensembles are not restored by finetuning, so intrinsic reward and
                          ensemble disagreement do not exist.

What changes in the tag names: the task actor's losses lose their `_task` suffix because there is
only one actor now, and Loss/ensemble_loss is gone with the ensembles.

Usage:
    python scripts/finetuning_metrics.py \\
        --traj-dir results/runs/finetune_seed0/trajectories \\
        --logdir <run>/version_0 \\
        --seed 0 --out results/summaries/finetune_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from metrics import read_trajectories, world_model_metrics  # noqa: E402

#: §13.3's world-model losses plus §12.1's actor/critic finiteness, under finetuning's names.
#: Loss/ensemble_loss is absent by design: p2e_dv3_finetuning does not restore the ensembles.
FINETUNING_TAGS = [
    # §13.3
    "Loss/world_model_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "State/kl",
    "Loss/state_loss",
    "Loss/continue_loss",
    # §12.1 -- unsuffixed, because finetuning trains only the task actor
    "Loss/policy_loss",
    "Loss/value_loss",
    # gradient norms, the finetuning analogue of Grads/actor_task and Grads/critic_task
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
]

#: p2e_dv3_finetuning.yaml's default. Episodes finishing before this many policy steps were driven
#: by random actions, not by the task actor.
DEFAULT_LEARNING_STARTS = 16_384


def task_metrics(episodes: np.ndarray) -> dict:
    """§13.1 from the recorded episode rows: (total_steps, return, final_distance, success)."""
    if not len(episodes):
        return {"episodes": 0}
    returns, distances, successes = episodes[:, 1], episodes[:, 2], episodes[:, 3]
    return {
        "success_rate": float((successes > 0.5).mean()),
        "return_mean": float(returns.mean()),
        "return_median": float(np.median(returns)),
        "final_distance_mean": float(distances.mean()),
        "final_distance_median": float(np.median(distances)),
        "episodes": int(len(episodes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-dir", type=Path, action="append", required=True, dest="traj_dirs", metavar="DIR",
        help="repeat once per segment, oldest first, for a run that was resumed",
    )
    parser.add_argument("--resumed-at", type=int, action="append", default=None, metavar="STEPS")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--logdir", action="append", required=True, dest="logdirs", metavar="DIR",
        help="the run's version_N directory; repeat once per segment",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=DEFAULT_LEARNING_STARTS,
        help="policy steps of random-action prefill to exclude from the post-prefill task metrics",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ee, episodes, _diag = read_trajectories(
        args.traj_dirs, resumed_at=args.resumed_at, num_envs=args.num_envs
    )
    n_envs = len(ee)

    # Episode rows carry a PER-ENVIRONMENT step count, so the prefill boundary has to be divided
    # by the environment count before it can be compared against them.
    prefill_per_env = args.learning_starts // max(n_envs, 1)
    after_prefill = episodes[episodes[:, 0] > prefill_per_env] if len(episodes) else episodes

    result = {
        "seed": args.seed,
        "phase": "finetuning",
        "actor": "task",
        "regime": "few-shot",
        "traj_dirs": [str(d) for d in args.traj_dirs],
        "logdirs": args.logdirs,
        "steps_recorded": int(sum(len(a) for a in ee)),
        "steps_per_env": [int(len(a)) for a in ee],
        "episodes_recorded": int(len(episodes)),
        "learning_starts": args.learning_starts,
        # On-policy: the task actor is what acts during finetuning, so unlike an exploration run
        # these ARE the task actor's numbers. They are still on-policy during training, which is
        # not the same as evaluate_task_actor.py's held-out evaluation of a finished checkpoint.
        "task_metrics_onpolicy_all": task_metrics(episodes),
        "task_metrics_onpolicy_after_prefill": task_metrics(after_prefill),
        "world_model_metrics": world_model_metrics(args.logdirs, tags=FINETUNING_TAGS),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "world_model_metrics"}, indent=2))
    missing = [k for k, v in result["world_model_metrics"].items() if v is None]
    print(f"\nworld-model tags resolved: {len(FINETUNING_TAGS) - len(missing)}/{len(FINETUNING_TAGS)}")
    if missing:
        print(f"  missing: {missing}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
