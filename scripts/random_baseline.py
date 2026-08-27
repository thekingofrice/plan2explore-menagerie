#!/usr/bin/env python
"""§14 Phase 8: Random Baseline.

Implements: References/SelfEx-WM_Notes.tex §14.

    "run the exact same Panda task with uniformly random normalized actions: a_t ~ U([-1,1]^m).
     Use the same environment, reset distribution, episode horizon, number of environment
     interactions, and seeds where appropriate. This determines whether Plan2Explore's intrinsic
     exploration actually produces additional coverage."

Writes trajectories in the same format a training run does, so scripts/metrics.py analyses both
through one code path and the two coverage numbers are comparable by construction rather than by
argument.

No GPU and no world model: this is environment stepping only, so it can run alongside a training job.

Usage -- one invocation per seed, matching the training budget:
    python scripts/random_baseline.py --steps 500000 --seed 0 \
        --traj-dir results/runs/random_seed0/trajectories
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))

from menagerie_panda import MenageriePandaReach  # noqa: E402


def run_baseline(steps: int, seed: int, traj_dir: Path) -> dict:
    """Step the environment under uniform random actions for a fixed number of interactions.

    Seeding mirrors a training run: the episode index offsets the base seed, so episode n of the
    baseline faces the same target and initial pose as episode n of a training run with the same
    base seed. That is what §14's "same reset distribution" and "same seeds where appropriate"
    require -- without it the two runs would face different tasks and the coverage comparison would
    conflate policy quality with task difficulty.
    """
    env = MenageriePandaReach(trajectory_log=str(traj_dir))

    taken = 0
    episode = 0
    started = time.time()

    while taken < steps:
        env.reset(seed=seed + episode)
        episode += 1

        while taken < steps:
            action = env.np_random.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            taken += 1

            if terminated or truncated:
                break

        if episode % 100 == 0:
            rate = taken / max(time.time() - started, 1e-9)
            print(f"  {taken}/{steps} steps, {episode} episodes, {rate:.1f} steps/s", flush=True)

    env.close()  # records the episode in flight and flushes both files

    elapsed = time.time() - started
    return {
        "steps": taken,
        "episodes": episode,
        "seconds": round(elapsed, 1),
        "steps_per_second": round(taken / max(elapsed, 1e-9), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="environment interactions, matched to the training run being compared against",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--traj-dir", type=Path, required=True)
    args = parser.parse_args()

    print(f"§14 random baseline: {args.steps} steps, seed {args.seed} -> {args.traj_dir}")
    stats = run_baseline(args.steps, args.seed, args.traj_dir)
    print(
        f"done: {stats['steps']} steps over {stats['episodes']} episodes "
        f"in {stats['seconds']}s ({stats['steps_per_second']} steps/s)"
    )
    print("\nanalyse with the same command used for a training run:")
    print(
        f"  python scripts/metrics.py run --traj-dir {args.traj_dir} "
        f"--seed {args.seed} --out results/summaries/random_seed{args.seed}.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
