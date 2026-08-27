#!/usr/bin/env python
"""§14 Phase 8: metrics for the random baseline.

Implements: References/SelfEx-WM_Notes.tex §14, reporting the §13.1 task metrics and §13.2
exploration metrics for a run produced by scripts/random_baseline.py.

Separate from scripts/metrics.py because a random baseline has no world model, so §13.3's losses and
the intrinsic-reward tags do not exist -- there is no TensorBoard event file to read. Everything it
*can* report is imported from metrics.py rather than reimplemented: voxelization, the reference
workspace and the task summary must be bit-identical across the two runs, or the comparison §14 asks
for compares analysis choices instead of policies.

Usage:
    python scripts/baseline_metrics.py --traj-dir results/runs/random_seed0/trajectories \
        --seed 0 --out results/summaries/random_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from menagerie_panda import MenageriePandaReach  # noqa: E402
from metrics import REFERENCE_SAMPLES, VOXEL_SIZE, reference_workspace, run_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=REFERENCE_SAMPLES,
        help="must match the value used for the training runs being compared against",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    env = MenageriePandaReach()
    print(f"building reference workspace from {args.reference_samples} samples...")
    reference = reference_workspace(env, args.reference_samples, seed=0)
    print(f"  {len(reference)} reachable voxels at {VOXEL_SIZE} m")
    env.close()

    result = {
        "policy": "random",
        "seed": args.seed,
        "traj_dir": str(args.traj_dir),
        **run_metrics(args.traj_dir, reference),
    }
    # Named for the acting policy, matching metrics.py, so the two JSONs line up key for key.
    result["task_metrics_random_actor"] = result.pop("task_metrics_exploration_actor", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["exploration_metrics"], indent=2)[:400])
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
