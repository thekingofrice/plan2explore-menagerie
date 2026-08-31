#!/usr/bin/env python
"""§13.2 actuator saturation and joint-limit visitation, recovered from a run's replay buffer.

Implements: References/SelfEx-WM_Notes.tex §13.2, for runs that predate the wrapper recording these
per step. The buffer stores every transition it collected, and a transition holds both inputs:
observations["state"][:9] is qpos (§8.2) and actions are normalized, pre-denormalization (§8.3).

Valid only while the buffer has never wrapped -- it is written front to back, so rows [0, _pos) are
chronological per environment and support curves as well as totals. buffer.size 1e6 over num_envs 4
gives 250,000 rows per environment against the 125,000 a 5e5-step run needs, so a §12.3 run fits.
Refuses if `full` is set, when row order no longer reflects time.

Prefer the wrapper's own stream where it exists: the buffer is gigabytes, gitignored, and lives in
version_0/memmap_buffer/ -- deleting that directory destroys the history for every resumed segment.

Usage:
    python scripts/buffer_metrics.py --checkpoint .../checkpoint/ckpt_340000_0.ckpt \\
        --seed 1 --out results/summaries/gateC_seed1_buffer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))
sys.path.insert(0, str(REPO_ROOT / "sheeprl"))

from metrics import joint_limit_metrics, saturation_metrics  # noqa: E402

N_QPOS = 9  # state = [qpos(9), qvel(9), p_ee(3), g(3)] per §8.2


def read_buffer(ckpt_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Return per-environment qpos and normalized-action arrays over the filled region."""
    original = torch.load

    def load(*args, **kwargs):
        # torch 2.6 defaults weights_only=True, which refuses the pickled replay buffer. Same
        # patch scripts/resume.py applies; see UPSTREAM.md.
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        rb = torch.load(str(ckpt_path), map_location="cpu")["rb"]
    finally:
        torch.load = original

    qpos, actions, lengths = [], [], []
    for i, sub in enumerate(rb.buffer):
        if getattr(sub, "full", False):
            raise RuntimeError(
                f"environment {i}'s buffer has wrapped, so row order no longer reflects time. "
                "Totals would still be valid over the retained window, but the curves would not."
            )
        pos = int(sub._pos)
        state = np.asarray(sub._buf["state"][:pos]).reshape(pos, -1)
        act = np.asarray(sub._buf["actions"][:pos]).reshape(pos, -1)
        qpos.append(state[:, :N_QPOS])
        actions.append(act)
        lengths.append(pos)
    return qpos, actions, lengths


def joint_limits_from_model() -> tuple[np.ndarray, np.ndarray]:
    """jnt_range and jnt_limited from the frozen MJCF, with the one-qpos-entry-per-joint check.

    Imported lazily by scripts/metrics.py, which otherwise needs no MuJoCo.
    """
    from menagerie_panda import MenageriePandaReach

    env = MenageriePandaReach()
    try:
        if env.model.nq != env.model.njnt:
            raise RuntimeError(
                f"nq={env.model.nq} != njnt={env.model.njnt}: some joint does not contribute "
                "exactly one qpos entry, so state[:9] is not a per-joint position vector and "
                "jnt_range cannot be indexed alongside it. Expected once §15 adds a free-floating "
                "cube (7 qpos, 1 joint); joint_limit_metrics needs a qpos-address map first."
            )
        return env.model.jnt_range.copy(), env.model.jnt_limited.copy()
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    qpos, actions, lengths = read_buffer(args.checkpoint)
    print(f"{len(lengths)} environments, rows per env: {lengths}")
    print(f"  {sum(lengths)} total interactions")

    jnt_range, jnt_limited = joint_limits_from_model()

    result = {
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "source": "replay_buffer",
        # The checkpoint's filename is its policy_step, but the buffer can hold a little more: the
        # steps collected after the last checkpoint and before the crash are still in the memmap.
        "steps_recorded": int(sum(lengths)),
        "steps_per_env": lengths,
        **saturation_metrics(actions),
        **joint_limit_metrics(qpos, jnt_range, jnt_limited),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("curve")}, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
