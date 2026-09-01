#!/usr/bin/env python
"""Map where the Panda can put its gripper at cube height without the arm being inside the table.

Implements: References/SelfEx-WM_Notes.tex §15, the Push analogue of the target-box validation
ENVIRONMENT_SPEC.md §2 records for Reach ("The target box must be validated as genuinely reachable").

Nothing else checks this. The smoke test steps zero actions, which §8.3 maps to the midpoint of each
ctrlrange -- nowhere near the cube. If the arm cannot get to cube height across the target region,
the task is unsolvable and every other test still passes.

Method: sample random arm configurations uniformly within jnt_range, run forward kinematics, and
keep the ones whose end effector lands at cube height. Same approach reference_workspace uses for
Reach's box. Unlike inverse kinematics it has no local minima and needs no seeding -- the Panda is
7-DOF and redundant, so a single IK solve finds one solution out of many and can report a point
blocked when another configuration reaches it cleanly.

Kinematic only. A configuration existing does not prove the position controller can drive to it, but
if no configuration exists then no policy can, and that is the cheap failure to rule out first.

Usage:
    MUJOCO_GL=egl python scripts/probe_push_reachability.py
    MUJOCO_GL=egl python scripts/probe_push_reachability.py --samples 500000 --step 0.025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))

from menagerie_panda_push import (  # noqa: E402
    EE_OFFSET,
    N_ARM_JOINTS,
    TABLE_GEOM,
    MenageriePandaPush,
)


def table_bounds(env) -> tuple[float, float, float, float]:
    """World-frame x/y extent of the table top, so the grid can be masked to it.

    Cells with no table beneath them are trivially contact-free, so counting them as reachable is
    measuring where the table is absent rather than where the arm can work.
    """
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    mujoco.mj_forward(env.model, env.data)
    c, h = env.data.geom_xpos[gid], env.model.geom_size[gid]
    return c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1]


def sample_reachable(env, n_samples: int, z_tol: float, seed: int):
    """Random arm configurations -> end-effector positions at cube height, split by table contact.

    Returns (clean_xy, blocked_xy): points where the gripper reached working height with the arm
    clear of the table, and where it reached but some link was inside it.
    """
    model, data = env.model, env.data
    rng = np.random.default_rng(seed)
    lo = model.jnt_range[:N_ARM_JOINTS, 0]
    hi = model.jnt_range[:N_ARM_JOINTS, 1]
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)

    env._reset_model()
    # Park the cube far away: arm/cube contacts would otherwise be indistinguishable from arm/table
    # ones. Only mj_forward runs below, never mj_step, so nothing falls.
    adr = env._cube_qpos_adr
    data.qpos[adr : adr + 3] = [10.0, 10.0, 10.0]

    clean, blocked = [], []
    for _ in range(n_samples):
        data.qpos[:N_ARM_JOINTS] = rng.uniform(lo, hi)
        mujoco.mj_forward(model, data)

        xmat = data.xmat[env._ee_body_id].reshape(3, 3)
        p_ee = data.xpos[env._ee_body_id] + xmat @ EE_OFFSET
        if abs(p_ee[2] - env.cube_rest_z) > z_tol:
            continue

        hit = any(
            table_gid in (data.contact[c].geom1, data.contact[c].geom2)
            for c in range(data.ncon)
        )
        (blocked if hit else clean).append(p_ee[:2].copy())

    return np.array(clean).reshape(-1, 2), np.array(blocked).reshape(-1, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--step", type=float, default=0.05, help="grid spacing, metres")
    parser.add_argument(
        "--z-tol", type=float, default=0.02,
        help="how close to the cube's centre height the gripper must land to count",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = MenageriePandaPush()
    print(f"base z {env.table_top_z:.3f}   cube rest z {env.cube_rest_z:.3f}   "
          f"z tolerance +/-{args.z_tol}")
    x_lo, x_hi, y_lo, y_hi = table_bounds(env)
    print(f"table spans x [{x_lo:.2f}, {x_hi:.2f}]  y [{y_lo:.2f}, {y_hi:.2f}]")
    print(f"sampling {args.samples} arm configurations...\n")

    clean, blocked = sample_reachable(env, args.samples, args.z_tol, args.seed)
    print(f"{len(clean)} clean and {len(blocked)} blocked configurations landed at cube height "
          f"({100 * (len(clean) + len(blocked)) / args.samples:.1f}% of samples)\n")

    xs = np.arange(x_lo, x_hi + 1e-9, args.step)
    ys = np.arange(y_lo, y_hi + 1e-9, args.step)

    def counts(points):
        grid = np.zeros((len(xs), len(ys)), dtype=int)
        if not len(points):
            return grid
        i = np.clip(((points[:, 0] - x_lo) / args.step).astype(int), 0, len(xs) - 1)
        j = np.clip(((points[:, 1] - y_lo) / args.step).astype(int), 0, len(ys) - 1)
        np.add.at(grid, (i, j), 1)
        return grid

    c_grid, b_grid = counts(clean), counts(blocked)

    print("      " + "".join(f"{y:>5.2f}" for y in ys))
    for i, x in enumerate(xs):
        row = "".join(
            "    ." if c_grid[i, j] else ("    T" if b_grid[i, j] else "    x")
            for j in range(len(ys))
        )
        print(f"{x:>5.2f} {row}")
    print("\n  .  reachable clear of the table    T  only with a link inside it    x  no configuration found")

    ok = c_grid > 0
    if not ok.any():
        print("\nNo configuration puts the gripper at cube height clear of the table.")
        env.close()
        return 1

    xi, yi = np.where(ok)
    fx = (xs[xi.min()], xs[xi.max()])
    fy = (ys[yi.min()], ys[yi.max()])
    hits = c_grid[ok]
    print(f"\nclean footprint over the table:  x [{fx[0]:.2f}, {fx[1]:.2f}]   y [{fy[0]:.2f}, {fy[1]:.2f}]")
    print(f"  {ok.sum()} of {ok.size} in-table cells reachable")
    print(f"  hits per reachable cell: min {hits.min()}  median {int(np.median(hits))}  max {hits.max()}")
    if hits.min() < 10:
        print("  ** some cells have very few hits -- they may be rare rather than unreachable; "
              "raise --samples before trusting a sparse edge")

    tl, th = env.target_box_low, env.target_box_high
    print(f"\ncurrent target region x [{tl[0]:.2f}, {th[0]:.2f}]  y [{tl[1]:.2f}, {th[1]:.2f}]")
    inside = fx[0] <= tl[0] and th[0] <= fx[1] and fy[0] <= tl[1] and th[1] <= fy[1]
    print("  fully inside the clean footprint" if inside
          else "  ** extends OUTSIDE the clean footprint -- some goals have no collision-free pose")

    env.close()
    return 0 if inside else 1


if __name__ == "__main__":
    raise SystemExit(main())
