#!/usr/bin/env python
"""Map where the Panda can actually work the table surface, and size the table from it.

Implements: References/SelfEx-WM_Notes.tex §15, the Push analogue of the target-box validation
ENVIRONMENT_SPEC.md §2 records for Reach ("The target box must be validated as genuinely reachable").

Nothing else checks this. The smoke test steps zero actions, which §8.3 maps to the midpoint of each
ctrlrange -- nowhere near the cube. If the arm cannot get down to cube height across the target
region, the task is unsolvable and every other test still passes.

Kinematic only: damped-least-squares IK on the 7 arm joints, from the home pose, for each point of a
grid over the table plane. If IK cannot find a pose, no policy can. IK converging does not prove a
position controller can get there dynamically, but it is the cheap failure to rule out first.

Two answers come out:
    reachable footprint   where the end effector can be placed at cube height, cleanly
    table size            the footprint bounds the table that is worth having

Usage:
    MUJOCO_GL=egl python scripts/probe_push_reachability.py
    MUJOCO_GL=egl python scripts/probe_push_reachability.py --step 0.025 --tol 0.005
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

#: Damping for the least-squares solve. Large enough to stay stable near singularities, small enough
#: not to stall convergence.
LAMBDA = 0.05

#: Cartesian step gain per iteration, and how many iterations before giving up.
GAIN = 0.5
MAX_ITERS = 300


def ee_position(model, data, body_id) -> np.ndarray:
    xmat = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + xmat @ EE_OFFSET


def solve_ik(env, target: np.ndarray, tol: float) -> tuple[float, bool, int]:
    """Damped least squares from the home pose to `target`.

    Returns the final Cartesian error, whether any arm joint ended clipped at a limit, and how many
    geom pairs are in contact with the table at the solution.

    Only the 7 arm joints move: the fingers are driven by the policy anyway, and the cube's free
    joint is not something IK should be steering.
    """
    model, data = env.model, env.data
    lo = model.jnt_range[:N_ARM_JOINTS, 0]
    hi = model.jnt_range[:N_ARM_JOINTS, 1]

    env._reset_model()
    # Park the cube far away so arm/cube contacts do not pollute the table-contact count. Only
    # mj_forward is called below, never mj_step, so nothing falls or moves.
    adr = env._cube_qpos_adr
    data.qpos[adr : adr + 3] = [10.0, 10.0, 10.0]
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(MAX_ITERS):
        current = ee_position(model, data, env._ee_body_id)
        error = target - current
        if np.linalg.norm(error) < tol:
            break

        mujoco.mj_jac(model, data, jacp, jacr, current, env._ee_body_id)
        jac = jacp[:, :N_ARM_JOINTS]
        # dq = J^T (J J^T + lambda^2 I)^-1 dx -- damped so a singular configuration produces a
        # bounded step instead of an enormous one.
        jjt = jac @ jac.T + (LAMBDA**2) * np.eye(3)
        dq = jac.T @ np.linalg.solve(jjt, error * GAIN)

        data.qpos[:N_ARM_JOINTS] = np.clip(data.qpos[:N_ARM_JOINTS] + dq, lo, hi)
        mujoco.mj_forward(model, data)

    final = float(np.linalg.norm(target - ee_position(model, data, env._ee_body_id)))
    q = data.qpos[:N_ARM_JOINTS]
    clipped = bool(np.any(np.isclose(q, lo, atol=1e-6)) or np.any(np.isclose(q, hi, atol=1e-6)))

    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    contacts = sum(
        1
        for c in range(data.ncon)
        if table_gid in (data.contact[c].geom1, data.contact[c].geom2)
    )
    return final, clipped, contacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-range", type=float, nargs=2, default=(0.00, 0.90))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-0.50, 0.50))
    parser.add_argument("--step", type=float, default=0.05, help="grid spacing, metres")
    parser.add_argument("--tol", type=float, default=0.01, help="a point counts as reached within this")
    args = parser.parse_args()

    env = MenageriePandaPush()
    z = env.cube_rest_z
    print(f"base raised to {env.table_top_z:.3f}   cube rest z {z:.3f}   tol {args.tol} m")
    env._reset_model()
    mujoco.mj_forward(env.model, env.data)
    print(f"p_ee(home) = {ee_position(env.model, env.data, env._ee_body_id).round(4)}\n")

    xs = np.arange(args.x_range[0], args.x_range[1] + 1e-9, args.step)
    ys = np.arange(args.y_range[0], args.y_range[1] + 1e-9, args.step)

    ok = np.zeros((len(xs), len(ys)), dtype=bool)
    print("      " + "".join(f"{y:>6.2f}" for y in ys))
    for i, x in enumerate(xs):
        row = []
        for j, y in enumerate(ys):
            err, clipped, contacts = solve_ik(env, np.array([x, y, z]), args.tol)
            reached = err < args.tol
            ok[i, j] = reached and contacts == 0
            # '.' clean; 'T' reachable only by putting the arm through the table; 'x' out of reach
            row.append("  .  " if ok[i, j] else ("  T  " if reached else "  x  "))
        print(f"{x:>5.2f} " + "".join(row))

    print("\n  .  reachable with no table contact    T  reachable but through the table    x  out of reach")

    if not ok.any():
        print("\nNOTHING is cleanly reachable at cube height. The table is too high, too close, "
              "or the base elevation is wrong.")
        env.close()
        return 1

    xi, yi = np.where(ok)
    x_lo, x_hi = xs[xi.min()], xs[xi.max()]
    y_lo, y_hi = ys[yi.min()], ys[yi.max()]
    print(f"\nreachable footprint at z={z:.3f}:  x [{x_lo:.2f}, {x_hi:.2f}]   y [{y_lo:.2f}, {y_hi:.2f}]")
    print(f"  {ok.sum()} of {ok.size} grid points clean")

    # The table only needs to span what the arm can work; anything beyond is decoration the arm
    # can collide with. Half-extents and centre for panda_push.xml, with the top left where it is.
    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    hx, hy = (x_hi - x_lo) / 2, (y_hi - y_lo) / 2
    print(f"\ntable sized to the footprint:")
    print(f'  <body name="table" pos="{cx:.3f} {cy:.3f} {env.table_top_z / 2:.3f}">')
    print(f'    <geom name="table_top" type="box" size="{hx:.3f} {hy:.3f} {env.table_top_z / 2:.3f}" .../>')

    print(f"\ncurrent target region x {tuple(env.target_box_low)[0]:.2f}..{tuple(env.target_box_high)[0]:.2f}"
          f"  y {tuple(env.target_box_low)[1]:.2f}..{tuple(env.target_box_high)[1]:.2f}")
    inside = (
        x_lo <= env.target_box_low[0] and env.target_box_high[0] <= x_hi
        and y_lo <= env.target_box_low[1] and env.target_box_high[1] <= y_hi
    )
    print("  target region is fully inside the reachable footprint" if inside
          else "  ** target region extends OUTSIDE the reachable footprint -- some goals are unsolvable")

    env.close()
    return 0 if inside else 1


if __name__ == "__main__":
    raise SystemExit(main())
