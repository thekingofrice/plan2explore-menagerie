#!/usr/bin/env python
"""§13 Phase 7: Metrics.

Implements: References/SelfEx-WM_Notes.tex §13.

    §13.1 Task metrics          success rate, mean/median task return, final target distance
    §13.2 Exploration metrics   workspace coverage, state-space coverage, actuator saturation,
                                joint-limit visitation, intrinsic reward and ensemble disagreement
    §13.3 World-model metrics   world-model, observation, reward, KL/state and ensemble losses

§13.1 and §13.2 are computed from rollouts. §13.3 is already logged to TensorBoard by upstream
SheepRL, so it is read back from the event files rather than recomputed -- §3 keeps the algorithm
frozen, and that includes not reimplementing its loss reporting.

Usage:
    # rollout metrics with a uniform-random policy (this is also §14's baseline)
    python scripts/metrics.py rollout --policy random --episodes 100 --seed 0 --out results/summaries/random.json

    # world-model losses from a finished run
    python scripts/metrics.py losses --logdir sheeprl/logs/runs/p2e_dv3_exploration/... --out results/summaries/run0_losses.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))

from menagerie_panda import MenageriePandaReach  # noqa: E402

#: §13.2 voxel edge length for workspace coverage, metres. Chosen equal to §8.1's 5 cm success
#: tolerance so a "visited voxel" is the same spatial resolution the task is scored at.
VOXEL_SIZE = 0.05

#: Number of random joint configurations used to build the reference reachable set (§13.2's
#: denominator). The end effector's reachable volume has no closed form, so it is sampled.
REFERENCE_SAMPLES = 200_000


# --------------------------------------------------------------------- §13.2

def voxelize(points: np.ndarray, voxel_size: float = VOXEL_SIZE) -> set[tuple[int, int, int]]:
    """Map 3-D points onto integer voxel indices."""
    idx = np.floor(np.asarray(points) / voxel_size).astype(np.int64)
    return {tuple(v) for v in idx}


def reference_workspace(env, n_samples: int = REFERENCE_SAMPLES, seed: int = 0) -> set:
    """The denominator of C_workspace: voxels the end effector can reach at all (§13.2).

    Sampled by drawing uniform joint configurations within jnt_range and running forward kinematics.
    This is the *kinematically* reachable set -- it ignores whether a policy could get there under
    the dynamics, which makes it a conservative (large) denominator and C_workspace a lower bound.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    lo = env.model.jnt_range[:7, 0]
    hi = env.model.jnt_range[:7, 1]

    positions = np.empty((n_samples, 3))
    for i in range(n_samples):
        env.data.qpos[:7] = rng.uniform(lo, hi)
        mujoco.mj_forward(env.model, env.data)
        positions[i] = env._ee_position()

    return voxelize(positions)


def workspace_coverage(ee_per_env: list[np.ndarray], reference: set) -> dict:
    """C_workspace = #visited voxels / #reachable voxels (§13.2).

    Takes one position array per environment and unions the voxels they visited. Grouping does not
    affect the result -- a set union is order-independent -- so this number is identical to what a
    concatenated array would give, and is correct for any num_envs.
    """
    visited: set = set()
    for positions in ee_per_env:
        visited |= voxelize(positions)

    # Visits outside the sampled reference set are possible (the reference is itself a sample), so
    # the ratio is reported against the union to keep it bounded by 1.
    reachable = reference | visited
    return {
        "visited_voxels": len(visited),
        "reference_voxels": len(reference),
        "C_workspace": len(visited) / max(len(reachable), 1),
        "voxel_size_m": VOXEL_SIZE,
        "n_envs": len(ee_per_env),
    }


# --------------------------------------------------------------------- §13.1

def rollout(env, policy, episodes: int, seed: int) -> dict:
    """Run episodes and collect everything §13.1 and §13.2 need from the environment side."""
    returns, final_distances, successes = [], [], []
    ee_positions, states, saturations = [], [], []
    joint_limit_hits = np.zeros(env.model.njnt, dtype=np.int64)
    joint_steps = 0

    lo = env.model.jnt_range[:, 0]
    hi = env.model.jnt_range[:, 1]
    limited = env.model.jnt_limited.astype(bool)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return = 0.0

        while True:
            action = policy(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += reward
            ee_positions.append(info["ee_position"])
            states.append(obs["state"])
            saturations.append(info["ctrl_saturation"])

            qpos = env.data.qpos[: env.model.njnt]
            near = limited & (
                np.isclose(qpos, lo, atol=1e-3) | np.isclose(qpos, hi, atol=1e-3)
            )
            joint_limit_hits += near
            joint_steps += 1

            if terminated or truncated:
                break

        returns.append(ep_return)
        final_distances.append(float(info["distance"]))
        # An episode counts as successful if the end state is within §8.1's tolerance -- the arm must
        # still be on target when the episode truncates, not merely have passed through.
        successes.append(bool(info["success"]))

    return {
        "returns": np.array(returns),
        "final_distances": np.array(final_distances),
        "successes": np.array(successes),
        "ee_positions": np.array(ee_positions),
        "states": np.array(states),
        "saturations": np.array(saturations),
        "joint_limit_fraction": joint_limit_hits / max(joint_steps, 1),
    }


def task_metrics(data: dict) -> dict:
    """§13.1: success rate, mean/median task return, final target distance."""
    return {
        "success_rate": float(data["successes"].mean()),
        "return_mean": float(data["returns"].mean()),
        "return_median": float(np.median(data["returns"])),
        "final_distance_mean": float(data["final_distances"].mean()),
        "final_distance_median": float(np.median(data["final_distances"])),
        "episodes": int(len(data["returns"])),
    }


def exploration_metrics(data: dict, reference: set) -> dict:
    """§13.2, minus the intrinsic-reward and ensemble terms, which come from TensorBoard."""
    # A rollout drives a single environment, so its positions are one "per-env" array.
    out = workspace_coverage([data["ee_positions"]], reference)
    out["actuator_saturation_fraction"] = float(np.mean(data["saturations"]))
    out["joint_limit_visitation"] = data["joint_limit_fraction"].round(6).tolist()
    # State-space coverage: the same voxel idea applied to the full observation, reported as the
    # count of distinct cells rather than a ratio, since the reachable state set has no reference.
    grid = np.floor(data["states"] / VOXEL_SIZE).astype(np.int64)
    out["state_space_cells"] = int(len({tuple(row) for row in grid}))
    return out


# ------------------------------------------------- §13.1 + §13.2 from a run

def _read_rows(path: Path, width: int) -> np.ndarray:
    """Read a raw float32 stream as (-1, width), discarding any trailing partial row.

    The environment appends to these files continuously, so reading one mid-run can catch a torn
    write. Truncating to whole rows keeps a live check from crashing on reshape.
    """
    flat = np.fromfile(path, dtype=np.float32)
    return flat[: len(flat) - len(flat) % width].reshape(-1, width)


def read_trajectories(
    traj_dir: Path, max_steps: int | None = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Read the streams written by MenageriePandaReach's trajectory_log.

    Returns the position arrays as a LIST, one per environment process, rather than concatenated.
    Concatenating loses which positions came from which environment, and since the files are written
    independently, gluing them end to end produces "all of env 0, then part of env 1" -- fine for a
    final coverage number, which is an order-independent set union, but meaningless as a time axis.

    ``max_steps`` is a ONE-OFF remedy, not routine practice: it exists solely because Gate B's seed 0
    was launched with the wrong `algo.total_steps` and overran to ~117,200 interactions, and must be
    analysed at the same 100,000 as seeds 1 and 2 or the seed variance is meaningless. Every
    subsequent run sets its budget correctly and should be read with max_steps=None.

    When given, it caps TOTAL interactions across all environments (SheepRL's policy_step
    convention): each environment is truncated to max_steps // n_envs, and episode rows whose
    per-environment step count exceeds that are dropped. Truncation is lossless because training is
    causal -- the retained prefix is byte-identical to what stopping at that budget would have
    produced.

    Episodes are returned concatenated, shape (M, 4): (total_steps, return, final_distance, success),
    where total_steps is PER-ENVIRONMENT (a factor of n_envs smaller than TensorBoard's policy_step).
    """
    traj_dir = Path(traj_dir)
    ee = [_read_rows(p, 3) for p in sorted(traj_dir.glob("ee_*.f32"))]
    ep = [_read_rows(p, 4) for p in sorted(traj_dir.glob("episodes_*.f32"))]
    if not ee:
        raise FileNotFoundError(f"no ee_*.f32 files in {traj_dir}")

    if max_steps is not None:
        per_env = max_steps // len(ee)
        ee = [a[:per_env] for a in ee]
        ep = [rows[rows[:, 0] <= per_env] for rows in ep]

    episodes = np.concatenate(ep) if ep else np.empty((0, 4), np.float32)
    return ee, episodes


def coverage_curve(ee_per_env: list[np.ndarray], reference: set, n_points: int = 50) -> dict:
    """C_workspace as a function of total environment steps.

    A single end-of-run number cannot distinguish an agent still finding new regions from one that
    plateaued early, which is the substance of the exploration claim §14 tests.

    Environments step in lockstep -- one step each per iteration -- so coverage at total step T is
    the union over every environment's first T // n_envs positions. Taking a prefix of the
    environments concatenated end to end would instead give "all of env 0 plus part of env 1", which
    is not a point in training time at all. At num_envs=1 this reduces to the plain chronological
    prefix.

    ``steps`` is reported in TOTAL interactions, so it shares an x-axis with TensorBoard's
    policy_step rather than with the per-environment counts in the episode rows.
    """
    n_envs = len(ee_per_env)
    per_env_len = min((len(a) for a in ee_per_env), default=0)
    if per_env_len == 0:
        return {"steps": [], "C_workspace": []}

    marks = np.unique(
        np.linspace(max(per_env_len // n_points, 1), per_env_len, n_points).astype(int)
    )
    steps, values = [], []
    for m in marks:
        visited: set = set()
        for positions in ee_per_env:
            visited |= voxelize(positions[:m])
        values.append(len(visited) / max(len(reference | visited), 1))
        steps.append(int(m * n_envs))
    return {"steps": steps, "C_workspace": values}


def run_metrics(traj_dir: Path, reference: set, max_steps: int | None = None) -> dict:
    """§13.1 and §13.2 from a completed run's recorded trajectory.

    The acting policy during exploration is the EXPLORATION actor, so the task numbers describe what
    that actor achieved -- they are not an evaluation of the task actor, which needs sheeprl_eval.py
    against a checkpoint. Coverage is unambiguous: it is exactly the set of positions visited while
    exploring, which is what §13.2 asks for.

    ``max_steps`` is the one-off cap described in read_trajectories; leave it None for any run whose
    budget was set correctly.
    """
    ee, ep = read_trajectories(traj_dir, max_steps=max_steps)
    exploration = workspace_coverage(ee, reference)
    exploration["coverage_curve"] = coverage_curve(ee, reference)

    out = {
        # Total interactions across all environments, matching TensorBoard's policy_step. ee is a
        # list of per-environment arrays, so this sums them rather than counting the list.
        "steps_recorded": int(sum(len(a) for a in ee)),
        "steps_per_env": [int(len(a)) for a in ee],
        "episodes_recorded": int(len(ep)),
        "exploration_metrics": exploration,
    }
    if len(ep):
        returns, distances, successes = ep[:, 1], ep[:, 2], ep[:, 3]
        out["task_metrics_exploration_actor"] = {
            "success_rate": float((successes > 0.5).mean()),
            "return_mean": float(returns.mean()),
            "return_median": float(np.median(returns)),
            "final_distance_mean": float(distances.mean()),
            "final_distance_median": float(np.median(distances)),
            "episodes": int(len(ep)),
        }
    return out


# --------------------------------------------------------------------- §13.3

def world_model_metrics(logdir: str) -> dict:
    """§13.3, read back from the TensorBoard events SheepRL already writes.

    Also covers the intrinsic-reward and ensemble-disagreement terms §13.2 asks for, since those are
    logged by the algorithm rather than observable from the environment.
    """
    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(
        str(logdir), size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()

    # Exploration-critic metrics are suffixed with the critic name, per the p2e_dv3 convention
    # <metric_key>_<critic_key>. cfg.algo.critics_exploration has two entries, `intrinsic` and
    # `extrinsic`, so "Rewards/intrinsic" is really "Rewards/intrinsic_intrinsic" -- the suffix
    # appended to a key that was already called intrinsic. Not a typo; do not "correct" it.
    wanted = [
        # §13.3
        "Loss/world_model_loss",
        "Loss/observation_loss",
        "Loss/reward_loss",
        "State/kl",
        "Loss/state_loss",
        "Loss/ensemble_loss",
        # §13.2 -- intrinsic reward, which in Plan2Explore IS the ensemble disagreement:
        # reward = next_state_embedding.var(0).mean(-1) * intrinsic_reward_multiplier, and the
        # multiplier is 1. One quantity, not two.
        "Rewards/intrinsic_intrinsic",
        # §12.1's actor/critic finiteness
        "Loss/policy_loss_task",
        "Loss/value_loss_task",
        "Loss/policy_loss_exploration",
        "Loss/value_loss_exploration_intrinsic",
        "Loss/value_loss_exploration_extrinsic",
    ]

    available = set(acc.Tags().get("scalars", []))
    out = {}
    for tag in wanted:
        if tag not in available:
            out[tag] = None
            continue
        values = np.array([e.value for e in acc.Scalars(tag)])
        out[tag] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "first": float(values[0]),
            "last": float(values[-1]),
            "finite": bool(np.all(np.isfinite(values))),
            "n": int(len(values)),
        }
    return out


# ---------------------------------------------------------------------- CLI

def random_policy(obs, env):
    """§14: a_t ~ U([-1, 1]^m). Drawn from the env's RNG so a seed reproduces the whole rollout."""
    return env.np_random.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_roll = sub.add_parser("rollout", help="§13.1 + §13.2 from environment rollouts")
    p_roll.add_argument("--policy", choices=["random"], default="random")
    p_roll.add_argument("--episodes", type=int, default=100)
    p_roll.add_argument("--seed", type=int, default=0)
    p_roll.add_argument("--reference-samples", type=int, default=REFERENCE_SAMPLES)
    p_roll.add_argument("--out", type=Path, required=True)

    p_loss = sub.add_parser("losses", help="§13.3 from a run's TensorBoard events")
    p_loss.add_argument("--logdir", required=True)
    p_loss.add_argument("--out", type=Path, required=True)

    p_run = sub.add_parser("run", help="every §12.2 metric for one finished training run")
    p_run.add_argument("--traj-dir", type=Path, required=True, help="env.wrapper.trajectory_log")
    p_run.add_argument("--logdir", required=True, help="the run's version_0 directory")
    p_run.add_argument("--seed", type=int, required=True)
    p_run.add_argument("--reference-samples", type=int, default=REFERENCE_SAMPLES)
    p_run.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="ONE-OFF: cap total interactions, for a run whose budget was set wrong. Gate B seed 0 "
        "overran to ~117200 and needs --max-steps 100000 to match seeds 1 and 2. Omit otherwise.",
    )
    p_run.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "rollout":
        env = MenageriePandaReach()
        print(f"building reference workspace from {args.reference_samples} samples...")
        reference = reference_workspace(env, args.reference_samples, seed=args.seed)
        print(f"  {len(reference)} reachable voxels at {VOXEL_SIZE} m")

        print(f"rolling out {args.episodes} episodes with the {args.policy} policy...")
        data = rollout(env, random_policy, args.episodes, args.seed)
        result = {
            "policy": args.policy,
            "seed": args.seed,
            "task_metrics": task_metrics(data),
            "exploration_metrics": exploration_metrics(data, reference),
        }
        env.close()
    elif args.command == "losses":
        result = {"logdir": args.logdir, "world_model_metrics": world_model_metrics(args.logdir)}

    else:  # "run": every §12.2 metric for one finished training run
        env = MenageriePandaReach()
        print(f"building reference workspace from {args.reference_samples} samples...")
        reference = reference_workspace(env, args.reference_samples, seed=0)
        print(f"  {len(reference)} reachable voxels at {VOXEL_SIZE} m")
        env.close()

        result = {
            "seed": args.seed,
            "traj_dir": str(args.traj_dir),
            "logdir": args.logdir,
            "max_steps": args.max_steps,
            **run_metrics(args.traj_dir, reference, max_steps=args.max_steps),
            "world_model_metrics": world_model_metrics(args.logdir),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
