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

    # a run resumed from ckpt_160000_0.ckpt: one --traj-dir per segment, oldest first
    python scripts/metrics.py run --traj-dir .../trajectories_seg0 --traj-dir .../trajectories \
        --resumed-at 160000 --num-envs 4 --logdir ... --seed 0 --out results/summaries/gateC_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))

#: §13.2 voxel edge length for workspace coverage, metres. Chosen equal to §8.1's 5 cm success
#: tolerance so a "visited voxel" is the same spatial resolution the task is scored at.
VOXEL_SIZE = 0.05

#: §8.1's reachable box, the region targets are drawn from. Source of truth is ENVIRONMENT_SPEC.md;
#: these must match menagerie_panda_reach.yaml. Divided into VOXEL_SIZE cubes it is C_workspace's
#: denominator: 6 x 12 x 8 = 576 voxels.
BOX_LOW = (0.30, -0.30, 0.20)
BOX_HIGH = (0.60, 0.30, 0.60)

#: §13.2 actuator saturation: |a| above this counts as commanded to the edge of the actuator's
#: range. Applied to the NORMALIZED action, where §8.3's affine map makes one dimensionless
#: threshold mean the same thing for every actuator whatever its units. Frozen; see
#: ENVIRONMENT_SPEC.md.
SATURATION_THRESHOLD = 0.95

#: §13.2 joint-limit visitation: "at the limit" means within this fraction of the joint's OWN span.
#: A fraction rather than an absolute keeps one constant correct across hinge joints in radians and
#: slide joints in metres. MuJoCo's limit constraints are soft, so a joint pressed into its stop
#: sits near the limit rather than exactly on it.
JOINT_LIMIT_TOL_FRAC = 0.005

#: qpos[:7] are the arm joints. The two finger joints are driven to their extremes every episode by
#: design, so they are reported individually but kept out of the scalar summary.
N_ARM_JOINTS = 7


# --------------------------------------------------------------------- §13.2

def voxelize(points: np.ndarray, voxel_size: float = VOXEL_SIZE) -> set[tuple[int, int, int]]:
    """Map 3-D points onto integer voxel indices."""
    idx = np.floor(np.asarray(points) / voxel_size).astype(np.int64)
    return {tuple(v) for v in idx}


def box_voxels(
    low=BOX_LOW, high=BOX_HIGH, voxel_size: float = VOXEL_SIZE
) -> set[tuple[int, int, int]]:
    """C_workspace's denominator: every voxel of the §8.1 box (§13.2).

    Enumerated, not sampled, so the denominator is a fixed property of the task rather than of the
    run -- every policy is scored against the same 576 cubes, which is what makes §14's comparison
    a comparison of policies. The epsilons absorb float division landing a hair either side of a
    cube boundary.
    """
    lo = np.floor(np.asarray(low, dtype=float) / voxel_size + 1e-9).astype(np.int64)
    hi = np.ceil(np.asarray(high, dtype=float) / voxel_size - 1e-9).astype(np.int64)
    return {
        (i, j, k)
        for i in range(lo[0], hi[0])
        for j in range(lo[1], hi[1])
        for k in range(lo[2], hi[2])
    }


def workspace_coverage(ee_per_env: list[np.ndarray], box: set) -> dict:
    """C_workspace = #box voxels visited / #box voxels (§13.2).

    Takes one position array per environment and unions the voxels they visited. Grouping does not
    affect the result -- a set union is order-independent -- so this is identical to what a
    concatenated array would give, and correct for any num_envs.

    The arm reaches far outside the box, and those visits are not scored: the box is the region the
    task lives in. ``visited_outside_box`` reports how much exploration that discards, so a run
    ranging widely but ignoring the task region is visible rather than hidden.
    """
    visited: set = set()
    for positions in ee_per_env:
        visited |= voxelize(positions)

    inside = visited & box
    return {
        "visited_voxels": len(inside),
        "box_voxels": len(box),
        "C_workspace": len(inside) / max(len(box), 1),
        "visited_outside_box": len(visited) - len(inside),
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


def exploration_metrics(data: dict, box: set) -> dict:
    """§13.2, minus the intrinsic-reward and ensemble terms, which come from TensorBoard."""
    # A rollout drives a single environment, so its positions are one "per-env" array.
    out = workspace_coverage([data["ee_positions"]], box)
    out["actuator_saturation_fraction"] = float(np.mean(data["saturations"]))
    out["joint_limit_visitation"] = data["joint_limit_fraction"].round(6).tolist()
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
    traj_dirs: Path | list[Path],
    max_steps: int | None = None,
    resumed_at: list[int] | None = None,
    num_envs: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Read the streams written by MenageriePandaReach's trajectory_log.

    One position array per environment, not concatenated -- gluing the files end to end gives "all
    of env 0, then part of env 1", meaningless as a time axis. Episodes come back concatenated,
    (M, 4): total_steps, return, final_distance, success, with total_steps PER-ENVIRONMENT.

    A killed-and-resumed run writes fresh files, since filenames carry the pid. Pass one directory
    per segment chronologically; ``resumed_at`` gives each seam in TOTAL interactions, the number in
    the checkpoint filename. Segments are cut there and concatenated slot-wise -- the cross-segment
    pairing is arbitrary because environments step in lockstep, and rows past a seam had their
    gradient updates discarded by the resume. One directory holding every segment would read 8 files
    as 8 environments and wreck coverage_curve; ``num_envs`` catches that.

    ``max_steps`` is a ONE-OFF for Gate B seed 0, which overran to ~117,200 interactions and must be
    analysed at the same 100,000 as seeds 1 and 2. It caps total interactions across all envs.
    """
    if isinstance(traj_dirs, (str, Path)):
        traj_dirs = [traj_dirs]
    traj_dirs = [Path(d) for d in traj_dirs]
    resumed_at = list(resumed_at or [])
    if len(resumed_at) != len(traj_dirs) - 1:
        raise ValueError(
            f"got {len(traj_dirs)} segment director{'y' if len(traj_dirs) == 1 else 'ies'} but "
            f"{len(resumed_at)} seam(s); every segment except the last needs the total-interaction "
            "count it was resumed at."
        )

    segments = []
    for d in traj_dirs:
        seg_ee = [_read_rows(p, 3) for p in sorted(d.glob("ee_*.f32"))]
        seg_ep = [_read_rows(p, 4) for p in sorted(d.glob("episodes_*.f32"))]
        # diag_<ncols>_... -- the row width is model-dependent (nu + njnt), so it travels in the
        # filename rather than being assumed here. Absent for runs recorded before the stream
        # existed, which is why nothing downstream requires it.
        seg_diag = [
            _read_rows(p, int(p.name.split("_")[1])) for p in sorted(d.glob("diag_*_*.f32"))
        ]
        if not seg_ee:
            raise FileNotFoundError(f"no ee_*.f32 files in {d}")
        if seg_diag and len(seg_diag) != len(seg_ee):
            raise ValueError(
                f"{d} holds {len(seg_ee)} ee_*.f32 but {len(seg_diag)} diag_*.f32; the two streams "
                "are written by the same environments and must come in matching sets."
            )
        segments.append([seg_ee, seg_ep, seg_diag])

    n = len(segments[0][0])
    for d, (seg_ee, _, _) in zip(traj_dirs, segments):
        if len(seg_ee) != n:
            raise ValueError(
                f"segments hold different environment counts: {traj_dirs[0]} has {n} ee_*.f32 "
                f"files, {d} has {len(seg_ee)}. Each segment directory must hold exactly one "
                "launch's files."
            )
    if num_envs is not None and n != num_envs:
        raise ValueError(
            f"found {n} ee_*.f32 files per segment but --num-envs is {num_envs}. Two launches "
            "writing into one directory is the usual cause: give each its own directory and pass "
            "--traj-dir once per segment, in chronological order."
        )

    # Cut each segment back to its seam. The episode step column is that environment's own counter,
    # which restarts at zero every segment, so the filter is per-segment and the offset comes after.
    prev = 0
    for k, seam_total in enumerate(resumed_at):
        keep = seam_total // n - prev
        if keep <= 0:
            raise ValueError(f"seam {seam_total} is not after the previous seam ({prev * n})")
        seg_ee, seg_ep, seg_diag = segments[k]
        # Writes flush once per episode, so a killed process can leave a file a little short of its
        # own checkpoint. Harmless at this scale, but it is real data loss and should not be silent.
        for p, a in zip(sorted(traj_dirs[k].glob("ee_*.f32")), seg_ee):
            if len(a) < keep:
                print(f"  warning: {p.name} holds {len(a)} rows, {keep - len(a)} short of the seam")
        segments[k] = [
            [a[:keep] for a in seg_ee],
            [rows[rows[:, 0] <= keep] for rows in seg_ep],
            [a[:keep] for a in seg_diag],
        ]
        prev = seam_total // n

    ee_parts: list[list[np.ndarray]] = [[] for _ in range(n)]
    diag_parts: list[list[np.ndarray]] = [[] for _ in range(n)]
    episode_rows, offset = [], 0
    for seg_ee, seg_ep, seg_diag in segments:
        for i, a in enumerate(seg_ee):
            ee_parts[i].append(a)
        for i, a in enumerate(seg_diag):
            diag_parts[i].append(a)
        for rows in seg_ep:
            rows = rows.copy()
            rows[:, 0] += offset  # back onto one axis across the seam
            episode_rows.append(rows)
        offset += min(len(a) for a in seg_ee)

    ee = [np.concatenate(parts) for parts in ee_parts]
    diag = [np.concatenate(parts) for parts in diag_parts if parts]

    if max_steps is not None:
        per_env = max_steps // n
        ee = [a[:per_env] for a in ee]
        diag = [a[:per_env] for a in diag]
        episode_rows = [rows[rows[:, 0] <= per_env] for rows in episode_rows]

    episodes = np.concatenate(episode_rows) if episode_rows else np.empty((0, 4), np.float32)
    return ee, episodes, diag


def coverage_curve(ee_per_env: list[np.ndarray], box: set, n_points: int = 50) -> dict:
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
        values.append(len(visited & box) / max(len(box), 1))
        steps.append(int(m * n_envs))
    return {"steps": steps, "C_workspace": values}


def _marks(per_env_len: int, n_points: int) -> np.ndarray:
    """Prefix lengths at which a curve is sampled, shared with coverage_curve's x-axis."""
    if per_env_len == 0:
        return np.empty(0, dtype=int)
    return np.unique(
        np.linspace(max(per_env_len // n_points, 1), per_env_len, n_points).astype(int)
    )


def _prefix_curve(per_env: list[np.ndarray], marks: np.ndarray) -> list[float]:
    """Mean over environments and over the first m steps, for each mark m.

    A prefix mean rather than a windowed one, so the last point equals the headline metric exactly,
    matching coverage_curve. It damps late changes; read the headline against an early mark to see
    whether the policy drifted.
    """
    cums = [np.cumsum(a, axis=0) for a in per_env]
    return [float(np.mean([c[m - 1] / m for c in cums], axis=0)) for m in marks]


def saturation_metrics(actions_per_env: list[np.ndarray], n_points: int = 50) -> dict:
    """§13.2 actuator saturation, from NORMALIZED actions -- one array per environment, (T, nu).

    Clipping to [-1, 1] mirrors what _denormalize_action feeds the simulator, so this describes the
    action that was executed rather than the one the policy asked for. ``action_magnitude_mean`` is
    the threshold-free companion: it answers "how much of the action range is in use" without
    depending on SATURATION_THRESHOLD being the right cutoff.
    """
    mag = [np.abs(np.asarray(a, dtype=np.float64).reshape(len(a), -1)).clip(max=1.0)
           for a in actions_per_env]
    sat = [(m > SATURATION_THRESHOLD).mean(axis=-1) for m in mag]
    mean_mag = [m.mean(axis=-1) for m in mag]

    n_envs = len(mag)
    marks = _marks(min((len(m) for m in mag), default=0), n_points)
    return {
        "actuator_saturation_fraction": float(np.mean([s.mean() for s in sat])),
        "action_magnitude_mean": float(np.mean([m.mean() for m in mean_mag])),
        "saturation_threshold": SATURATION_THRESHOLD,
        "saturation_curve": {
            "steps": [int(m * n_envs) for m in marks],
            "fraction": _prefix_curve(sat, marks),
        },
    }


def joint_limit_metrics(
    qpos_per_env: list[np.ndarray],
    jnt_range: np.ndarray,
    jnt_limited: np.ndarray,
    n_points: int = 50,
) -> dict:
    """§13.2 joint-limit visitation -- one qpos array per environment, (T, njnt).

    Per joint, the fraction of steps within JOINT_LIMIT_TOL_FRAC of that joint's own span from
    either end of jnt_range. Unlimited joints report 0.0. The scalar summary covers the arm joints
    only; the fingers sit at their stops every episode by design and would dominate it.

    Valid only when every joint contributes one qpos entry -- true for a serial arm, false once §15
    adds a free-floating cube (7 qpos, 1 joint). The caller must check nq == njnt.
    """
    lo, hi = np.asarray(jnt_range)[:, 0], np.asarray(jnt_range)[:, 1]
    limited = np.asarray(jnt_limited).astype(bool)
    tol = JOINT_LIMIT_TOL_FRAC * (hi - lo)

    near = []
    for q in qpos_per_env:
        q = np.asarray(q, dtype=np.float64).reshape(len(q), -1)
        near.append(((np.abs(q - lo) <= tol) | (np.abs(q - hi) <= tol)) & limited)

    per_joint = np.mean([n.mean(axis=0) for n in near], axis=0)
    arm = [n[:, :N_ARM_JOINTS].mean(axis=-1) for n in near]

    n_envs = len(near)
    marks = _marks(min((len(n) for n in near), default=0), n_points)
    return {
        "joint_limit_visitation": [round(float(v), 6) for v in per_joint],
        "joint_limit_visitation_arm_mean": float(per_joint[:N_ARM_JOINTS].mean()),
        "joint_limit_tol_frac": JOINT_LIMIT_TOL_FRAC,
        "joint_limit_curve": {
            "steps": [int(m * n_envs) for m in marks],
            "arm_mean": _prefix_curve(arm, marks),
        },
    }


def _diagnostics_metrics(diag: list[np.ndarray], checkpoint: Path | None) -> dict:
    """§13.2 saturation and joint-limit visitation, from the stream or from a replay buffer.

    Returns the reason it found neither rather than raising, so a run recorded before the stream
    existed still yields its coverage and task metrics.
    """
    # Imported here: both paths need MuJoCo for jnt_range, and the buffer path needs torch and an
    # importable sheeprl. Neither is required to read the .f32 streams.
    if not diag and checkpoint is None:
        return {
            "diagnostics_source": None,
            "diagnostics_note": "no diag_*.f32 stream; pass --checkpoint to read the replay buffer",
        }

    from buffer_metrics import joint_limits_from_model

    jnt_range, jnt_limited, nu = joint_limits_from_model()

    if diag:
        # Row layout is [action(nu), qpos(...)]. Split on nu, not on the joint count: a task can log
        # fewer joints than the model has, and §15's free cube adds a joint that carries no limits
        # and is deliberately not logged. Splitting on the joint count would take an action column
        # as a joint position and report it silently.
        n_joints = diag[0].shape[1] - nu
        if n_joints <= 0 or n_joints > len(jnt_range):
            raise ValueError(
                f"diag rows are {diag[0].shape[1]} wide with nu={nu}, implying {n_joints} joint "
                f"columns against a model with {len(jnt_range)} joints."
            )
        actions = [d[:, :nu] for d in diag]
        qpos = [d[:, nu:] for d in diag]
        jnt_range, jnt_limited = jnt_range[:n_joints], jnt_limited[:n_joints]
        source, steps = "trajectory_log", int(sum(len(d) for d in diag))
    else:
        from buffer_metrics import read_buffer

        qpos, actions, lengths = read_buffer(Path(checkpoint))
        source, steps = "replay_buffer", int(sum(lengths))

    return {
        "diagnostics_source": source,
        # Records separately from steps_recorded: the buffer can hold a few thousand steps past the
        # checkpoint it was saved with, so the two need not agree.
        "diagnostics_steps": steps,
        **saturation_metrics(actions),
        **joint_limit_metrics(qpos, jnt_range, jnt_limited),
    }


def run_metrics(
    traj_dirs: Path | list[Path],
    box: set,
    max_steps: int | None = None,
    resumed_at: list[int] | None = None,
    num_envs: int | None = None,
    checkpoint: Path | None = None,
) -> dict:
    """§13.1 and §13.2 from a completed run's recorded trajectory.

    The acting policy during exploration is the EXPLORATION actor, so the task numbers describe what
    that actor achieved -- they are not an evaluation of the task actor, which needs sheeprl_eval.py
    against a checkpoint. Coverage is unambiguous: it is exactly the set of positions visited while
    exploring, which is what §13.2 asks for.

    Saturation and joint-limit visitation come from the diag_*.f32 stream when the run recorded one,
    and otherwise from ``checkpoint``'s replay buffer, which holds the same quantities for runs that
    predate it. ``traj_dirs``, ``resumed_at`` and ``max_steps`` are described in read_trajectories.
    """
    ee, ep, diag = read_trajectories(
        traj_dirs, max_steps=max_steps, resumed_at=resumed_at, num_envs=num_envs
    )
    exploration = workspace_coverage(ee, box)
    exploration["coverage_curve"] = coverage_curve(ee, box)
    exploration.update(_diagnostics_metrics(diag, checkpoint))

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

def world_model_metrics(logdirs: str | list[str], tags: list[str] | None = None) -> dict:
    """§13.3, read back from the TensorBoard events SheepRL already writes.

    Also covers the intrinsic-reward and ensemble-disagreement terms §13.2 asks for, since those are
    logged by the algorithm rather than observable from the environment.

    A resumed run logs to a fresh version_N directory. Pass one logdir per segment, chronologically:
    series are keyed by policy_step and a later segment overwrites an earlier one at a shared step,
    so the overlap a crash leaves behind resolves to the values whose gradient history continued.
    """
    from tensorboard.backend.event_processing import event_accumulator

    if isinstance(logdirs, (str, Path)):
        logdirs = [logdirs]

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
    # p2e_dv3_finetuning declares different keys -- no ensembles, and the task actor's losses are
    # unsuffixed. scripts/finetuning_metrics.py passes that list rather than duplicating this
    # function's merge-and-summarize logic.
    wanted = list(tags) if tags is not None else wanted

    series: dict[str, dict[int, float]] = {tag: {} for tag in wanted}
    available: set[str] = set()
    for d in logdirs:
        acc = event_accumulator.EventAccumulator(
            str(d), size_guidance={event_accumulator.SCALARS: 0}
        )
        acc.Reload()
        tags = set(acc.Tags().get("scalars", []))
        available |= tags
        for tag in wanted:
            if tag in tags:
                for e in acc.Scalars(tag):
                    series[tag][e.step] = e.value  # later segment wins

    out = {}
    for tag in wanted:
        if tag not in available:
            out[tag] = None
            continue
        steps = sorted(series[tag])
        values = np.array([series[tag][s] for s in steps])
        out[tag] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "first": float(values[0]),
            "last": float(values[-1]),
            "finite": bool(np.all(np.isfinite(values))),
            "n": int(len(values)),
            # The span is how you check the segments actually joined: it should reach the run's
            # budget, not stop at the step the first segment died on.
            "step_first": int(steps[0]),
            "step_last": int(steps[-1]),
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
    p_roll.add_argument("--out", type=Path, required=True)

    p_loss = sub.add_parser("losses", help="§13.3 from a run's TensorBoard events")
    p_loss.add_argument("--logdir", action="append", required=True, dest="logdirs", metavar="DIR")
    p_loss.add_argument("--out", type=Path, required=True)

    p_run = sub.add_parser("run", help="every §12.2 metric for one finished training run")
    p_run.add_argument(
        "--traj-dir",
        type=Path,
        action="append",
        required=True,
        dest="traj_dirs",
        metavar="DIR",
        help="env.wrapper.trajectory_log. Repeat once per segment, in chronological order, for a "
        "run that was resumed from a checkpoint",
    )
    p_run.add_argument(
        "--resumed-at",
        type=int,
        action="append",
        default=None,
        metavar="STEPS",
        help="total interactions at each resume, i.e. the number in the checkpoint filename "
        ,
    )
    p_run.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="env.num_envs of the run; checked against the file count per segment",
    )
    p_run.add_argument(
        "--logdir",
        action="append",
        required=True,
        dest="logdirs",
        metavar="DIR",
        help="the run's version_N directory. Repeat once per segment, in chronological order",
    )
    p_run.add_argument("--seed", type=int, required=True)
    p_run.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="a ckpt_*.ckpt whose replay buffer supplies §13.2's saturation and joint-limit "
        "visitation. Only for runs recorded before the diag_*.f32 stream existed; when the stream "
        "is present it is used instead and this is ignored",
    )
    p_run.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="ONE-OFF: cap total interactions, for a run whose budget was set wrong. Gate B seed 0 "
        "overran to ~117200 and needs --max-steps 100000 to match seeds 1 and 2. Omit otherwise.",
    )
    p_run.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    box = box_voxels()
    print(f"{len(box)} voxels in the §8.1 box at {VOXEL_SIZE} m: {BOX_LOW} to {BOX_HIGH}")

    if args.command == "rollout":
        # Imported here, not at module scope, so `run` and `losses` need no MuJoCo: the box is
        # arithmetic, and only a live rollout needs the simulator.
        from menagerie_panda import MenageriePandaReach

        env = MenageriePandaReach()
        print(f"rolling out {args.episodes} episodes with the {args.policy} policy...")
        data = rollout(env, random_policy, args.episodes, args.seed)
        result = {
            "policy": args.policy,
            "seed": args.seed,
            "task_metrics": task_metrics(data),
            "exploration_metrics": exploration_metrics(data, box),
        }
        env.close()
    elif args.command == "losses":
        result = {
            "logdirs": args.logdirs,
            "world_model_metrics": world_model_metrics(args.logdirs),
        }

    else:  # "run": every §12.2 metric for one finished training run
        result = {
            "seed": args.seed,
            "traj_dirs": [str(d) for d in args.traj_dirs],
            "resumed_at": args.resumed_at,
            "logdirs": args.logdirs,
            "max_steps": args.max_steps,
            **run_metrics(
                args.traj_dirs,
                box,
                max_steps=args.max_steps,
                resumed_at=args.resumed_at,
                num_envs=args.num_envs,
                checkpoint=args.checkpoint,
            ),
            "world_model_metrics": world_model_metrics(args.logdirs),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
