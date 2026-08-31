#!/usr/bin/env python
"""§13.1 task metrics by evaluating the zero-shot task actor from a checkpoint.

Implements: References/SelfEx-WM_Notes.tex §13.1 -- "SuccessRate = #successful evaluation episodes /
#evaluation episodes", mean and median task return, and final target distance.

Every task number scripts/metrics.py reports comes from the EXPLORATION actor, which drives data
collection. This evaluates `actor_task`, trained purely in imagination on the world model's reward
predictions and never used to act during exploration -- Plan2Explore's zero-shot claim.

Upstream's own path is not usable for §13.1: sheeprl's `test()` runs exactly one episode and logs
only cumulative reward, giving no success rate, no median and no final distance. §3 permits
standalone evaluation scripts, so this is one. sheeprl/algos/ is not modified.

Repeat --checkpoint to get zero-shot performance as a function of exploration budget; checkpoints
exist every `checkpoint.every` steps, so the curve is nearly free.

Usage:
    python scripts/evaluate_task_actor.py \\
        --checkpoint .../version_1/checkpoint/ckpt_500000_0.ckpt \\
        --episodes 100 --seed 1 --out results/summaries/gateC_seed1_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))
sys.path.insert(0, str(REPO_ROOT / "sheeprl"))

#: Evaluation episodes are seeded from here, independent of the run's training seed. Fixed rather
#: than offset per seed so every seed and both §14 arms are scored on the SAME target sequence: the
#: comparison becomes paired, and seed-to-seed spread stops carrying task-difficulty variance. The
#: value is far outside the range training seeds reach, so no evaluation episode was trained on.
DEFAULT_EVAL_SEED = 100



def _load(path: Path):
    """torch.load with the pre-2.6 default; see scripts/resume.py and UPSTREAM.md."""
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        return original(str(path), map_location="cpu", weights_only=False)
    finally:
        torch.load = original


def load_run_config(ckpt_path: Path):
    """The resolved config the run trained with, beside the checkpoint directory.

    Returned twice over: the OmegaConf node for hydra.utils.instantiate, and the dotdict sheeprl's
    own functions expect.
    """
    from omegaconf import OmegaConf

    from sheeprl.utils.utils import dotdict

    raw = OmegaConf.load(ckpt_path.parent.parent / "config.yaml")
    return raw, dotdict(OmegaConf.to_container(raw, resolve=True))


def write_video(directory: Path, index: int, frames: list, fps: int, target) -> Path:
    """Write one episode's frames, naming the file after the target it was attempting.

    mp4 needs imageio's ffmpeg plugin; GIF needs nothing extra. Falling back rather than failing
    keeps a missing optional dependency from costing the whole evaluation.
    """
    import imageio.v2 as imageio

    directory.mkdir(parents=True, exist_ok=True)
    stem = f"episode{index}_target_{target[0]:.3f}_{target[1]:.3f}_{target[2]:.3f}"
    path = directory / f"{stem}.mp4"
    try:
        imageio.mimwrite(path, frames, fps=fps)
    except Exception as exc:  # noqa: BLE001 - any writer/plugin failure should degrade, not abort
        path = directory / f"{stem}.gif"
        imageio.mimwrite(path, frames, duration=1.0 / fps)
        print(f"  mp4 writer unavailable ({type(exc).__name__}); wrote {path.name}")
    return path


def make_envs(raw_cfg, n: int, record: bool = False) -> list:
    """Instantiate n copies of the environment the run actually used.

    No robot or task is named here. `env.wrapper._target_` and every frozen constant come from the
    run's own config, so this evaluates whichever §16 arm or §15 task the checkpoint was trained on,
    and the evaluation environment cannot disagree with training about alpha, success_tol or the
    target box, because it never restates them.

    trajectory_log is the one override: an evaluation environment writing into the run's coverage
    directory is exactly what algo.run_test=False exists to prevent (SYNTAX.md §5).
    """
    import hydra

    # Only environment 0 renders, and only when video was asked for. Rendering costs a frame per
    # step through EGL, so the other three stay off and every §13.1 number comes from the same
    # rollout at full speed. Setting render_mode also makes the wrapper attach the camera and the
    # goal marker (render_scene.py), which is what makes the frames readable.
    return [
        hydra.utils.instantiate(
            raw_cfg.env.wrapper,
            trajectory_log=None,
            render_mode="rgb_array" if (record and i == 0) else None,
            _convert_="all",
        )
        for i in range(n)
    ]


def build_task_player(fabric, cfg, ckpt_path: Path, env):
    """Reconstruct the agent and return a player holding actor_task.

    The run's own config supplies the model geometry, so the DreamerV3-S sizing is never repeated
    here and cannot drift from what was trained.
    """
    from sheeprl.algos.p2e_dv3.agent import build_agent

    state = _load(ckpt_path)
    *_, player = build_agent(
        fabric,
        tuple(env.action_space.shape),  # (nu,); §8.3 makes every robot's action space continuous
        True,
        cfg,
        env.observation_space,
        world_model_state=state["world_model"],
        actor_task_state=state["actor_task"],
    )
    player.actor_type = "task"
    return player


@torch.no_grad()
def rollout_task_actor(
    player,
    envs,
    fabric,
    cfg,
    episodes: int,
    seed: int,
    video_episodes: int = 0,
    video_dir: Path | None = None,
) -> dict:
    """Run `episodes` episodes under actor_task across `envs`, stepped in lockstep.

    Drives the environments directly rather than through a vector wrapper: the player's recurrent
    state is (1, num_envs, ...) so the environments must advance together, and stepping them here
    keeps each episode's final info dict -- which carries §13.1's distance and success -- instead of
    losing it to an autoreset.

    Actions are sampled, not taken at the mode: that is what the frozen implementation's own
    zero-shot evaluation does (`test(..., greedy=False)`), and action selection is the algorithm's
    behaviour, not the environment's (§3).
    """
    from sheeprl.algos.dreamer_v3.utils import prepare_obs

    n = len(envs)
    returns, distances, successes = [], [], []
    ep_return = np.zeros(n)

    # Only environment 0 is recorded, and only its first `video_episodes` episodes. Because seeds
    # are handed out in order, those are the lowest eval seeds -- so the same two videos come back
    # on a rerun, and each shows a different target.
    frames: list = []
    videos_written = 0
    fps = int(envs[0].metadata.get("render_fps", 20))

    # One distinct seed per episode, handed out in order, so a rerun reproduces the whole evaluation.
    next_seed = seed
    obs_list = []
    for e in envs:
        o, _ = e.reset(seed=next_seed)
        next_seed += 1
        obs_list.append(o)
    player.init_states()

    while len(returns) < episodes:
        batched = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        torch_obs = prepare_obs(fabric, batched, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=n)
        actions = player.get_actions(torch_obs, greedy=False)
        act = torch.cat(actions, -1).cpu().numpy().reshape(n, -1)

        finished = []
        for i, e in enumerate(envs):
            o, reward, terminated, truncated, info = e.step(act[i])
            ep_return[i] += float(reward)
            obs_list[i] = o

            recording = i == 0 and videos_written < video_episodes
            if recording:
                frames.append(e.render())

            if terminated or truncated:
                if recording and frames:
                    path = write_video(
                        video_dir, videos_written, frames, fps, info["target_position"]
                    )
                    print(f"  video: {path.name}  ({len(frames)} frames, "
                          f"success={info['success']}, d={info['distance']:.4f})")
                    videos_written += 1
                    frames = []

                returns.append(ep_return[i])
                distances.append(float(info["distance"]))
                successes.append(bool(info["success"]))
                ep_return[i] = 0.0
                obs_list[i], _ = e.reset(seed=next_seed)
                next_seed += 1
                finished.append(i)

        # Same treatment the training loop gives a finished environment: clear that environment's
        # latent so the next episode does not inherit the last one's recurrent state.
        if finished:
            player.init_states(reset_envs=finished)

    # Environments finish together, so the last iteration can push past the requested count.
    returns = np.array(returns[:episodes])
    distances = np.array(distances[:episodes])
    successes = np.array(successes[:episodes], dtype=float)

    # One point per episode. Unlike coverage_curve, where each mark costs a set union over a long
    # prefix, every running mean here falls out of one cumsum -- so downsampling would save nothing
    # and only discard resolution. The curve shows when the estimate stops moving.
    k = np.arange(1, len(returns) + 1)
    return {
        # §13.1: the end state must be within §8.1's tolerance, not merely have been passed through.
        "success_rate": float(successes.mean()),
        "return_mean": float(returns.mean()),
        "return_median": float(np.median(returns)),
        "final_distance_mean": float(distances.mean()),
        "episodes": int(len(returns)),
        "evaluation_curve": {
            "episodes": k.tolist(),
            "success_rate": (np.cumsum(successes) / k).tolist(),
            "return_mean": (np.cumsum(returns) / k).tolist(),
            "final_distance_mean": (np.cumsum(distances) / k).tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        dest="checkpoints",
        metavar="CKPT",
        help="repeat to evaluate several, giving zero-shot performance vs exploration budget",
    )
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="defaults to the run's own env.num_envs, which is what the player's recurrent state "
        "was allocated for. Override only to deviate deliberately",
    )
    parser.add_argument("--seed", type=int, required=True, help="the run's training seed, recorded")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="base seed for the evaluation episodes. Like 1, 2, 3, 4, 5, ...",
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=0,
        help="record this many episodes of environment 0. Each faces a different target, so 2 gives "
        "two videos from one run. Rendering is off entirely at 0",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="where to write them; defaults to a 'videos' folder beside --out",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    video_dir = args.video_dir or args.out.parent / "videos"

    from lightning import Fabric

    # The first checkpoint's config decides the robot, the task and the environment count; every
    # checkpoint passed belongs to the same run, so they agree by construction.
    raw_cfg, cfg = load_run_config(args.checkpoints[0])
    n_envs = args.num_envs if args.num_envs is not None else int(cfg.env.num_envs)
    envs = make_envs(raw_cfg, n_envs, record=args.video_episodes > 0)
    print(f"{cfg.env.wrapper['_target_']} x{n_envs}")
    if args.video_episodes:
        print(f"recording the first {args.video_episodes} episode(s) of env 0 -> {video_dir}")

    fabric = Fabric(accelerator=cfg.fabric.get("accelerator", "cpu"), devices=1)
    fabric.launch()

    points = []
    for ckpt in args.checkpoints:
        step = int(m.group(1)) if (m := re.search(r"ckpt_(\d+)_", ckpt.name)) else None
        print(f"evaluating {ckpt.name} ({args.episodes} episodes)...")
        player = build_task_player(fabric, cfg, ckpt, envs[0])
        result = rollout_task_actor(
            player, envs, fabric, cfg, args.episodes, args.eval_seed,
            video_episodes=args.video_episodes, video_dir=video_dir,
        )
        points.append({"policy_step": step, "checkpoint": str(ckpt), **result})
        print(f"  success_rate={result['success_rate']:.3f} "
              f"return_mean={result['return_mean']:.2f} "
              f"final_distance_mean={result['final_distance_mean']:.4f}")

    for e in envs:
        e.close()

    points.sort(key=lambda p: (p["policy_step"] is None, p["policy_step"]))
    out = {
        "seed": args.seed,
        "actor": "task",
        "regime": "zero-shot",
        "eval_seed": args.eval_seed,
        "task_metrics_task_actor": points[-1],
        "zero_shot_curve": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
