#!/usr/bin/env python
"""§14 Phase 8: Random Baseline.

Implements: References/SelfEx-WM_Notes.tex §14 -- the same Panda task under a_t ~ U([-1,1]^m), with
"the same environment, reset distribution, episode horizon, number of environment interactions, and
seeds where appropriate", to determine whether Plan2Explore's intrinsic exploration actually
produces additional coverage.

Runs the FULL p2e_dv3 pipeline on the GPU -- world model, ensembles, both actor-critics, replay,
TensorBoard -- and changes exactly one thing: the action sent to the environment is drawn uniformly
instead of from the exploration actor. That isolates the acting policy as the only variable, so the
comparison is about which actions were taken and not about which arm had a world model. It also
gives the baseline its own §13.3 losses and its own Rewards/intrinsic, so "what disagreement would
random actions have left unresolved?" becomes measurable rather than assumed.

§3 stays intact: sheeprl/algos/ is untouched on disk and verify_algorithm_untouched.sh still passes.
The substitution is a runtime patch of PlayerDV3.get_actions made by this standalone run script,
which §3's allowed-modification list explicitly permits.

Every Hydra override is passed straight through, so §14's "same everything" is enforced by using the
SAME command as the training run being compared against:

`root_dir` is the one override that must DIFFER from the training run. Everything else is shared, so
the default run_name (timestamp + exp + env + seed) is identical in form for both arms and the two
would interleave in one directory, told apart only by timestamp.

    python scripts/random_baseline.py \\
      exp=p2e_dv3_exploration env=menagerie_panda_reach env.num_envs=4 \\
      root_dir=random_baseline/MenageriePandaReach \\
      env.wrapper.trajectory_log=$REPO/results/runs/random_seed0/trajectories \\
      algo.cnn_keys.encoder=[] algo.cnn_keys.decoder=[] \\
      algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] \\
      algo.dense_units=512 algo.mlp_layers=2 \\
      algo.world_model.recurrent_model.recurrent_state_size=512 \\
      algo.world_model.transition_model.hidden_size=512 \\
      algo.world_model.representation_model.hidden_size=512 \\
      algo.total_steps=500000 algo.run_test=False \\
      metric.log_every=1000 checkpoint.every=10000 \\
      fabric.accelerator=gpu fabric.devices=1 seed=0

Analyse it with `metrics.py run`, the same path a training run takes -- it now has a TensorBoard
event file, so scripts/baseline_metrics.py is obsolete.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SHEEPRL_DIR = REPO_ROOT / "sheeprl"


def patch_uniform_random_actions() -> None:
    """Make the player return a_t ~ U([-1,1]^m) instead of the exploration actor's action (§14).

    Wraps rather than replaces. The original call still runs so every side effect of a normal step
    survives -- the RSSM advances, the actor's distributions are built -- and the returned tensors
    only supply the shapes, which keeps this independent of the action layout. ``self.actions`` is
    then set to what was actually executed, because the player feeds the previous action back into
    the recurrent model on the next step; leaving the actor's own sample there would advance the
    latent state along a trajectory the environment never took.

    The training loop assigns `real_actions = actions = player.get_actions(...)` and writes that
    same array to both `rb.add` and `envs.step`, so the replay buffer records the executed action
    and the world model learns the correct dynamics.
    """
    from sheeprl.algos.dreamer_v3.agent import PlayerDV3

    original = PlayerDV3.get_actions

    def get_actions(self, *args, **kwargs):
        actions = original(self, *args, **kwargs)
        if not getattr(self.actor, "is_continuous", True):
            raise RuntimeError(
                "random_baseline.py draws from U([-1,1]^m) and so assumes the continuous action "
                "space of §8.3; this run has a discrete actor."
            )
        # Uniform on the actor's own support: DreamerV3 squashes continuous actions into [-1, 1],
        # which is also the Box the environment advertises. Draws from torch's global RNG, which
        # SheepRL seeds from cfg.seed, so a seed reproduces the whole baseline.
        random_actions = tuple(torch.empty_like(a).uniform_(-1.0, 1.0) for a in actions)
        self.actions = torch.cat(random_actions, -1)
        return random_actions

    PlayerDV3.get_actions = get_actions


def main() -> int:
    overrides = sys.argv[1:]
    if not overrides:
        print(__doc__)
        return 2

    if not SHEEPRL_DIR.is_dir():
        print(f"SheepRL not cloned at {SHEEPRL_DIR} -- see SYNTAX.md step 1", file=sys.stderr)
        return 1

    # Hydra resolves configs relative to the entry point, and the training runs are launched from
    # sheeprl/, so the baseline is too. Anything else changes where logs land.
    os.chdir(SHEEPRL_DIR)
    sys.path.insert(0, str(SHEEPRL_DIR))

    patch_uniform_random_actions()

    print("§14 random baseline: p2e_dv3 pipeline with a_t ~ U([-1,1]^m)")
    print("  the acting policy is the ONLY difference from the training run.")
    print("  Pass the same overrides as the run being compared against, especially")
    print("  env.num_envs, algo.total_steps and the model-size flags.\n")

    from sheeprl.cli import run

    # run() is @hydra.main-decorated and parses sys.argv[1:] when called, so this hands Hydra the
    # overrides exactly as `python sheeprl.py <overrides>` would. It blocks until the run finishes.
    sys.argv = ["sheeprl.py", *overrides]
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
