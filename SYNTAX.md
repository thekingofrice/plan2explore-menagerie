# SYNTAX.md — running everything, start to finish

> Implements: `References/SelfEx-WM_Notes.tex` **§18 Minimal Execution Order** and **§19 Definition of
> Success** — "an independent researcher can clone the standalone repository and ... reproduce the
> result from recorded commits and configs."

Every command needed, in order, with the flags explained. Assumes Linux with an NVIDIA GPU.
`ENVIRONMENT_SPEC.md` explains *what* the task is; this explains *how to run it*.

---

## 0. Conventions used throughout

```bash
REPO="$HOME/plan2explore-menagerie"
```

Three different step counts appear in this project. They are not interchangeable:

| Name | Counts | Where you see it |
|---|---|---|
| **policy step** | interactions summed over **all** environments | TensorBoard x-axis, `algo.total_steps` |
| **per-environment step** | interactions by **one** environment | column 1 of `episodes_*.f32` |
| **physics substep** | MuJoCo integration steps, 25 per environment step | never counted anywhere |

`algo.total_steps=100000` with `env.num_envs=4` means 25,000 steps *per environment*.

---

## 1. Clone and pin (§6)

```bash
git clone https://github.com/thekingofrice/plan2explore-menagerie.git
cd plan2explore-menagerie

git clone https://github.com/Eclectic-Sheep/sheeprl.git
git -C sheeprl checkout 33b636681fd8b5340b284f2528db8821ab8dcd0b

mkdir -p third_party
git clone https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie checkout da76818e269b82289eba39808e2fb91d679d6994
```

The SHAs are the pins recorded in `UPSTREAM.md`. Do not track a floating `main`: if upstream changes
`dreamer_v3` midway through a multi-seed run, early and late seeds are different algorithms and the
result is meaningless.

## 2. Environment (§7)

```bash
python3.11 -m venv .venv311          # SheepRL requires >=3.8,<3.12
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e "./sheeprl[mujoco,dev,test]" --no-deps
```

Two variables are load-bearing on a headless machine and are needed in **every** shell:

```bash
export MUJOCO_GL=egl                                  # offscreen GL; without it MuJoCo needs X11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduces allocator fragmentation
```

`setuptools<81` is pinned in the lock file and is required: newer setuptools removed
`pkg_resources`, which `lightning-utilities 0.9.0` still imports. Regenerate the lock with
`pip freeze --all`, never plain `pip freeze` — the latter omits setuptools and drops the pin.

## 3. Install the environment wrapper (§11.1 Option A)

```bash
bash "$REPO/scripts/install_env_wrapper.sh"
```

Symlinks two files into the SheepRL checkout:

```
sheeprl/sheeprl/envs/menagerie_panda.py                  -> menagerie_integration/menagerie_panda.py
sheeprl/sheeprl/configs/env/menagerie_panda_reach.yaml   -> menagerie_integration/menagerie_panda_reach.yaml
```

They live in `menagerie_integration/` because the SheepRL checkout is gitignored; symlinking gives
the Option A layout at runtime with one source of truth. Re-run this after editing either file.

## 4. Verify the environment (§10)

```bash
cd "$REPO"
pytest tests/ -v
```

Ten tests, one per row of §10's table. All must pass before training.

---

## 5. The run command, explained

Every flag, and why it is there:

```bash
cd "$REPO/sheeprl"
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python sheeprl.py \
  exp=p2e_dv3_exploration \
  env=menagerie_panda_reach \
  env.num_envs=4 \
  env.wrapper.trajectory_log="$REPO/results/runs/RUNNAME/trajectories" \
  algo.cnn_keys.encoder=[] algo.cnn_keys.decoder=[] \
  algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] \
  algo.dense_units=512 algo.mlp_layers=2 \
  algo.world_model.recurrent_model.recurrent_state_size=512 \
  algo.world_model.transition_model.hidden_size=512 \
  algo.world_model.representation_model.hidden_size=512 \
  algo.total_steps=500000 \
  algo.run_test=False \
  metric.log_every=1000 \
  checkpoint.every=10000 \
  fabric.accelerator=gpu fabric.devices=1 \
  seed=0
```

| Flag | Why |
|---|---|
| `exp=p2e_dv3_exploration` | Plan2Explore on DreamerV3 (§4.1) |
| `env=menagerie_panda_reach` | our env config, resolved via the Option A symlink |
| `env.num_envs=4` | explicit because it changes results, not just speed — see §"num_envs" below |
| `env.wrapper.trajectory_log=...` | **required for coverage.** Without it, end-effector positions are computed and discarded, and §13.2's `C_workspace` is unrecoverable |
| `algo.cnn_keys.*=[]`, `algo.mlp_keys.*=[state]` | the stock config expects pixels (`[rgb]`); our env emits only `{"state": ...}` (§8.2). Without these the run dies with "keys `['rgb']` are not a subset of the environment ... keys" |
| `algo.dense_units`, `mlp_layers`, `recurrent_state_size`, `*_model.hidden_size` | DreamerV3-**S** instead of SheepRL's default XL. XL is sized for 64x64 pixels; our observation is a 24-vector, and XL needs ~12 GB the shared GPU does not have |
| `algo.total_steps` | **set explicitly.** The default is 5,000,000, far beyond any budget in §12 |
| `algo.run_test=False` | skips a single evaluation episode at the end. That episode runs in its own environment and would otherwise write trajectory files that contaminate the training coverage number |
| `metric.log_every=1000` | must be a multiple of `num_envs` or SheepRL rounds it up |
| `checkpoint.every=10000` | the default 100000 writes ~one checkpoint per run; on a shared GPU an OOM at hour 9 would lose everything |
| `seed` | appears in the log directory name, so runs never collide |

Run it under `tmux` — the sessions outlive SSH disconnects:

```bash
tmux new -s gateB          # Ctrl-B then D to detach
tmux attach -t gateB       # to return
```

### Resuming from checkpoint
tmux new -s gateC_s1        # or: tmux attach -t gateC_s1

REPO="$HOME/plan2explore-menagerie"
source "$REPO/.venv311/bin/activate"
RUNNAME=2026-08-27_21-29-34_p2e_dv3_exploration_MenageriePandaReach_1
RUNDIR="$REPO/sheeprl/logs/runs/p2e_dv3_exploration/MenageriePandaReach/$RUNNAME"

# confirm 340000 really is the highest — resume 1 wrote into version_1, not version_0
ls -1 "$RUNDIR"/version_*/checkpoint/ckpt_*.* | sort -t_ -k2 -n | tail -3

CKPT=$(ls -1 "$RUNDIR"/version_*/checkpoint/ckpt_340000_0.* | head -1)
echo "CKPT=$CKPT"

TRAJ=$(grep -m1 trajectory_log "$RUNDIR/version_0/config.yaml" | awk '{print $2}')
echo "TRAJ=$TRAJ"

Vacate the path — whatever is at trajectories now is segment 1:

mv "$TRAJ" "${TRAJ%/}_seg1"
ls -d "$(dirname "$TRAJ")"/trajectories_seg*     # expect _seg0 and _seg1

If trajectories_seg0 is missing, stop — coverage before the first seam is gone and worth knowing about now rather than at analysis time.

Then:

cd "$REPO"
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/resume.py \
  exp=p2e_dv3_exploration \
  env=menagerie_panda_reach \
  checkpoint.resume_from="$CKPT" \
  root_dir=p2e_dv3_exploration/MenageriePandaReach \
  run_name="$RUNNAME" \
  algo.total_steps=500000 \
  algo.learning_starts=0 \
  2>&1 | tee -a "$REPO/gateC_seed1_resume.log"

Unchanged from last time: total_steps is absolute, and learning_starts=0 again because the += start_iter behaviour applies on every resume.

Analysis, when it finishes

Three trajectory directories, two seams, three logdirs:

python scripts/metrics.py run \
  --traj-dir "$(dirname "$TRAJ")/trajectories_seg0" \
  --traj-dir "$(dirname "$TRAJ")/trajectories_seg1" \
  --traj-dir "$TRAJ" \
  --resumed-at 220000 --resumed-at 340000 --num-envs 4 \
  --logdir "$RUNDIR/version_0" --logdir "$RUNDIR/version_1" --logdir "$RUNDIR/version_2" \
  --seed 1 --out results/summaries/gateC_seed1.json

## 6. The gates (§12)

```bash
# Gate A  (§12.1) — smoke, one seed, tiny budget
algo.total_steps=5000 metric.log_every=250

# Gate B  (§12.2) — pilot, 3 seeds x 1e5
for SEED in 0 1 2; do ... algo.total_steps=100000 seed=$SEED ; done

# Gate C  (§12.3) — baseline, 5 seeds x 5e5
for SEED in 0 1 2 3 4; do ... algo.total_steps=500000 seed=$SEED ; done
```

**§12.3 requires the exact budget to be frozen before decisive results are inspected.** Watching
losses and checking a run is alive is fine; looking at success rate or coverage and *then* choosing
how long to run is not — that makes the stopping point a tuned parameter. Gate B is the sanctioned
place to look at real numbers and use them to size Gate C.

Give each run its own `trajectory_log` directory. The files open in **append** mode with
PID-based names, so two runs sharing a directory do not overwrite — they silently merge into one
inflated coverage number, with no error.

## 6c. Panda Push (§15)

Same pipeline, different task. Step 3's install script places three extra links, one of them inside
the **Menagerie** clone — `panda_push.xml` must sit beside `scene.xml` for its `<include>` to
resolve. Re-run it after pulling:

```bash
bash "$REPO/scripts/install_env_wrapper.sh"
ls -l "$REPO/third_party/mujoco_menagerie/franka_emika_panda/panda_push.xml"
```

```bash
cd "$REPO/sheeprl"
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python sheeprl.py \
  exp=p2e_dv3_exploration \
  env=menagerie_panda_push \
  env.num_envs=4 \
  env.wrapper.trajectory_log="$REPO/results/runs/push_seed1/trajectories" \
  root_dir=p2e_dv3_exploration/MenageriePandaPush \
  algo.cnn_keys.encoder=[] algo.cnn_keys.decoder=[] \
  algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] \
  algo.dense_units=512 algo.mlp_layers=2 \
  algo.world_model.recurrent_model.recurrent_state_size=512 \
  algo.world_model.transition_model.hidden_size=512 \
  algo.world_model.representation_model.hidden_size=512 \
  algo.total_steps=500000 \
  algo.run_test=False \
  metric.log_every=1000 \
  checkpoint.every=10000 \
  fabric.accelerator=gpu fabric.devices=1 \
  seed=1
```

| Differs from §5 | Why |
|---|---|
| `env=menagerie_panda_push` | resolves through the Option A symlink to `MenageriePandaPush` |
| `root_dir=.../MenageriePandaPush` | keeps Push's logs out of Reach's tree; without it both tasks land under the same run directory |

Unchanged and load-bearing: `env.num_envs=4`, the DreamerV3-S sizing, and `algo.mlp_keys` — the
observation is still a single `state` key, now 30-dimensional rather than 24. `nu` is still 8, so the
action space and every `algo.*` flag carry over untouched.

Resume, metrics and evaluation are identical to Reach's §6b, §8 and §9 — nothing in those commands
names a task. `metrics.py run` reads Push's `diag_*.f32` with no extra flags, because the row stays
17 columns wide (`nu + 9`) and the reader splits on `nu`.

`beta` is **30.0**, frozen 2026-08-31 — not Reach's 10. Push's distances are half Reach's and enter
squared, so `alpha=10` would compress the reward into its top 60 %. `ENVIRONMENT_SPEC.md` §13 carries
the measurement.

## 6d. Finetuning — the few-shot regime (§13.1)

Exploration trains `actor_task` **purely in imagination**: it never acts, and §9's evaluation of an
exploration checkpoint is therefore Plan2Explore's *zero-shot* claim. `p2e_dv3_finetuning` is the
other half — it loads the exploration checkpoint, keeps the world model and both task actor-critics,
throws the ensembles away, and continues training on data the **task** actor collects. That is the
*few-shot* claim. Same task, same environment, different number.

Launch it through `scripts/resume.py`, **not** `sheeprl.py`: the exploration checkpoint carries a
pickled replay buffer, and torch ≥ 2.6 refuses to unpickle it (see *Known issues*). The script
accepts `checkpoint.exploration_ckpt_path` for exactly this caller.

```bash
cd "$REPO"
RUNNAME=<the exploration run directory name>
RUNDIR="$REPO/sheeprl/logs/runs/p2e_dv3_exploration/MenageriePandaPush/$RUNNAME"

# the LAST checkpoint of the exploration run — finetuning from an earlier one measures a different
# exploration budget, which is not the comparison §13.1 asks for
CKPT=$(ls -1 "$RUNDIR"/version_*/checkpoint/ckpt_500000_0.* | head -1)
echo "CKPT=$CKPT"

MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/resume.py \
  exp=p2e_dv3_finetuning \
  env=menagerie_panda_push \
  env.wrapper.trajectory_log="$REPO/results/runs/push_finetune_seed1/trajectories" \
  checkpoint.exploration_ckpt_path="$CKPT" \
  root_dir=p2e_dv3_finetuning/MenageriePandaPush \
  algo.total_steps=100000 \
  algo.run_test=False \
  metric.log_every=1000 \
  checkpoint.every=10000 \
  fabric.accelerator=gpu fabric.devices=1 \
  seed=1
```

Reach is the same command with `env=menagerie_panda_reach` and
`root_dir=p2e_dv3_finetuning/MenageriePandaReach`. Nothing else in it names a task.

| Flag | Why |
|---|---|
| `exp=p2e_dv3_finetuning` | defaults `algo.learning_starts=16384`, `algo.total_steps=1000000`, `buffer.load_from_exploration=False`, and declares its own aggregator keys — see the tag table below |
| `env=menagerie_panda_push` | **load-bearing.** Finetuning builds the environment from the *current* config, not the checkpoint's, so the task is chosen here. A mismatch (Push checkpoint, Reach env) fails on the 30-vs-24 observation shape rather than training something wrong |
| `env.wrapper.trajectory_log=...` | **a new directory.** The files open in append mode; pointed at the exploration run's directory the two phases merge into one inflated, meaningless coverage number with no error |
| `checkpoint.exploration_ckpt_path=...` | the exploration checkpoint to adapt from. `resume.py` refuses a path that is not a file — SheepRL treats an unset value as "start from scratch" and would silently train a world model from nothing |
| `root_dir=p2e_dv3_finetuning/MenageriePandaPush` | keeps the phase and the task out of each other's log trees |
| `algo.total_steps` | **set explicitly.** The default is 1,000,000 — twice the exploration budget, which is not a few-shot regime |
| `algo.run_test=False`, `metric.log_every`, `checkpoint.every`, `fabric.*` | as §5 |
| `seed` | the finetuning run's seed. Not inherited from the exploration run, and it appears in the log directory name |

**Do not pass these — they are overwritten from the exploration run's config and any value you type
is discarded:**

    algo.gamma  algo.lmbda  algo.horizon  algo.layer_norm  algo.dense_units  algo.mlp_layers
    algo.dense_act  algo.cnn_act  algo.unimix  algo.hafner_initialization
    algo.world_model  algo.actor  algo.critic
    algo.cnn_keys  algo.mlp_keys  env.clip_rewards  env.num_envs

That is why §5's DreamerV3-S sizing and `algo.mlp_keys.encoder=[state]` are absent above: the
finetuned model is required to be the model that was explored with, so restating the geometry could
only introduce drift. `env.num_envs` comes along with it — the command cannot change it, and
`num_envs` therefore stays at the exploration run's 4 for the buffer, the metrics and §13.

`algo.learning_starts=16384` is the upstream default and is left alone: the finetuning buffer starts
**empty** (`buffer.load_from_exploration=False`), so the first 16,384 policy steps are random
actions, not the task actor. At `num_envs=4` that is 4,096 steps per environment — about **20 Push
episodes per environment**, against 40 for Reach, because Push's episodes are 200 steps to Reach's
100. Metrics below split on that boundary.

### Metrics and evaluation

```bash
FT_RUNDIR="$REPO/sheeprl/logs/runs/p2e_dv3_finetuning/MenageriePandaPush/<finetuning run name>"

python scripts/finetuning_metrics.py \
  --traj-dir results/runs/push_finetune_seed1/trajectories \
  --logdir "$FT_RUNDIR/version_0" \
  --seed 1 --out results/summaries/push_finetune_seed1.json

python scripts/evaluate_task_actor.py \
  --checkpoint "$FT_RUNDIR/version_0/checkpoint/ckpt_100000_0.ckpt" \
  --episodes 5000 --seed 1 \
  --out results/summaries/push_finetune_seed1_eval.json
```

`finetuning_metrics.py`, not `metrics.py`. Two of §13's metrics stop existing here and it declines to
compute them rather than reporting a wrong one:

| §13 metric | Under finetuning |
|---|---|
| §13.2 `C_workspace` coverage | **not computed.** The acting policy is the task actor, so the trajectories are task-directed. Pooling that with Gate C's exploration coverage would compare two different questions |
| §13.2 intrinsic reward / ensemble disagreement | **gone.** The ensembles are neither restored nor built, so `Rewards/intrinsic_intrinsic` and `Loss/ensemble_loss` are absent from the event files |
| §13.3 world-model losses | present, same tags |
| §12.1 actor/critic losses | `Loss/policy_loss`, `Loss/value_loss` — **no `_task` suffix**, and `Grads/actor`, `Grads/critic` rather than `Grads/actor_task`, `Grads/critic_task`. Only one actor is trained now |
| §13.1 task success/return/distance | present and, unlike an exploration run, genuinely the task actor's — but *on-policy during training*, which is not `evaluate_task_actor.py`'s held-out score |

Pass `--learning-starts` to `finetuning_metrics.py` only if you overrode `algo.learning_starts`; its
default already matches the config's 16,384. It reports task metrics twice, over all episodes and
over the post-prefill ones, because the first ~20 episodes per environment are random actions.

`evaluate_task_actor.py` needs no new flag: it rebuilds the agent from the checkpoint's own
`config.yaml`, reads `state["world_model"]` and `state["actor_task"]` — key names a finetuning
checkpoint shares with an exploration one — and tags the result `few-shot` from `algo.name`. The same
command scores both phases; only the regime in the output JSON differs.

**Read Push's success rate against a 6 % floor, not 0 %** (`ENVIRONMENT_SPEC.md` §13) — six times
Reach's 1.05 %, because the cube starts near the centre of a 0.30 x 0.40 m target region. A
finetuning run reporting 8 % has moved almost nothing.

## 7. Random baseline (§14)

The same pipeline as §5 at the same budget and seeds — the acting policy is the only difference. The
script takes no flags of its own: every argument is forwarded to Hydra, so paste the §5 command and
change only the two lines marked below.

```bash
cd "$REPO"
for SEED in 0 1 2 3 4; do
  MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/random_baseline.py \
    exp=p2e_dv3_exploration \
    env=menagerie_panda_reach \
    env.num_envs=4 \
    root_dir=random_baseline/MenageriePandaReach \
    env.wrapper.trajectory_log="$REPO/results/runs/random_seed${SEED}/trajectories" \
    algo.cnn_keys.encoder=[] algo.cnn_keys.decoder=[] \
    algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] \
    algo.dense_units=512 algo.mlp_layers=2 \
    algo.world_model.recurrent_model.recurrent_state_size=512 \
    algo.world_model.transition_model.hidden_size=512 \
    algo.world_model.representation_model.hidden_size=512 \
    algo.total_steps=500000 \
    algo.run_test=False \
    metric.log_every=1000 \
    checkpoint.every=10000 \
    fabric.accelerator=gpu fabric.devices=1 \
    seed=$SEED
done
```

| Differs from §5 | Why |
|---|---|
| `scripts/random_baseline.py` instead of `sheeprl.py` | patches `PlayerDV3.get_actions` to return `a_t ~ U([-1,1]^m)`, then hands everything to SheepRL. `sheeprl/algos/` is untouched on disk, so §3 holds |
| `root_dir=random_baseline/...` | **required.** Every other key is shared, so the default run name is identical in form to a training run's and the two arms would interleave in one log directory, told apart only by timestamp |

This is a full GPU training run, not a CPU stepping loop: world model, ensembles and both
actor-critics all train on the randomly-collected data. That is what makes §14's comparison about the
acting policy rather than about which arm had a world model, and it gives the baseline its own §13.3
losses and its own `Rewards/intrinsic_intrinsic`. Budget for it as a second run per seed.

Resuming a baseline works exactly as §6b describes — it is an ordinary SheepRL run.

## 8. Metrics (§13)

Per training run:

```bash
RUN=$(ls -td sheeprl/logs/runs/p2e_dv3_exploration/MenageriePandaReach/*/ | head -1)version_0

python scripts/metrics.py run \
  --traj-dir results/runs/gateB_seed0/trajectories \
  --logdir "$RUN" \
  --seed 0 \
  --out results/summaries/gateB_seed0.json
```

The §14 baseline is a full training run, so it has TensorBoard events and is read with the same
command — point `--traj-dir` and `--logdir` at that seed's `random_baseline/...` run.

A resumed run leaves one trajectory directory and one `version_N` per segment. Pass them oldest
first, with the step count from each checkpoint resumed at:

    --traj-dir <segment 0> --traj-dir <segment 1> --resumed-at <ckpt step> --num-envs 4
    --logdir <version_0>   --logdir <version_1>

`--num-envs` is checked against the file count in each segment. That is what catches two segments
sharing a directory, which otherwise reads N files as N environments and corrupts `coverage_curve`
without erroring.

`--checkpoint` reads §13.2's actuator saturation and joint-limit visitation out of a checkpoint's
replay buffer. Only for runs recorded before the `diag_*.f32` stream existed; when the stream is
present it is used instead and the flag is ignored.

`--max-steps` is a one-off for a run whose budget was set wrong; omit it normally.

`C_workspace`'s denominator is the §8.1 box divided into `VOXEL_SIZE` cubes — enumerated, not
sampled, so it is identical for every run and needs no flag.

## 9. Task-actor evaluation (§13.1)

`metrics.py` reports the **exploration** actor's task numbers, since that actor collects the data.
§13.1's evaluation episodes need the **task** actor, which is trained in imagination and never acts
during training:

```bash
python scripts/evaluate_task_actor.py \
  --checkpoint "$RUN/checkpoint/ckpt_500000_0.ckpt" \
  --episodes 5000 \
  --seed 0 \
  --out results/summaries/gateC_seed0_eval.json
```

| Flag | Why |
|---|---|
| `--checkpoint` | repeatable; each is evaluated separately and reported as its own point |
| `--episodes` | total across all environments, **not** per environment. Every episode runs `max_episode_steps`, so 5,000 episodes is 500,000 environment steps — size it against the success rate's standard error, `sqrt(p(1-p)/n)` |
| `--seed` | the run's training seed. Recorded only; it does not seed the evaluation |
| `--eval-seed` | base seed for the evaluation episodes, independent of `--seed`, so every seed and both §14 arms are scored on the same target sequence |
| `--num-envs` | defaults to the run's own `env.num_envs`, which is what the player's recurrent state was allocated for |
| `--video-episodes` | record this many episodes of environment 0. Each faces a different target, so `2` gives two videos from one run. `0` (default) renders nothing |
| `--video-dir` | defaults to a `videos/` folder beside `--out` |

### Video

```bash
MUJOCO_GL=egl python scripts/evaluate_task_actor.py \
  --checkpoint "$RUNDIR/version_2/checkpoint/ckpt_500000_0.ckpt" \
  --episodes 5000 --video-episodes 2 \
  --seed 1 --out results/summaries/gateC_seed1_eval.json
```

Only environment 0 renders, and only for the first `--video-episodes` episodes; the other three run
unrendered, so every §13.1 number still comes from a full-speed rollout. Setting `render_mode` is
also what makes the wrapper attach a fixed camera and a sphere marking `g` — without them MuJoCo
falls back to its free camera and the goal has no geometry at all, so the footage shows an arm moving
toward nothing from an arbitrary angle (`menagerie_integration/render_scene.py`).

Files are named for the target they attempted, e.g. `episode0_target_0.348_0.141_0.245.mp4`. Seeds
are handed out in order from `--eval-seed`, so a rerun reproduces the same videos.

`MUJOCO_GL=egl` is required — rendering headless needs an offscreen GL context. mp4 needs imageio's
ffmpeg plugin; without it the script writes a GIF instead and says so.

**Nothing is rendered during training.** Both env configs ship `render_mode: null`, so a training run
never takes the MjSpec path and never pays for a frame. `capture_video` is a SheepRL `make_env`
setting and has no effect here — this script instantiates `env.wrapper` directly.

Actions are sampled rather than taken at the mode, matching the frozen implementation's own
zero-shot evaluation (`test(..., greedy=False)`).

The robot, the task and every frozen constant are instantiated from the checkpoint's `config.yaml`,
so no flag names a robot and §15/§16 need no change here. This requires step 3's symlink to be
installed, because `env.wrapper._target_` resolves through `sheeprl.envs`.

Live monitoring:

```bash
tensorboard --logdir logs/runs/p2e_dv3_exploration --port 6007
# from your laptop:  ssh -L 6007:localhost:6007 user@host   then http://localhost:6007
```

---

## Where the numbers come from

| §13 metric | Source |
|---|---|
| task success / return / distance, **exploration** actor | `episodes_*.f32` → `metrics.py run` |
| task success / return / distance, **task** actor (§13.1) | checkpoint → `evaluate_task_actor.py` |
| end-effector workspace coverage + curve | `ee_*.f32` → `metrics.py run` |
| actuator saturation, joint-limit visitation | `diag_*.f32` → `metrics.py run`; or `--checkpoint` for runs predating that stream |
| intrinsic reward | TensorBoard `Rewards/intrinsic_intrinsic` |
| ensemble disagreement | **the same tag** — in Plan2Explore the intrinsic reward *is* the variance across ensemble predictions, and `intrinsic_reward_multiplier` is 1. One number, not two |
| world-model losses | `Loss/world_model_loss`, `Loss/observation_loss`, `Loss/reward_loss`, `State/kl`, `Loss/ensemble_loss` |

The tag is `Rewards/intrinsic_intrinsic`, not `Rewards/intrinsic`: the algorithm copies the generic
aggregator key into one per exploration critic (`f"Rewards/intrinsic_{k}"`) and deletes the generic
one, and the intrinsic critic's key is itself `intrinsic`. Not a typo.

`metrics.py`'s task numbers come from the **exploration** actor, which drives data collection. They
are not an evaluation of the task actor; that is §9's job.

## Trajectory file format

```
ee_<pid>_<id>.f32            3 float32 per environment step   x, y, z of the end effector
episodes_<pid>_<id>.f32      4 float32 per episode            per-env step count, return,
                                                              final distance, success
diag_<ncols>_<pid>_<id>.f32  ncols float32 per step           normalized action (nu),
                                                              then qpos (njnt)
```

One set per environment process. Read back with:

```python
ee   = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
ep   = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
diag = np.fromfile(path, dtype=np.float32).reshape(-1, ncols)   # ncols is in the filename
```

`diag` stores the raw action and joint positions rather than derived flags, so §13.2's
`SATURATION_THRESHOLD` and `JOINT_LIMIT_TOL_FRAC` can be changed and reapplied without re-running.
Its width is `nu + njnt`, which differs per robot, so it travels in the filename.

Raw streams rather than `.npy` because they are appended to across a multi-hour run and must survive
an abrupt kill — there is no header to finalize. Files are created on the first *step*, so an
environment that SheepRL builds only to read its spaces leaves nothing behind.

## On `num_envs`

It is a property of the experiment, not a speed dial. Keep it fixed across every run you compare,
including the §14 baseline.

- **Does not change** how much the model learns: `replay_ratio: 1` fixes gradient updates per
  environment step, so `algo.total_steps` determines total training either way.
- **Does change** what the buffer holds. Environments step in lockstep, one step each per iteration,
  each with its own physics state, its own RSSM recurrent state, and its own sampled action. Four
  environments explore four different regions at once, which affects coverage.
- **Barely changes speed.** Collection is ~7% of runtime at the measured ~2.9 steps/s; the other 93%
  is gradient updates. Going 4 → 1 costs roughly 18%.

The replay buffer divides `buffer.size` by `num_envs`, so total capacity stays at 1,000,000
regardless. Sequences are always drawn from within a single environment.

## Known issues

**Intermittent SIGFPE in `mj_step`.** One `pytest tests/` run died with `Floating point exception
(core dumped)` and was never reproduced. A native crash gives no Python traceback. If a long run dies
silently, check this before suspecting SheepRL. See `ENVIRONMENT_SPEC.md` §10.

**Runs recorded before the `diag_*.f32` stream** have no actuator saturation or joint-limit
visitation in their trajectory files. Recover both from the replay buffer with `metrics.py run
--checkpoint <ckpt>`; that works only while `full` is False on the buffer, i.e. while
`algo.total_steps < buffer.size`.

**`torch.load` needs `weights_only=False` to resume.** torch 2.6 flipped the default, and the
checkpoint holds a pickled `EnvIndependentReplayBuffer`. `scripts/resume.py` patches it; launching a
resume through `sheeprl.py` directly fails with `UnpicklingError`.