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
  algo.total_steps=100000 \
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

## 6b. Resuming a killed run

`resume_from_checkpoint()` in `sheeprl/cli.py` loads the checkpoint's `config.yaml` and merges it
over your CLI, keeping only `root_dir`, `run_name`, `algo.total_steps` and `algo.learning_starts`.
Every other override you type is silently discarded, so the resume command is short:

```bash
# vacate the trajectory path first -- filenames carry the pid, so the resume cannot reopen the
# old files, and both sets landing in one directory corrupts coverage_curve (see §8)
mv "$REPO/results/runs/RUNNAME/trajectories" "$REPO/results/runs/RUNNAME/trajectories_seg0"

cd "$REPO/sheeprl"
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python sheeprl.py \
  exp=p2e_dv3_exploration \
  env=menagerie_panda_reach \
  checkpoint.resume_from=".../version_0/checkpoint/ckpt_160000_0.ckpt" \
  algo.total_steps=500000 \
  algo.learning_starts=0
```

| Flag | Why |
|---|---|
| `exp=`, `env=` | the resume raises `ValueError` unless `algo.name` and `env.id` match the checkpoint |
| `algo.total_steps` | popped from the old config, so omitting it falls back to the stock 5,000,000. **Absolute, not remaining** — set it to the original budget |
| `algo.learning_starts=0` | on resume the code does `learning_starts += start_iter`, so a nonzero value re-prefills the buffer with random actions mid-run. `buffer.checkpoint` defaults to `True`, so the buffer is already restored |

The checkpoint carries the world model, every optimizer state, the moments, the ratio and the replay
buffer, so the optimizer resumes with its Adam moments intact. `checkpoint.every=10000` over
`num_envs=4` is 25 episodes exactly, so a checkpoint always lands on an episode boundary and no
partial episode is cut. Do not delete the old run directory — with `buffer.memmap=True` the restored
buffer reads from its `memmap_buffer/`.

## 7. Random baseline (§14)

Same pipeline as §5, same budget, same seeds — the acting policy is the only difference. The script
takes no flags of its own: every argument is forwarded to Hydra, so pass the §5 command verbatim and
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
actor-critics all train on the randomly-collected data. That is what makes §14's comparison about
the acting policy rather than about which arm had a world model, and it gives the baseline its own
§13.3 losses and its own `Rewards/intrinsic`. Budget for it accordingly — it costs the same as the
run it is compared against.

## 8. Metrics (§13)

One command for both arms — the §14 baseline is a full training run, so it has TensorBoard events
and is read exactly like a Plan2Explore run:

```bash
RUN=$(ls -td sheeprl/logs/runs/p2e_dv3_exploration/MenageriePandaReach/*/ | head -1)version_0

python scripts/metrics.py run \
  --traj-dir results/runs/gateB_seed0/trajectories \
  --logdir "$RUN" \
  --seed 0 \
  --out results/summaries/gateB_seed0.json
```

For the baseline, point `--traj-dir` and `--logdir` at that seed's `random_baseline/...` run.
Comparing the two arms requires the same `--reference-samples` for both.

A resumed run leaves one trajectory directory per segment. Pass them oldest first, with the step
count from each checkpoint resumed at; the segments are cut at their seams and concatenated per
environment:

    --traj-dir <segment 0> --traj-dir <segment 1> --resumed-at <ckpt step> --num-envs 4

`--num-envs` is checked against the file count in each segment. That is what catches two segments
sharing a directory, which otherwise reads N files as N environments and corrupts `coverage_curve`
without erroring.

`--reference-samples` (default 200000) builds `C_workspace`'s denominator by sampling joint
configurations. **Use the same value for every run you compare**, or the coverage ratios differ for
reasons unrelated to the policy. Lower it to 20000 for a quick check only.

`--max-steps` is a one-off for a run whose budget was set wrong; omit it normally.

Live monitoring:

```bash
tensorboard --logdir logs/runs/p2e_dv3_exploration --port 6007
# from your laptop:  ssh -L 6007:localhost:6007 user@host   then http://localhost:6007
```

---

## Where the numbers come from

| §12.2 metric | Source |
|---|---|
| task success, task return | `episodes_*.f32` → `metrics.py` |
| end-effector workspace coverage | `ee_*.f32` → `metrics.py` |
| intrinsic reward | TensorBoard `Rewards/intrinsic` |
| ensemble disagreement | **the same tag** — in Plan2Explore the intrinsic reward *is* the variance across ensemble predictions, and `intrinsic_reward_multiplier` is 1. One number, not two |
| world-model losses | `Loss/world_model_loss`, `Loss/observation_loss`, `Loss/reward_loss`, `State/kl`, `Loss/ensemble_loss` |

Task success and return come from the **exploration** actor, which is what drives data collection.
They are not an evaluation of the task actor; that needs `sheeprl_eval.py` against a checkpoint.

## Trajectory file format

```
ee_<pid>_<id>.f32         3 float32 per environment step   x, y, z of the end effector
episodes_<pid>_<id>.f32   4 float32 per episode            per-env step count, return,
                                                           final distance, success
```

One pair per environment process. Read back with:

```python
ee = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
ep = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
```

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

**Three §13.2 metrics are not recorded** — actuator saturation fraction, joint-limit visitation, and
state-space coverage. The trajectory files hold only end-effector positions, so these cannot be
computed after a run. `C_workspace`, the metric §14's comparison rests on, is unaffected.