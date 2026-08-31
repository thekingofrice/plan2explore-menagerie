"""Menagerie Panda Reach - a standalone Gymnasium environment.

Implements: References/SelfEx-WM_Notes.tex
    §8    Phase 2: First Menagerie Task - Panda Reach (§8.1 task, §8.2 observation, §8.3 action,
          §8.4 control interval)
    §9    Phase 3: Implement a Gymnasium-Compatible Environment
    §11.1 Phase 5 Option A - installed to sheeprl/sheeprl/envs/menagerie_panda.py
    §13   trajectory logging, so coverage and task metrics survive a run

Canonical copy lives in menagerie_integration/ because the SheepRL checkout is a gitignored pinned
clone; scripts/install_env_wrapper.sh symlinks it into place.

Every frozen constant has its source of truth in ENVIRONMENT_SPEC.md and arrives as a constructor
argument from sheeprl/configs/env/menagerie_panda_reach.yaml. The defaults below match the spec so
the class is usable standalone in tests.

Never imports from sheeprl. That is what keeps §3 intact: the environment adapts to the baseline,
not the reverse.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import sys

# Path.resolve() follows the §11.1 symlink back to menagerie_integration/, so this import works both
# from here and from sheeprl/envs/ where this file is linked -- the same mechanism DEFAULT_XML uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_scene import (  # noqa: E402
    CAMERA_NAME,
    load_model,
    move_target_site,
    target_site_id,
)

# --- §8 frozen constants (source of truth: ENVIRONMENT_SPEC.md) -------------

#: Resolved from this file's location, not the working directory: Path.resolve() follows the §11.1
#: symlink back here, and SheepRL is launched from sheeprl/ where a cwd-relative path would not work.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Menagerie's own scene.xml, unmodified -- see ENVIRONMENT_SPEC.md "Why no task MJCF".
DEFAULT_XML = str(
    REPO_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
)

DEFAULT_CONTROL_DT = 0.05
DEFAULT_MAX_EPISODE_STEPS = 100
DEFAULT_ALPHA = 10.0
DEFAULT_SUCCESS_TOL = 0.05
DEFAULT_TARGET_BOX_LOW = (0.30, -0.30, 0.20)
DEFAULT_TARGET_BOX_HIGH = (0.60, 0.30, 0.60)
DEFAULT_JOINT_JITTER = 0.05

#: Menagerie's Panda declares no sites, so p_ee is derived from the hand body's pose. 0.1034 m is the
#: Franka TCP offset along the hand frame's +z, between the fingers.
EE_BODY = "hand"
EE_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float64)

N_ARM_JOINTS = 7  # qpos[:7]; qpos[7:9] are the fingers

#: All hinge/slide joints. Fixes the width of the §13.2 diagnostics row so scripts/metrics.py can
#: read the stream back, the same contract ee_*.f32 has at 3 columns.
N_JOINTS = 9

#: §13.2 actuator saturation: |a| above this counts as commanded to the edge of the actuator's
#: ctrlrange. Applied to the NORMALIZED action, where §8.3's affine map makes one dimensionless
#: threshold mean the same thing for every actuator regardless of its units.
SATURATION_THRESHOLD = 0.95

#: §13.2 joint-limit visitation: "at the limit" means within this fraction of the joint's OWN span.
#: A fraction rather than an absolute keeps one constant correct across hinge joints in radians and
#: slide joints in metres, and across §16's other arms. MuJoCo's limit constraints are soft by
#: default, so a joint pressed into its stop sits near the limit rather than exactly on it.
JOINT_LIMIT_TOL_FRAC = 0.005


class MenageriePandaReach(gym.Env):
    """Free-space reaching with the Franka Emika Panda (§8).

    Observation is a single flat vector under the key ``"state"`` (§8.2). Actions are normalized to
    ``[-1, 1]`` and mapped onto the model's native actuator ranges (§8.3). One ``step`` advances the
    simulation by ``control_dt`` seconds via an integer number of physics substeps (§8.4).

    Set ``trajectory_log`` to a directory to record end-effector positions and per-episode outcomes
    for §13's metrics.
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 20,  # 1 / control_dt
    }

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        render_mode: str | None = None,
        control_dt: float = DEFAULT_CONTROL_DT,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        alpha: float = DEFAULT_ALPHA,
        success_tol: float = DEFAULT_SUCCESS_TOL,
        target_box_low: tuple[float, float, float] = DEFAULT_TARGET_BOX_LOW,
        target_box_high: tuple[float, float, float] = DEFAULT_TARGET_BOX_HIGH,
        joint_jitter: float = DEFAULT_JOINT_JITTER,
        render_height: int = 480,
        render_width: int = 640,
        trajectory_log: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Build the model and derive every space and constant from it.

        Raises if the control interval is not an integer number of physics substeps (§8.4) or if any
        actuator is unlimited (§8.3) -- either would make the advertised contract untrue rather than
        merely approximate.
        """
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        # A fixed camera and a visible goal marker are attached only when rendering is on, so a
        # training run never takes the MjSpec path. Both are cosmetic -- see render_scene.py.
        # Reach has no table, so the base stays on the floor and only render assets are ever added.
        self.model = load_model(xml_path, render=render_mode is not None)
        self._target_site = target_site_id(self.model)
        self.data = mujoco.MjData(self.model)

        # §8.4 control interval
        self.control_dt = float(control_dt)
        self.sim_dt = float(self.model.opt.timestep)
        substeps = self.control_dt / self.sim_dt
        if abs(substeps - round(substeps)) > 1e-9:
            raise ValueError(
                f"control_dt={self.control_dt} is not an integer multiple of the model timestep "
                f"{self.sim_dt} (got {substeps} substeps)"
            )
        self.n_substeps = int(round(substeps))

        # §8.3 action space. Bounds come from the model so a Menagerie bump cannot silently change
        # what a normalized action means.
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        if not np.all(self.model.actuator_ctrllimited):
            raise ValueError(
                "every actuator must be ctrl-limited for the §8.3 affine map to be well defined; "
                f"got ctrllimited={self.model.actuator_ctrllimited.tolist()}"
            )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # §8.2 observation space
        obs_dim = self.model.nq + self.model.nv + 3 + 3  # q, qdot, p_ee, g
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
            }
        )

        # §8.1 task parameters
        self.alpha = float(alpha)
        self.success_tol = float(success_tol)
        self.target_box_low = np.asarray(target_box_low, dtype=np.float64)
        self.target_box_high = np.asarray(target_box_high, dtype=np.float64)
        self.joint_jitter = float(joint_jitter)
        self.max_episode_steps = int(max_episode_steps)

        self._ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        if self._ee_body_id < 0:
            raise ValueError(f"model has no body named {EE_BODY!r}")
        self._home_key_id = 0 if self.model.nkey > 0 else -1

        self.steps = 0
        self.target = np.zeros(3, dtype=np.float64)
        self._episode_return = 0.0
        self._total_steps = 0

        # Files are opened lazily on the first step, not here -- see _open_trajectory_files.
        self._traj_dir = Path(trajectory_log) if trajectory_log else None
        self._ee_file = None
        self._episode_file = None
        self._diag_file = None

        # §13.2's joint-limit visitation indexes jnt_range alongside qpos, which is only valid when
        # every joint contributes exactly one qpos entry. True for a serial arm; false once §15 adds
        # a free-floating cube (7 qpos, 1 joint). Fail at construction rather than mis-pair silently.
        if self.model.nq != self.model.njnt:
            raise ValueError(
                f"nq={self.model.nq} != njnt={self.model.njnt}: some joint does not contribute one "
                "qpos entry, so the §13.2 diagnostics row cannot be read back per joint."
            )

        # Renderer is lazy: it needs a GL context, and the env must import and step without one.
        self.render_height = int(render_height)
        self.render_width = int(render_width)
        self._renderer: mujoco.Renderer | None = None

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------- §13

    def _open_trajectory_files(self) -> None:
        """Open the two raw float32 append streams that §13's metrics are computed from.

        SheepRL does not log info-dict contents, so end-effector positions and episode outcomes are
        otherwise computed every step and discarded. §13.2's C_workspace needs the positions visited
        *while exploring*, which cannot be reconstructed from a checkpoint afterwards.

            ee_<pid>_<id>.f32           3 floats/step      x, y, z
            episodes_<pid>_<id>.f32     4 floats/episode   total_steps, return, final_distance, success
            diag_<ncols>_<pid>_<id>.f32 ncols floats/step  normalized action (nu), then qpos (njnt)

        The diagnostics row carries the raw quantities, not the derived flags, so §13.2's saturation
        threshold and joint-limit tolerance can be changed without re-running. Its width is in the
        filename because it is model-dependent -- §16's other arms have a different nu and njnt.

        Read back with np.fromfile(...).reshape(-1, 3) and (-1, 4). Raw streams rather than .npy
        because both are appended to across a multi-hour run and must survive an abrupt kill: there
        is no header to finalize. Filenames are per-instance because a vectorized run has several
        environments live in separate processes.

        Called on the first step rather than from __init__. SheepRL's vectorized wrapper constructs a
        throwaway environment purely to read the observation and action spaces, and never steps it;
        opening files eagerly left a zero-length pair behind that the reader then tried to parse.
        """
        self._traj_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{os.getpid()}_{id(self):x}"
        self._ee_file = open(self._traj_dir / f"ee_{tag}.f32", "ab")
        self._episode_file = open(self._traj_dir / f"episodes_{tag}.f32", "ab")
        ncols = self.model.nu + self.model.njnt
        self._diag_file = open(self._traj_dir / f"diag_{ncols}_{tag}.f32", "ab")

    def _record_episode(self) -> None:
        """Append the finished episode's outcome.

        Called from reset rather than on truncation: SheepRL's vectorized wrapper auto-resets, so
        that is the only point guaranteed to see every episode's final state. Also called from close,
        so the episode in flight when a run ends is not dropped. ``total_steps`` gives the §13 curves
        an honest x-axis -- with several environments interleaving, an episode's ordinal is not
        proportional to the step it ended at.
        """
        self._episode_file.write(
            np.array(
                [
                    self._total_steps,
                    self._episode_return,
                    self._distance(),
                    float(self._success()),
                ],
                dtype=np.float32,
            ).tobytes()
        )
        self._episode_file.flush()
        self._ee_file.flush()
        self._diag_file.flush()

    # ------------------------------------------------------------------ §8.3

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map a normalized action in [-1, 1]^m onto native actuator controls (§8.3).

            u = u_min + (a + 1) / 2 * (u_max - u_min)

        Clipping first puts the result inside ctrlrange for any finite input, which is what §10's
        "Action bounds" row asserts.
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    # ------------------------------------------------------------------ §8.1

    def _ee_position(self) -> np.ndarray:
        """End-effector position p_ee, from the hand body's pose plus the TCP offset (§8.1)."""
        xpos = self.data.xpos[self._ee_body_id]
        xmat = self.data.xmat[self._ee_body_id].reshape(3, 3)
        return xpos + xmat @ EE_OFFSET

    def _distance(self) -> float:
        """d_t = || p_ee(s_t) - g ||_2 (§8.1)."""
        return float(np.linalg.norm(self._ee_position() - self.target))

    def _sample_target(self) -> np.ndarray:
        """Sample g uniformly from the fixed reachable box (§8.1).

        Unweighted: the box is sized so every corner is comfortably inside the Panda's reach, so
        there is no unreachable region to bias away from. Draws from ``self.np_random`` only, never
        np.random, which is what makes §10's reset-determinism and seed-separation rows hold.
        """
        return self.np_random.uniform(self.target_box_low, self.target_box_high)

    def _task_reward(self) -> float:
        """Bounded dense reward r_task = exp(-alpha * d^2) (§8.1).

        alpha = 10.0 is frozen, confirmed against the measured start-state distribution over 2000
        seeds: r(0.017)=0.997, r(0.27 median)=0.478, r(0.50)=0.082, so the reward spans its full
        range across the distances the task actually produces.

        Exists only so the task actor can be evaluated.
        """
        d = self._distance()
        return float(np.exp(-self.alpha * d * d))

    def _success(self) -> bool:
        """success_t = 1[d_t < epsilon] (§8.1), epsilon = 5 cm."""
        return bool(self._distance() < self.success_tol)

    # ------------------------------------------------------------------ §8.2

    def _get_obs(self) -> dict[str, np.ndarray]:
        """o_t = [q_t, qdot_t, p_ee,t, g] as a flat single-key dict (§8.2).

        q_t is the full nq=9 vector, both finger joints included, because §8.3 actuates the gripper:
        a joint the policy can move but the world model cannot see appears as irreducible noise --
        the aleatoric uncertainty Plan2Explore's ensemble must not mistake for epistemic.
        finger_joint2 mirrors finger_joint1 by equality constraint, so one dimension is exactly
        redundant, which is harmless and keeps q_t equal to data.qpos with no index masking.

        Unnormalized; SheepRL handles its own input scaling.
        """
        state = np.concatenate(
            [self.data.qpos, self.data.qvel, self._ee_position(), self.target]
        ).astype(np.float32)
        return {"state": state}

    def _reset_model(self) -> None:
        """Restore the home keyframe with per-joint jitter on the arm (§8.1).

        The keyframe restores qpos, qvel *and* ctrl together. That matters with position actuators:
        resetting qpos alone while leaving ctrl at zero would command every joint toward position 0
        and lurch on the first step of every episode.

        Jitter is clipped to the joint limits so a perturbed start is never out of range, and is not
        applied to the fingers, whose position the policy drives from the first step anyway. It
        exists so different seeds give different initial states, not only different targets (§10
        "Seed separation").
        """
        if self._home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        if self.joint_jitter > 0.0:
            noise = self.np_random.uniform(
                -self.joint_jitter, self.joint_jitter, size=N_ARM_JOINTS
            )
            lo = self.model.jnt_range[:N_ARM_JOINTS, 0]
            hi = self.model.jnt_range[:N_ARM_JOINTS, 1]
            self.data.qpos[:N_ARM_JOINTS] = np.clip(
                self.data.qpos[:N_ARM_JOINTS] + noise, lo, hi
            )

        self.data.qvel[:] = 0.0  # start at rest: a pose, not a pose plus momentum

    # -------------------------------------------------------------------- §9

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Sample a new target and restore the start pose (§9).

        Records the outgoing episode first, while its final state is still intact.
        """
        if self._episode_file is not None and self.steps > 0:
            self._record_episode()

        super().reset(seed=seed)  # seeds self.np_random; all sampling below draws from it

        mujoco.mj_resetData(self.model, self.data)
        self._reset_model()
        self.target = self._sample_target()
        move_target_site(self.model, self._target_site, self.target)
        self.steps = 0
        self._episode_return = 0.0

        mujoco.mj_forward(self.model, self.data)  # populate xpos/xmat before the first observation

        return self._get_obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one control interval: n_substeps physics steps under a held control (§8.4).

        ``terminated`` is always False by design. Reaching is not absorbing, and an early-terminating
        episode would leak task information into the exploration objective through DreamerV3's
        continue-predictor.
        """
        self.data.ctrl[:] = self._denormalize_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        self._total_steps += 1

        obs = self._get_obs()
        reward = self._task_reward()
        self._episode_return += reward

        if self._traj_dir is not None:
            if self._ee_file is None:
                self._open_trajectory_files()
            self._ee_file.write(self._ee_position().astype(np.float32).tobytes())
            # The action as executed -- clipped exactly as _denormalize_action clips it -- followed
            # by the resulting qpos. Raw, so §13.2's threshold and tolerance stay changeable.
            self._diag_file.write(
                np.concatenate(
                    [np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0), self.data.qpos]
                )
                .astype(np.float32)
                .tobytes()
            )

        return obs, reward, False, self.steps >= self.max_episode_steps, self._info()

    def _info(self) -> dict[str, Any]:
        """Per-step diagnostics consumed by §13's metrics."""
        ctrl = self.data.ctrl
        at_limit = np.isclose(ctrl, self._ctrl_low) | np.isclose(ctrl, self._ctrl_high)
        return {
            "distance": self._distance(),
            "success": self._success(),
            "ee_position": self._ee_position().astype(np.float32),
            "target_position": self.target.astype(np.float32),
            "ctrl_saturation": float(np.mean(at_limit)),
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )
        # Named camera rather than MuJoCo's free camera, which points wherever it defaults to and
        # not at the workspace. render_scene.py attaches it when render_mode is set.
        self._renderer.update_scene(self.data, camera=CAMERA_NAME)
        return self._renderer.render()

    def close(self) -> None:
        """Release the renderer and trajectory files, recording the episode still in flight."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        if self._episode_file is not None and self.steps > 0:
            self._record_episode()

        for attr in ("_ee_file", "_episode_file", "_diag_file"):
            handle = getattr(self, attr, None)
            if handle is not None:
                handle.flush()
                handle.close()
                setattr(self, attr, None)
