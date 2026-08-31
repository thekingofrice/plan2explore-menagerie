"""Menagerie Panda Push - a standalone Gymnasium environment.

Implements: References/SelfEx-WM_Notes.tex
    §15   Phase 9: Second Task - Panda Push (table, free cube, planar target, robot/cube contacts)
    §9    Phase 3: Implement a Gymnasium-Compatible Environment
    §13   trajectory logging, so coverage and task metrics survive a run

Mirrors menagerie_panda.py's structure deliberately; the numbers differ, the shape does not.

Unlike Reach, this task needs a MJCF: a cube and a table are physics, not bookkeeping. It lives in
menagerie_tasks/panda_push.xml per §5's layout, and is symlinked beside Menagerie's scene.xml so its
`<include file="scene.xml"/>` is a bare sibling reference. ENVIRONMENT_SPEC.md §1 records why the
cross-directory include fails.

Never imports from sheeprl. That is what keeps §3 intact.
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
    measure_scene,
    move_target_site,
    target_site_id,
)

# --- §15 constants (source of truth: ENVIRONMENT_SPEC.md) -------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The task file as seen from inside the Menagerie directory, via the symlink
#: scripts/install_env_wrapper.sh creates. It must be loaded from here, not from
#: menagerie_integration/: only a sibling of scene.xml makes that file's own includes resolve.
DEFAULT_XML = str(
    REPO_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda_push.xml"
)

DEFAULT_CONTROL_DT = 0.05

#: 200 steps x 0.05 s = 10 s. Longer than Reach's 5 s: pushing needs approach, contact and transport,
#: where reaching needs only approach.
DEFAULT_MAX_EPISODE_STEPS = 200

#: FROZEN 2026-08-31, confirmed against the measured cube-to-goal distribution over 2000 seeds:
#: r(0.006)=0.999, r(0.146 median)=0.526, r(0.304)=0.063. The reward spans its full range across the
#: distances this task produces. Reach's alpha=10 does NOT transfer -- Push's distances are half
#: Reach's and enter squared, which left the reward compressed into its top 60%.
DEFAULT_BETA = 30.0

DEFAULT_SUCCESS_TOL = 0.05

#: Planar target region on the table top, per §15. z is fixed at the cube's resting height, so the
#: goal is a region of the surface rather than a volume.
DEFAULT_TARGET_BOX_LOW = (0.35, -0.20)
DEFAULT_TARGET_BOX_HIGH = (0.65, 0.20)

#: Cube start, jittered like the arm so seeds differ in more than the goal.
DEFAULT_CUBE_INIT_XY = (0.45, 0.0)
DEFAULT_CUBE_JITTER = 0.03

DEFAULT_JOINT_JITTER = 0.05

EE_BODY = "hand"
EE_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float64)
CUBE_BODY = "cube"
CUBE_JOINT = "cube_free"
CUBE_GEOM = "cube_geom"
TABLE_GEOM = "table_top"

N_ARM_JOINTS = 7
#: qpos[:9] is arm + fingers; qpos[9:16] is the cube's free joint. Everything §13.2 measures about
#: joints concerns the arm, and a free joint has no limits, so the diagnostics row stops at 9.
N_ACTUATED_QPOS = 9


class MenageriePandaPush(gym.Env):
    """Pushing a free cube to a planar target with the Franka Emika Panda (§15).

    Observation is a single flat vector under ``"state"``. Actions are normalized to ``[-1, 1]`` and
    mapped onto the model's native actuator ranges, identically to Reach -- ``nu`` is unchanged by
    adding the cube, so the two tasks share an action space.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        render_mode: str | None = None,
        control_dt: float = DEFAULT_CONTROL_DT,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        beta: float = DEFAULT_BETA,
        success_tol: float = DEFAULT_SUCCESS_TOL,
        target_box_low: tuple[float, float] = DEFAULT_TARGET_BOX_LOW,
        target_box_high: tuple[float, float] = DEFAULT_TARGET_BOX_HIGH,
        cube_init_xy: tuple[float, float] = DEFAULT_CUBE_INIT_XY,
        cube_jitter: float = DEFAULT_CUBE_JITTER,
        joint_jitter: float = DEFAULT_JOINT_JITTER,
        render_height: int = 480,
        render_width: int = 640,
        trajectory_log: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        # Measured from a plain compile first, because the base is raised BY the table height and
        # the MJCF is the only place that height is declared. Raising the base does not move the
        # table, so these values hold for the real model too.
        self.cube_half, self.table_top_z = measure_scene(
            mujoco.MjModel.from_xml_path(xml_path), CUBE_GEOM, TABLE_GEOM
        )
        self.cube_rest_z = self.table_top_z + self.cube_half

        # The arm is mounted ON the work surface, as a real tabletop setup is. With the base on the
        # floor the table sits inside the arm's swept volume: random actions were in contact with it
        # on 989 of 1000 steps. MJCF cannot express this -- see render_scene.load_model.
        self.model = load_model(
            xml_path, base_height=self.table_top_z, render=render_mode is not None
        )
        self._target_site = target_site_id(self.model)
        self.data = mujoco.MjData(self.model)

        self.control_dt = float(control_dt)
        self.sim_dt = float(self.model.opt.timestep)
        substeps = self.control_dt / self.sim_dt
        if abs(substeps - round(substeps)) > 1e-9:
            raise ValueError(
                f"control_dt={self.control_dt} is not an integer multiple of the model timestep "
                f"{self.sim_dt} (got {substeps} substeps)"
            )
        self.n_substeps = int(round(substeps))

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

        # §15: o_t = [q, qdot, p_ee, p_cube, pdot_cube, g]. q and qdot cover the actuated joints
        # only -- the cube's free-joint coordinates would duplicate p_cube and add a quaternion the
        # notes do not ask for.
        obs_dim = N_ACTUATED_QPOS + N_ACTUATED_QPOS + 3 + 3 + 3 + 3
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)}
        )

        self.beta = float(beta)
        self.success_tol = float(success_tol)
        self.target_box_low = np.asarray(target_box_low, dtype=np.float64)
        self.target_box_high = np.asarray(target_box_high, dtype=np.float64)
        self.cube_init_xy = np.asarray(cube_init_xy, dtype=np.float64)
        self.cube_jitter = float(cube_jitter)
        self.joint_jitter = float(joint_jitter)
        self.max_episode_steps = int(max_episode_steps)

        self._ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
        for name, ident in ((EE_BODY, self._ee_body_id), (CUBE_BODY, self._cube_body_id)):
            if ident < 0:
                raise ValueError(f"model has no body named {name!r}")

        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
        if cube_joint_id < 0:
            raise ValueError(f"model has no joint named {CUBE_JOINT!r}")
        # Read the addresses rather than assuming 9: they shift if the MJCF's body order changes.
        self._cube_qpos_adr = int(self.model.jnt_qposadr[cube_joint_id])
        self._cube_qvel_adr = int(self.model.jnt_dofadr[cube_joint_id])


        self._home_key_id = 0 if self.model.nkey > 0 else -1

        self.steps = 0
        self.target = np.zeros(3, dtype=np.float64)
        self._episode_return = 0.0
        self._total_steps = 0

        self._traj_dir = Path(trajectory_log) if trajectory_log else None
        self._ee_file = None
        self._episode_file = None
        self._diag_file = None

        self.render_height = int(render_height)
        self.render_width = int(render_width)
        self._renderer: mujoco.Renderer | None = None

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------- §13

    def _open_trajectory_files(self) -> None:
        """Open the three raw float32 append streams §13's metrics are computed from.

            ee_<pid>_<id>.f32            3 floats/step      x, y, z of the end effector
            episodes_<pid>_<id>.f32      4 floats/episode   total_steps, return, distance, success
            diag_<ncols>_<pid>_<id>.f32  ncols floats/step  action (nu), then qpos[:9]

        Identical to Reach's format, so scripts/metrics.py reads both tasks unchanged. The diag row
        stops at the actuated joints: a free joint has no limits for §13.2 to measure.
        """
        self._traj_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{os.getpid()}_{id(self):x}"
        self._ee_file = open(self._traj_dir / f"ee_{tag}.f32", "ab")
        self._episode_file = open(self._traj_dir / f"episodes_{tag}.f32", "ab")
        ncols = self.model.nu + N_ACTUATED_QPOS
        self._diag_file = open(self._traj_dir / f"diag_{ncols}_{tag}.f32", "ab")

    def _record_episode(self) -> None:
        """Append the finished episode's outcome, from reset so no episode is missed."""
        self._episode_file.write(
            np.array(
                [self._total_steps, self._episode_return, self._distance(), float(self._success())],
                dtype=np.float32,
            ).tobytes()
        )
        self._episode_file.flush()
        self._ee_file.flush()
        self._diag_file.flush()

    # ------------------------------------------------------------------ §8.3

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """u = u_min + (a + 1) / 2 * (u_max - u_min), clipped first (§8.3)."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    # ------------------------------------------------------------------- §15

    def _ee_position(self) -> np.ndarray:
        """p_ee from the hand body's pose plus the Franka TCP offset."""
        xpos = self.data.xpos[self._ee_body_id]
        xmat = self.data.xmat[self._ee_body_id].reshape(3, 3)
        return xpos + xmat @ EE_OFFSET

    def _cube_position(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_id].copy()

    def _cube_velocity(self) -> np.ndarray:
        """Linear velocity of the cube. A free joint's first three dofs are world-frame."""
        return self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 3].copy()

    def _distance(self) -> float:
        """d_t = || p_cube(s_t) - g ||_2 -- cube to goal, not end effector to goal (§15)."""
        return float(np.linalg.norm(self._cube_position() - self.target))

    def _sample_target(self) -> np.ndarray:
        """Sample g uniformly from the planar target region, at the cube's resting height.

        Uniform with no minimum separation from the cube's start, matching Reach's decision to keep
        sampling exactly as specified and measure the resulting no-op floor instead of patching it.
        That floor must be measured for Push before the success rate is read.
        """
        xy = self.np_random.uniform(self.target_box_low, self.target_box_high)
        return np.array([xy[0], xy[1], self.cube_rest_z])

    def _task_reward(self) -> float:
        """r_task = exp(-beta * ||p_cube - g||^2) (§15). beta is not yet frozen."""
        d = self._distance()
        return float(np.exp(-self.beta * d * d))

    def _success(self) -> bool:
        return bool(self._distance() < self.success_tol)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """o_t = [q, qdot, p_ee, p_cube, pdot_cube, g] (§15), unnormalized."""
        state = np.concatenate(
            [
                self.data.qpos[:N_ACTUATED_QPOS],
                self.data.qvel[:N_ACTUATED_QPOS],
                self._ee_position(),
                self._cube_position(),
                self._cube_velocity(),
                self.target,
            ]
        ).astype(np.float32)
        return {"state": state}

    def _reset_model(self) -> None:
        """Restore the home keyframe, jitter the arm, then place the cube.

        The cube is positioned after the keyframe restore, not by it: scene.xml's keyframe was
        written for a 9-qpos model, so whatever the compiler put in the free joint's seven slots is
        not a pose we want -- in particular a zero quaternion is not a valid rotation.
        """
        if self._home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        if self.joint_jitter > 0.0:
            noise = self.np_random.uniform(-self.joint_jitter, self.joint_jitter, size=N_ARM_JOINTS)
            lo = self.model.jnt_range[:N_ARM_JOINTS, 0]
            hi = self.model.jnt_range[:N_ARM_JOINTS, 1]
            self.data.qpos[:N_ARM_JOINTS] = np.clip(
                self.data.qpos[:N_ARM_JOINTS] + noise, lo, hi
            )

        xy = self.cube_init_xy
        if self.cube_jitter > 0.0:
            xy = xy + self.np_random.uniform(-self.cube_jitter, self.cube_jitter, size=2)
        adr = self._cube_qpos_adr
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], self.cube_rest_z]
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion

        self.data.qvel[:] = 0.0  # arm and cube both start at rest

    # -------------------------------------------------------------------- §9

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._episode_file is not None and self.steps > 0:
            self._record_episode()

        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._reset_model()
        self.target = self._sample_target()
        move_target_site(self.model, self._target_site, self.target)
        self.steps = 0
        self._episode_return = 0.0

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one control interval.

        ``terminated`` is always False, as in Reach: an early-terminating episode would leak task
        information into the exploration objective through DreamerV3's continue-predictor.
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
            self._diag_file.write(
                np.concatenate(
                    [
                        np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0),
                        self.data.qpos[:N_ACTUATED_QPOS],
                    ]
                )
                .astype(np.float32)
                .tobytes()
            )

        return obs, reward, False, self.steps >= self.max_episode_steps, self._info()

    def _info(self) -> dict[str, Any]:
        return {
            "distance": self._distance(),
            "success": self._success(),
            "ee_position": self._ee_position().astype(np.float32),
            "cube_position": self._cube_position().astype(np.float32),
            "target_position": self.target.astype(np.float32),
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
