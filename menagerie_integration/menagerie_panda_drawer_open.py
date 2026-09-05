"""Menagerie Panda Drawer Open - a standalone Gymnasium environment.

Not one of References/SelfEx-WM_Notes.tex's numbered phases; a new task alongside §15 Panda Push.
Follows §9's Gymnasium contract and §13's trajectory logging so scripts/metrics.py reads it
unchanged.

Mirrors menagerie_panda_push.py's structure deliberately; the numbers differ, the shape does not.
Three things genuinely differ:

  - No task MJCF at runtime. The scene is built from Python by render_scene.add_drawer_scene, the
    route probe_push_mjspec.py was written to justify, so nothing is written into the pinned
    Menagerie clone. menagerie_tasks/panda_drawer_open.xml is a readable reference, not the source.
  - No table, so the arm base is NOT raised, and measure_scene is unused -- it wants a table geom.
  - The drawer always starts CLOSED, but the goal is sampled: a target opening drawn uniformly from
    [0, open_travel], mapped onto the handle's travel line. So d_0 varies per episode even though
    the start state does not, and the goal is a partly-open drawer as often as a fully-open one.

Never imports from sheeprl. That is what keeps §3 intact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# Path.resolve() follows the §11.1 symlink back to menagerie_integration/, so this import works both
# from here and from sheeprl/envs/ where this file is linked -- the same mechanism DEFAULT_XML uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_scene import (  # noqa: E402
    CAMERA_NAME,
    add_drawer_scene,
    load_model,
    move_target_site,
    target_site_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Menagerie's stock scene, loaded unmodified as Reach does. The cabinet and drawer are attached from
#: Python by render_scene.add_drawer_scene, so nothing is written into the pinned clone: no symlink,
#: no `.git/info/exclude` entry, and none of the include-resolution constraint panda_push.xml carries.
#: menagerie_tasks/panda_drawer_open.xml is the readable reference for the same geometry, and must be
#: kept in step with add_drawer_scene by hand.
#:
#: scene.xml, not panda.xml: the latter is the robot alone, with no floor, light or skybox.
DEFAULT_XML = str(
    REPO_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
)

DEFAULT_CONTROL_DT = 0.05

#: 200 steps x 0.05 s = 10 s, as Push. Approach, hook and pull needs at least what pushing needed.
DEFAULT_MAX_EPISODE_STEPS = 200

#: NOT frozen; not measured against a rollout the way Push's beta=30 was. Chosen analytically, which
#: is possible here and was not for Push: the drawer always starts closed, so d_0 IS the sampled goal
#: opening, with no physics in between. d_0 ~ U(0.010, 0.102), median 0.056.
#:
#: beta is a LENGTH SCALE, not a shape parameter -- it carries units of 1/m^2, and exp(-beta d^2) is
#: exp(-(d/L)^2) with L = 1/sqrt(beta). So "same reward function as Push" does not mean "same beta":
#: the form only behaves the same if L tracks the task's own distances. Push's median distance is
#: 0.146 m against this task's 0.056 m, and d enters squared, so beta scales by (0.146/0.056)^2 = 6.8
#: -- which is the whole of the 30 -> 200 difference:
#:
#:      task    | constant | L = 1/sqrt   | median d | d / L | r at median
#:      Reach   | a =  10  |   0.316 m    |  0.272   | 0.86  |    0.477
#:      Push    | b =  30  |   0.183 m    |  0.146   | 0.80  |    0.528
#:      Drawer  | b = 200  |   0.071 m    |  0.056   | 0.79  |    0.534
#:
#: All three sit at d/L ~ 0.8 and r(median) ~ 0.5. They are one choice on three rulers, not three
#: choices. Carrying Push's 30 here instead scores an untouched drawer 0.910 at the median and
#: confines the whole reward to [0.73, 1.00] -- worse than the beta=10 compression ENVIRONMENT_SPEC.md
#: §13 rejected for Push.
#:
#: Fitted TO the sampling bounds below. Change either bound and refit; do not carry this across.
DEFAULT_BETA = 200.0

#: 1 cm of the 12 cm travel. Reach and Push both use 5 cm, which here would call the drawer open
#: while it is still 42 % shut, so the tolerance does not transfer.
DEFAULT_SUCCESS_TOL = 0.01

#: Goal opening, sampled per episode and mapped onto the handle's travel line. Both ends exclude a
#: band one success_tol wide, because a goal inside either band is satisfied without the behaviour
#: the task is trying to measure:
#:
#:   - below success_tol, the drawer is already there at reset -- it starts closed, so a do-nothing
#:     policy scores. Reach's floor is 1.05 % and Push's is 6 % (ENVIRONMENT_SPEC.md §13); sampling
#:     from 0 would give this task 8-10 %, the worst of the three. Starting at success_tol makes it
#:     exactly 0, which is the one genuine advantage Drawer Open's fixed start state buys.
#:   - within success_tol of the joint limit, the hard stop halts the drawer inside tolerance no
#:     matter how it was driven, so the goal rewards yanking rather than control.
#:
#: The high end is a FRACTION of the joint's own limit, read off the compiled model, so the scene
#: stays the single source of truth for the travel. 0.85 leaves 0.018 m beyond the furthest goal.
DEFAULT_TARGET_OPENING_LOW = DEFAULT_SUCCESS_TOL
DEFAULT_TARGET_OPENING_FRAC = 0.85

DEFAULT_JOINT_JITTER = 0.05

EE_BODY = "hand"
EE_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float64)
HANDLE_GEOM = "handle"
DRAWER_JOINT = "drawer_slide"

N_ARM_JOINTS = 7
#: qpos[:9] is arm + fingers; qpos[9] is the drawer slide. Unlike Push there is no free joint, so
#: nq == njnt and nothing here needs the free-joint caveat ENVIRONMENT_SPEC.md §12 records.
N_ACTUATED_QPOS = 9


class MenageriePandaDrawerOpen(gym.Env):
    """Pulling a drawer to a sampled opening by its handle with the Franka Emika Panda.

    Observation is a single flat vector under ``"state"``. Actions are normalized to ``[-1, 1]`` and
    mapped onto the model's native actuator ranges, identically to Reach and Push -- the drawer's
    slide joint is unactuated, so ``nu`` is 8 and all three tasks share an action space.
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
        target_opening_low: float = DEFAULT_TARGET_OPENING_LOW,
        target_opening_frac: float = DEFAULT_TARGET_OPENING_FRAC,
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

        # base_height stays 0: the cabinet is welded in mid-air at working height, so unlike Push
        # there is no table for the arm to sit inside and nothing to raise the base above.
        # add_bodies runs before the render assets, so the drawer's joint lands at qpos 9 -- the same
        # slot a task MJCF's own <worldbody> would give it.
        self.model = load_model(
            xml_path,
            base_height=0.0,
            render=render_mode is not None,
            add_bodies=add_drawer_scene,
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

        self._ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        if self._ee_body_id < 0:
            raise ValueError(f"model has no body named {EE_BODY!r}")

        # The handle is the only part of the drawer or cabinet whose position is observed, reported
        # or logged. It is a geom, not a body, so its pose comes from geom_xpos.
        self._handle_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, HANDLE_GEOM)
        if self._handle_geom_id < 0:
            raise ValueError(f"model has no geom named {HANDLE_GEOM!r}")

        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, DRAWER_JOINT)
        if joint_id < 0:
            raise ValueError(f"model has no joint named {DRAWER_JOINT!r}")
        # Read the addresses rather than assuming 9: they shift if the MJCF's body order changes.
        self._drawer_qpos_adr = int(self.model.jnt_qposadr[joint_id])
        self._drawer_dof_adr = int(self.model.jnt_dofadr[joint_id])
        #: Fully-open opening, in metres, read off the compiled model so the MJCF's `range` stays the
        #: single source of truth.
        self.open_travel = float(self.model.jnt_range[joint_id, 1])

        self.beta = float(beta)
        self.success_tol = float(success_tol)
        self.joint_jitter = float(joint_jitter)
        self.max_episode_steps = int(max_episode_steps)

        self.target_opening_low = float(target_opening_low)
        self.target_opening_high = float(target_opening_frac) * self.open_travel
        if not 0.0 <= self.target_opening_low <= self.target_opening_high <= self.open_travel:
            raise ValueError(
                f"goal openings must satisfy 0 <= low <= frac * travel <= {self.open_travel}; got "
                f"low={self.target_opening_low}, high={self.target_opening_high}"
            )
        self.target = self._handle_at(self.target_opening_high)

        self._home_key_id = 0 if self.model.nkey > 0 else -1

        self.steps = 0
        self._episode_return = 0.0
        self._total_steps = 0

        # o_t = [q, qdot, p_ee, p_handle, pdot_handle, g] -- 30, the same as Push.
        obs_dim = N_ACTUATED_QPOS + N_ACTUATED_QPOS + 3 + 3 + 3 + 3
        self.observation_space = spaces.Dict(
            {"state": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)}
        )

        self._traj_dir = Path(trajectory_log) if trajectory_log else None
        self._ee_file = None
        self._episode_file = None
        self._diag_file = None

        self.render_height = int(render_height)
        self.render_width = int(render_width)
        self._renderer: mujoco.Renderer | None = None

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ setup

    def _handle_at(self, opening: float) -> np.ndarray:
        """World position of the handle when the drawer is open by `opening` metres.

        Driven through forward kinematics rather than derived from the slide axis, so the answer
        stays right if the MJCF's axis, sign or handle offsets change. Called once, before the
        episode loop, and it leaves qpos as it found it.
        """
        saved = self.data.qpos.copy()
        self.data.qpos[self._drawer_qpos_adr] = opening
        mujoco.mj_forward(self.model, self.data)
        position = self.data.geom_xpos[self._handle_geom_id].copy()
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return position

    # ------------------------------------------------------------------- §13

    def _open_trajectory_files(self) -> None:
        """Open the three raw float32 append streams §13's metrics are computed from.

            ee_<pid>_<id>.f32            3 floats/step      x, y, z of the end effector
            episodes_<pid>_<id>.f32      4 floats/episode   total_steps, return, distance, success
            diag_<ncols>_<pid>_<id>.f32  ncols floats/step  action (nu), then qpos[:9]

        Identical to Reach's and Push's format, so scripts/metrics.py reads all three unchanged.
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

    # ------------------------------------------------------------------- task

    def _ee_position(self) -> np.ndarray:
        """p_ee from the hand body's pose plus the Franka TCP offset."""
        xpos = self.data.xpos[self._ee_body_id]
        xmat = self.data.xmat[self._ee_body_id].reshape(3, 3)
        return xpos + xmat @ EE_OFFSET

    def _handle_position(self) -> np.ndarray:
        return self.data.geom_xpos[self._handle_geom_id].copy()

    def _handle_velocity(self) -> np.ndarray:
        """Linear world velocity of the handle geom.

        mj_objectVelocity rather than qvel[dof] * axis, for the reason _handle_at uses forward
        kinematics: it does not encode which way the slide joint points. Returns [angular, linear].
        """
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, self._handle_geom_id, vel, 0
        )
        return vel[3:6].copy()

    def opening(self) -> float:
        """How far the drawer is pulled out, metres. 0 closed, `open_travel` fully open."""
        return float(self.data.qpos[self._drawer_qpos_adr])

    def _sample_target(self) -> np.ndarray:
        """g: the handle's position at a uniformly sampled goal opening.

        Uniform with no minimum separation from the closed drawer, matching Reach's and Push's
        decision to sample exactly as specified and measure the resulting no-op floor rather than
        patch it. That floor is target_opening_low's docstring, and §14's random-action baseline
        measures it directly.
        """
        return self._handle_at(
            self.np_random.uniform(self.target_opening_low, self.target_opening_high)
        )

    def _distance(self) -> float:
        """d_t = || p_handle(s_t) - g ||_2.

        The handle translates along one axis only, so this equals the gap between the current and
        the goal opening exactly. It is written on the handle's position because that is the only
        pose of the drawer or cabinet the task records.
        """
        return float(np.linalg.norm(self._handle_position() - self.target))

    def _task_reward(self) -> float:
        """r_task = exp(-beta * ||p_handle - g||^2)."""
        d = self._distance()
        return float(np.exp(-self.beta * d * d))

    def _success(self) -> bool:
        return bool(self._distance() < self.success_tol)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """o_t = [q, qdot, p_ee, p_handle, pdot_handle, g], unnormalized.

        Block for block this is Push's vector with the handle in the cube's place, so both tasks
        observe the same entities and the same 30 dimensions.
        """
        state = np.concatenate(
            [
                self.data.qpos[:N_ACTUATED_QPOS],
                self.data.qvel[:N_ACTUATED_QPOS],
                self._ee_position(),
                self._handle_position(),
                self._handle_velocity(),
                self.target,
            ]
        ).astype(np.float32)
        return {"state": state}

    def _reset_model(self) -> None:
        """Restore the home keyframe, jitter the arm, close the drawer.

        The drawer is closed explicitly rather than relying on the keyframe: scene.xml's was written
        for a 9-qpos model and MuJoCo zero-pads the tenth slot, which happens to be closed. Writing
        it makes that an intent rather than a coincidence.
        """
        if self._home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        if self.joint_jitter > 0.0:
            noise = self.np_random.uniform(-self.joint_jitter, self.joint_jitter, size=N_ARM_JOINTS)
            lo = self.model.jnt_range[:N_ARM_JOINTS, 0]
            hi = self.model.jnt_range[:N_ARM_JOINTS, 1]
            self.data.qpos[:N_ARM_JOINTS] = np.clip(self.data.qpos[:N_ARM_JOINTS] + noise, lo, hi)

        self.data.qpos[self._drawer_qpos_adr] = 0.0
        self.data.qvel[:] = 0.0  # arm and drawer both start at rest

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

        ``terminated`` is always False, as in Reach and Push: an early-terminating episode would
        leak task information into the exploration objective through DreamerV3's continue-predictor.
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
            "handle_position": self._handle_position().astype(np.float32),
            "target_position": self.target.astype(np.float32),
            "drawer_opening": self.opening(),
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )
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
