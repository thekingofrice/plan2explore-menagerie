"""Menagerie Panda Reach - a standalone Gymnasium environment.

Implements: References/SelfEx-WM_Notes.tex
    §8  Phase 2: First Menagerie Task - Panda Reach   (§8.1 task, §8.2 observation,
                                                       §8.3 action, §8.4 control interval)
    §9  Phase 3: Implement a Gymnasium-Compatible Environment
    §11.1 Phase 5 Option A - installed to sheeprl/sheeprl/envs/menagerie_panda.py

Canonical copy lives in menagerie_integration/ because the SheepRL checkout is a pinned clone and is
gitignored. scripts/install_env_wrapper.sh links it into the checkout, so at runtime the Option A
layout holds and `_target_` resolves to sheeprl.envs.menagerie_panda.MenageriePandaReach.

Every frozen constant here has its source of truth in ENVIRONMENT_SPEC.md. They arrive as
constructor arguments from sheeprl/configs/env/menagerie_panda_reach.yaml; the defaults below match
the spec so the class is usable standalone in tests.

This file must never import from sheeprl. It is a plain Gymnasium environment, which is what keeps
§3's algorithm-purity rule intact: the environment adapts to the baseline, not the reverse.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# --- §8 frozen constants (see ENVIRONMENT_SPEC.md) --------------------------

#: Loaded unmodified from the pinned Menagerie checkout. Relative to the repository root.
#: Menagerie's own scene.xml is used directly rather than a task MJCF of ours -- see
#: ENVIRONMENT_SPEC.md "Why no task MJCF".
DEFAULT_XML = "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"

DEFAULT_CONTROL_DT = 0.05           # §8.4
DEFAULT_MAX_EPISODE_STEPS = 100     # 5.0 s at 0.05 s/step
DEFAULT_ALPHA = 10.0                # §8.1, frozen before training
DEFAULT_SUCCESS_TOL = 0.05          # §8.1, 5 cm
DEFAULT_TARGET_BOX_LOW = (0.30, -0.30, 0.20)
DEFAULT_TARGET_BOX_HIGH = (0.60, 0.30, 0.60)
DEFAULT_JOINT_JITTER = 0.05         # rad, uniform, on the 7 arm joints

#: §8.1 p_ee. Menagerie's Panda declares no sites, so the end effector is derived from the `hand`
#: body's pose. 0.1034 m is the Franka TCP offset along the hand frame's +z, between the fingers.
EE_BODY = "hand"
EE_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float64)

N_ARM_JOINTS = 7                    # qpos[:7]; qpos[7:9] are the two fingers


class MenageriePandaReach(gym.Env):
    """Free-space reaching with the Franka Emika Panda (§8).

    Observation is a single flat vector under the key ``"state"`` (§8.2). Actions are normalized to
    ``[-1, 1]`` and mapped onto the model's native actuator ranges (§8.3). One ``step`` advances the
    simulation by ``control_dt`` seconds via an integer number of physics substeps (§8.4).
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
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        # --- Model -----------------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --- §8.4 Control interval ------------------------------------------
        # The substep count must be exact. A non-integral value would mean the advertised control
        # rate is a lie, which would silently corrupt every trajectory in the replay buffer.
        self.control_dt = float(control_dt)
        self.sim_dt = float(self.model.opt.timestep)
        substeps = self.control_dt / self.sim_dt
        if abs(substeps - round(substeps)) > 1e-9:
            raise ValueError(
                f"control_dt={self.control_dt} is not an integer multiple of the model timestep "
                f"{self.sim_dt} (got {substeps} substeps)"
            )
        self.n_substeps = int(round(substeps))

        # --- §8.3 Action space ----------------------------------------------
        # Bounds are read from the model, never hard-coded, so a Menagerie version bump cannot
        # silently change what a normalized action means.
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

        # --- §8.2 Observation space ------------------------------------------
        obs_dim = self.model.nq + self.model.nv + 3 + 3  # q, qdot, p_ee, g
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
            }
        )

        # --- §8.1 Task parameters --------------------------------------------
        self.alpha = float(alpha)
        self.success_tol = float(success_tol)
        self.target_box_low = np.asarray(target_box_low, dtype=np.float64)
        self.target_box_high = np.asarray(target_box_high, dtype=np.float64)
        self.joint_jitter = float(joint_jitter)
        self.max_episode_steps = int(max_episode_steps)

        # --- Cached ids -------------------------------------------------------
        self._ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        if self._ee_body_id < 0:
            raise ValueError(f"model has no body named {EE_BODY!r}")
        self._home_key_id = 0 if self.model.nkey > 0 else -1

        # --- Episode state ----------------------------------------------------
        self.steps = 0
        self.target = np.zeros(3, dtype=np.float64)

        # --- Rendering --------------------------------------------------------
        # Constructed lazily: building a Renderer requires a GL context, and the environment must be
        # importable and steppable on machines that have none.
        self.render_height = int(render_height)
        self.render_width = int(render_width)
        self._renderer: mujoco.Renderer | None = None

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ §8.3

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map a normalized action in [-1, 1]^m onto native actuator controls (§8.3).

            u = u_min + (a + 1) / 2 * (u_max - u_min)

        Clipping first guarantees the result is inside ctrlrange for any finite input, which is what
        the §10 "Action bounds" and "Joint safety" tests assert.
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    # ------------------------------------------------------------------ §8.1

    def _ee_position(self) -> np.ndarray:
        """End-effector position p_ee, derived from the hand body's pose (§8.1)."""
        xpos = self.data.xpos[self._ee_body_id]
        xmat = self.data.xmat[self._ee_body_id].reshape(3, 3)
        return xpos + xmat @ EE_OFFSET

    def _distance(self) -> float:
        """d_t = || p_ee(s_t) - g ||_2 (§8.1)."""
        return float(np.linalg.norm(self._ee_position() - self.target))

    def _sample_target(self) -> np.ndarray:
        """Sample g uniformly from the fixed reachable box (§8.1).

        Uniform in the box, unweighted. The box is sized (ENVIRONMENT_SPEC.md) so that every corner
        is comfortably inside the Panda's reach, which is what makes an unweighted uniform draw
        safe -- there is no unreachable region to bias away from.

        Draws from ``self.np_random`` only -- never np.random -- so that §10's reset-determinism and
        seed-separation tests hold.
        """
        return self.np_random.uniform(self.target_box_low, self.target_box_high)

    def _task_reward(self) -> float:
        """Bounded dense reward r_task = exp(-alpha * d^2) (§8.1), with alpha frozen.

        alpha is provisional at 10.0 until scripts/stress_rollout.py measures the real distance
        distribution; §8.1 requires it frozen before training, so the §10 stress rollout is the last
        point at which it may change.

        This reward exists ONLY so the task actor can be evaluated. The exploration actor optimizes
        Plan2Explore's intrinsic objective and never sees it.
        """
        d = self._distance()
        return float(np.exp(-self.alpha * d * d))

    def _success(self) -> bool:
        """success_t = 1[d_t < epsilon] (§8.1), epsilon = 5 cm."""
        return bool(self._distance() < self.success_tol)

    # ------------------------------------------------------------------ §8.2

    def _get_obs(self) -> dict[str, np.ndarray]:
        """o_t = [q_t, qdot_t, p_ee,t, g] as a flat single-key dict (§8.2).

        q_t is the full nq=9 joint vector: 7 arm joints plus both finger joints. The fingers are
        included because §8.3 actuates the gripper -- if the policy can move a joint the world model
        cannot see, its effect appears as irreducible noise, which is precisely the aleatoric
        uncertainty Plan2Explore's ensemble must not confuse with epistemic uncertainty.

        finger_joint2 is an equality-constrained mirror of finger_joint1, so one dimension is exactly
        redundant. That is harmless -- a constant linear dependence the encoder learns to ignore --
        and it keeps q_t literally equal to data.qpos with no index masking.

        No normalization here -- SheepRL handles its own input scaling.
        """
        state = np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
                self._ee_position(),
                self.target,
            ]
        ).astype(np.float32)
        return {"state": state}

    def _reset_model(self) -> None:
        """Reset to the home keyframe with small per-joint jitter (§8.1).

        The keyframe restores qpos, qvel and ctrl together, so the arm starts in a pose consistent
        with the controls holding it -- with position actuators, resetting qpos alone while leaving
        ctrl at zero would command a lurch toward joint position 0 on the first step.

        Jitter is applied to the 7 arm joints only, and clipped to the joint limits so a perturbed
        start can never begin out of range. The fingers are left at the keyframe value: their
        position is driven by the policy through actuator8 from the first step anyway, so jittering
        them adds nothing.

        Jitter exists so that different seeds give different initial states, not only different
        targets (§10 "Seed separation").
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

        # Always start at rest, so the initial state is a pose rather than a pose plus momentum.
        self.data.qvel[:] = 0.0

    # ------------------------------------------------------------------ §9

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # Seeds self.np_random. Everything stochastic below must draw from it.
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._reset_model()
        self.target = self._sample_target()
        self.steps = 0

        # Populate derived quantities (xpos/xmat) before the first observation is read.
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.data.ctrl[:] = self._denormalize_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        obs = self._get_obs()
        reward = self._task_reward()

        # terminated is always False by design: reaching is not absorbing, and an early-terminating
        # episode would leak task information into the exploration objective through DreamerV3's
        # continue-predictor. See ENVIRONMENT_SPEC.md §6.
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        """Per-step diagnostics consumed by the §13 metrics."""
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
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
