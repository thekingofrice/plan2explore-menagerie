"""§10 Phase 4: Environment Tests Before Learning.

Implements: References/SelfEx-WM_Notes.tex §10 -- one test per row of the table, in table order.

    API test              Gymnasium environment checker passes.
    Reset determinism     Two resets with the same seed produce identical initial state and target.
    Action bounds         Normalized actions never produce controls outside actuator limits.
    Zero action           A zero action does not produce numerical explosion or invalid state.
    Finite state          All observations and rewards remain finite over a long random rollout.
    Joint safety          Robot joints remain inside model limits under clipped actions.
    Target sampling       All targets lie in the declared reachable sampling box.
    Reward monotonicity   Moving the end effector toward the target increases dense reward.
    Render test           rgb_array returns valid frames with the advertised shape.
    Seed separation       Different seeds produce different randomized targets/initial states.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))

from menagerie_panda import MenageriePandaReach  # noqa: E402

#: Absolute, so tests do not depend on the directory pytest was invoked from.
XML_PATH = str(
    REPO_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
)


def make_env(**kwargs) -> MenageriePandaReach:
    kwargs.setdefault("xml_path", XML_PATH)
    return MenageriePandaReach(**kwargs)


@pytest.fixture
def env():
    e = make_env()
    yield e
    e.close()


# --------------------------------------------------------------- API test

def test_api_gymnasium_env_checker_passes():
    # Rendering has its own row below, and needs a GL context.
    check_env(make_env(), skip_render_check=True)


# ------------------------------------------------------- Reset determinism

def test_reset_determinism(env):
    for seed in (0, 1, 42, 12345):
        obs_a, _ = env.reset(seed=seed)
        qpos_a = env.data.qpos.copy()
        qvel_a = env.data.qvel.copy()
        target_a = env.target.copy()

        obs_b, _ = env.reset(seed=seed)

        np.testing.assert_array_equal(obs_a["state"], obs_b["state"])
        np.testing.assert_array_equal(qpos_a, env.data.qpos)
        np.testing.assert_array_equal(qvel_a, env.data.qvel)
        np.testing.assert_array_equal(target_a, env.target)


# ----------------------------------------------------------- Action bounds

def test_action_bounds(env):
    """§8.3: the environment owns the clipping, so oversized actions must still be in range."""
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    for _ in range(500):
        action = rng.uniform(-3.0, 3.0, size=env.model.nu).astype(np.float32)
        env.step(action)
        assert np.all(env.data.ctrl >= env._ctrl_low - 1e-9)
        assert np.all(env.data.ctrl <= env._ctrl_high + 1e-9)


# ------------------------------------------------------------- Zero action

def test_zero_action(env):
    """Tested as hold-position, plus a 1e-6 perturbation.

    §8.3 maps a_i = 0 to the MIDPOINT of actuator i's ctrlrange, so the literal zero vector commands
    real motion. The action that holds the arm still is §8.3's map inverted on the ctrl the keyframe
    restored. The perturbation matters because an environment whose state diverges under a miniscule
    input change is unusable as a world model's prediction target.
    """
    noise, n_steps = 1e-6, 100

    def hold_action(e):
        lo, hi = e._ctrl_low, e._ctrl_high
        return (2.0 * (e.data.ctrl - lo) / (hi - lo) - 1.0).astype(np.float32)

    env.reset(seed=0)
    start_qpos = env.data.qpos.copy()
    a = hold_action(env)

    for _ in range(n_steps):
        env.step(a)
    qpos_clean = env.data.qpos.copy()

    assert np.all(np.isfinite(qpos_clean)) and np.all(np.isfinite(env.data.qvel))
    assert np.abs(qpos_clean - start_qpos).max() < 0.10, "hold action drifted off the start pose"

    rng = np.random.default_rng(0)
    perturbed = a + rng.uniform(-noise, noise, size=a.shape).astype(np.float32)
    env.reset(seed=0)
    for _ in range(n_steps):
        env.step(perturbed)

    assert np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel))
    dq = np.abs(env.data.qpos - qpos_clean).max()
    assert dq < 1e-2, f"{noise:g} action noise moved qpos by {dq:.3e} rad over {n_steps} steps"


# ------------------------------------------------------------- Finite state

def test_finite_state(env):
    """All observations and rewards remain finite over a long random rollout."""
    rng = np.random.default_rng(0)
    env.reset(seed=0)

    for step in range(2000):
        action = rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32)
        obs, reward, _, truncated, info = env.step(action)

        assert np.all(np.isfinite(obs["state"])), f"non-finite observation at step {step}"
        assert np.isfinite(reward), f"non-finite reward at step {step}"
        assert np.isfinite(info["distance"]), f"non-finite distance at step {step}"
        if truncated:
            env.reset(seed=step)


# ------------------------------------------------------------- Joint safety

def test_joint_safety(env):
    """Robot joints remain inside model limits under clipped actions."""
    rng = np.random.default_rng(0)
    env.reset(seed=0)

    limited = env.model.jnt_limited.astype(bool)
    lo = env.model.jnt_range[:, 0]
    hi = env.model.jnt_range[:, 1]

    for step in range(1000):
        action = rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32)
        env.step(action)

        qpos = env.data.qpos[: env.model.njnt]
        assert np.all(qpos[limited] >= lo[limited] - 1e-6), f"joint below limit at step {step}"
        assert np.all(qpos[limited] <= hi[limited] + 1e-6), f"joint above limit at step {step}"


# ----------------------------------------------------------- Target sampling

def test_target_sampling(env):
    """All targets lie in the declared reachable sampling box."""
    for seed in range(2000):
        _, info = env.reset(seed=seed)
        g = info["target_position"]
        assert np.all(g >= env.target_box_low - 1e-9), f"seed {seed}: {g} below box"
        assert np.all(g <= env.target_box_high + 1e-9), f"seed {seed}: {g} above box"


# ------------------------------------------------------- Reward monotonicity

def test_reward_monotonicity(env):
    """Moving the end effector toward the target increases dense reward.

    Joints are moved kinematically -- perturb qpos, run forward kinematics, keep the perturbation
    only when it brings p_ee closer to g -- and the reward is asserted to rise on every accepted
    move. Driving the arm with the controller instead would test the position servos, not r(d).
    """
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    prev_d = env._distance()
    prev_r = env._task_reward()
    accepted = 0

    for _ in range(3000):
        candidate = env.data.qpos.copy()
        candidate[:7] += rng.normal(0.0, 0.02, size=7)

        saved = env.data.qpos.copy()
        env.data.qpos[:] = candidate
        mujoco.mj_forward(env.model, env.data)

        d = env._distance()
        if d < prev_d:
            r = env._task_reward()
            assert r > prev_r, (
                f"distance fell {prev_d:.6f} -> {d:.6f} but reward fell {prev_r:.6f} -> {r:.6f}"
            )
            prev_d, prev_r = d, r
            accepted += 1
        else:
            env.data.qpos[:] = saved
            mujoco.mj_forward(env.model, env.data)

    assert accepted >= 20, f"only {accepted} approaching moves found; test is not exercising r(d)"


# -------------------------------------------------------------- Render test

def test_render():
    """rgb_array returns valid frames with the advertised shape."""
    e = make_env(render_mode="rgb_array")
    try:
        e.reset(seed=0)
        frame = e.render()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GL context available (set MUJOCO_GL=egl): {exc}")
    finally:
        e.close()

    assert frame is not None
    assert frame.shape == (e.render_height, e.render_width, 3)
    assert frame.dtype == np.uint8


# ----------------------------------------------------------- Seed separation

def test_seed_separation(env):
    """Different seeds produce different randomized targets/initial states."""
    n_seeds = 64
    targets, qpos = [], []
    for seed in range(n_seeds):
        env.reset(seed=seed)
        targets.append(env.target.copy())
        qpos.append(env.data.qpos.copy())

    assert len(np.unique(np.array(targets), axis=0)) == n_seeds
    assert len(np.unique(np.array(qpos), axis=0)) == n_seeds
