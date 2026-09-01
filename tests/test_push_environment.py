"""§10 Phase 4 applied to Panda Push (§15).

Implements: References/SelfEx-WM_Notes.tex §10's table for the Push task. The five rows that do not
depend on what the task measures are checked the same way as Reach; three change because the task
does; and two rows exist only here, because only Push has an object.

    API test              Gymnasium environment checker passes.
    Reset determinism     Same seed -> identical initial state, cube pose and target.
    Action bounds         Normalized actions never produce controls outside actuator limits.
    Zero action           A zero action does not produce numerical explosion or invalid state.
    Finite state          Observations and rewards remain finite over a long random rollout.
    Joint safety          ARM joints stay inside limits; the free joint has none (differs).
    Target sampling       Targets lie in the planar region at the cube's resting height (differs).
    Reward monotonicity   Moving the CUBE toward the target increases reward (differs).
    Render test           rgb_array returns valid frames with the advertised shape.
    Seed separation       Different seeds produce different targets and cube starts.
    Cube rests            The cube stays on the table under no action (Push only).
    Model geometry        Derived table/cube dimensions match the MJCF (Push only).

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

from menagerie_panda_push import MenageriePandaPush  # noqa: E402

#: Absolute, and pointing at the symlink inside the Menagerie checkout: panda_push.xml's
#: `<include file="scene.xml"/>` only resolves from beside scene.xml (ENVIRONMENT_SPEC.md §12).
XML_PATH = str(
    REPO_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda_push.xml"
)

N_ACTUATED = 9  # arm + fingers; the cube's free joint is not among them

#: Push tolerates far more joint-limit overshoot than Reach's 1e-3. MuJoCo's joint limits are soft
#: constraints, so a contact-rich task pushes joints past their stops by small amounts as ordinary
#: solver behaviour. §10's row is about the ACTION MAP -- §8.3 guarantees no valid action commands an
#: out-of-range target -- not about the physics never violating a soft constraint. Reach's strict
#: tolerance encodes an assumption that nothing pushes back, which a table breaks without anything
#: being wrong. 0.05 rad (2.9 deg) still fails loudly on real divergence, which would be radians.
CONTACT_TOL = 0.05


def make_env(**kwargs) -> MenageriePandaPush:
    kwargs.setdefault("xml_path", XML_PATH)
    return MenageriePandaPush(**kwargs)


@pytest.fixture
def env():
    e = make_env()
    yield e
    e.close()


# --------------------------------------------------------------- API test

def test_api_gymnasium_env_checker_passes():
    check_env(make_env(), skip_render_check=True)


# ------------------------------------------------------- Reset determinism

def test_reset_determinism(env):
    """Covers the cube too: it is placed by _reset_model, not by the keyframe."""
    for seed in (0, 1, 42, 12345):
        obs_a, info_a = env.reset(seed=seed)
        qpos_a, qvel_a = env.data.qpos.copy(), env.data.qvel.copy()
        target_a = env.target.copy()

        obs_b, info_b = env.reset(seed=seed)

        np.testing.assert_array_equal(obs_a["state"], obs_b["state"])
        np.testing.assert_array_equal(qpos_a, env.data.qpos)
        np.testing.assert_array_equal(qvel_a, env.data.qvel)
        np.testing.assert_array_equal(target_a, env.target)
        np.testing.assert_array_equal(info_a["cube_position"], info_b["cube_position"])


# ----------------------------------------------------------- Action bounds

def test_action_bounds(env):
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    for _ in range(500):
        action = rng.uniform(-3.0, 3.0, size=env.model.nu).astype(np.float32)
        env.step(action)
        assert np.all(env.data.ctrl >= env._ctrl_low - 1e-9)
        assert np.all(env.data.ctrl <= env._ctrl_high + 1e-9)


# ------------------------------------------------------------- Zero action

def test_zero_action(env):
    """§8.3 maps a_i = 0 to the midpoint of each ctrlrange, so this is a real command, not a hold."""
    env.reset(seed=0)
    for _ in range(200):
        obs, reward, _, _, _ = env.step(np.zeros(env.model.nu, dtype=np.float32))
        assert np.all(np.isfinite(obs["state"]))
        assert np.isfinite(reward)


# ------------------------------------------------------------ Finite state

def test_finite_state(env):
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    for step in range(2000):
        action = rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32)
        obs, reward, _, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs["state"])), f"non-finite observation at step {step}"
        assert np.isfinite(reward), f"non-finite reward at step {step}"
        assert np.isfinite(info["distance"])
        if truncated:
            env.reset(seed=step)


# ------------------------------------------------------------ Joint safety

def test_joint_safety(env):
    """Only the ARM joints have limits. The cube's free joint has none, so jnt_range is
    meaningless for it and checking it would fail on a perfectly healthy cube."""
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    lo = env.model.jnt_range[:N_ACTUATED, 0]
    hi = env.model.jnt_range[:N_ACTUATED, 1]
    limited = env.model.jnt_limited[:N_ACTUATED].astype(bool)

    worst = np.zeros(N_ACTUATED)
    for _ in range(1000):
        env.step(rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32))
        q = env.data.qpos[:N_ACTUATED]
        worst = np.maximum(worst, np.where(limited, np.maximum(lo - q, q - hi), 0.0))

    # Reported rather than asserted joint by joint: if this ever fires, the magnitude is what
    # separates solver softness from an arm that has escaped its range, and it should be in the
    # failure message rather than requiring a separate probe run to recover.
    over = {
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, j): round(float(worst[j]), 5)
        for j in range(N_ACTUATED)
        if worst[j] > CONTACT_TOL
    }
    assert not over, f"joints past their limits by more than {CONTACT_TOL} rad: {over}"


# --------------------------------------------------------- Target sampling

def test_target_sampling(env):
    """The region is planar: x and y are sampled, z is pinned to the cube's resting height."""
    for seed in range(500):
        _, info = env.reset(seed=seed)
        g = info["target_position"]
        assert env.target_box_low[0] <= g[0] <= env.target_box_high[0]
        assert env.target_box_low[1] <= g[1] <= env.target_box_high[1]
        assert g[2] == pytest.approx(env.cube_rest_z)


def test_cube_starts_within_jitter(env):
    for seed in range(500):
        _, info = env.reset(seed=seed)
        p = info["cube_position"]
        assert abs(p[0] - env.cube_init_xy[0]) <= env.cube_jitter + 1e-6
        assert abs(p[1] - env.cube_init_xy[1]) <= env.cube_jitter + 1e-6
        assert p[2] == pytest.approx(env.cube_rest_z, abs=1e-6)


# ----------------------------------------------------- Reward monotonicity

def test_reward_monotonicity(env):
    """r = exp(-beta d^2) on the CUBE's distance, so moving the cube closer must raise it.

    Driven by writing the cube's qpos directly rather than by pushing it: this asserts the reward
    function, not the controller's ability to make contact.
    """
    env.reset(seed=0)
    adr = env._cube_qpos_adr
    direction = env.target[:2] - env.data.qpos[adr : adr + 2]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    start = env.data.qpos[adr : adr + 2].copy()

    rewards = []
    for frac in np.linspace(0.0, 1.0, 11):
        env.data.qpos[adr : adr + 2] = start + direction * frac * np.linalg.norm(
            env.target[:2] - start
        )
        mujoco.mj_forward(env.model, env.data)
        rewards.append(env._task_reward())

    assert np.all(np.diff(rewards) > 0), f"reward not increasing toward the goal: {rewards}"
    assert rewards[-1] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------- Render test

def test_render():
    e = make_env(render_mode="rgb_array")
    try:
        e.reset(seed=0)
        frame = e.render()
        assert frame.shape == (e.render_height, e.render_width, 3)
        assert frame.dtype == np.uint8
    finally:
        e.close()


# ---------------------------------------------------------- Seed separation

def test_seed_separation(env):
    targets, cubes = [], []
    for seed in range(20):
        _, info = env.reset(seed=seed)
        targets.append(tuple(info["target_position"]))
        cubes.append(tuple(info["cube_position"]))

    assert len(set(targets)) == len(targets), "targets repeat across seeds"
    assert len(set(cubes)) == len(cubes), "cube starts repeat across seeds"


# ------------------------------------------------- Cube rests (Push only)

def test_cube_rests_on_table(env):
    """With no arm contact the cube must sit still on the surface.

    A cube spawned intersecting the table would be ejected by the contact solver, and a cube spawned
    above it would fall -- either way the task starts from a state the spec does not describe.
    Soft contacts allow a fraction of a millimetre of penetration, hence the tolerance.
    """
    env.reset(seed=0)
    z_start = float(env._cube_position()[2])

    # Physics is advanced directly rather than through env.step: §8.3 maps a zero action to the
    # MIDPOINT of each ctrlrange, which would drive the arm out of its home pose and possibly into
    # the cube -- failing this test for a reason that has nothing to do with the cube resting.
    # mj_resetDataKeyframe restored ctrl along with qpos, so leaving ctrl alone holds the home pose.
    for _ in range(200 * env.n_substeps):
        mujoco.mj_step(env.model, env.data)

    z_end = float(env._cube_position()[2])
    assert abs(z_end - env.cube_rest_z) < 1e-3, f"cube left the surface: z={z_end}"
    assert abs(z_end - z_start) < 1e-3, f"cube drifted vertically: {z_start} -> {z_end}"


# --------------------------------------------- Model geometry (Push only)

def test_derived_geometry_matches_model(env):
    """_measure_scene reads the MJCF instead of repeating it, so a tuned XML cannot silently
    disagree with the Python. These are the values panda_push.xml declares today."""
    assert env.cube_half == pytest.approx(0.025)
    assert env.table_top_z == pytest.approx(0.22)
    assert env.cube_rest_z == pytest.approx(0.245)


def test_action_space_matches_reach(env):
    """§15 must not change the action space: ENVIRONMENT_SPEC.md §12 keeps Push and Reach
    comparable by keeping nu at 8, which is also why Reach carries the gripper actuator."""
    assert env.model.nu == 8
    assert env.action_space.shape == (8,)


def test_observation_dimension(env):
    """§15: o_t = [q(9), qdot(9), p_ee(3), p_cube(3), pdot_cube(3), g(3)] = 30."""
    obs, _ = env.reset(seed=0)
    assert obs["state"].shape == (30,)
    assert env.observation_space["state"].shape == (30,)
