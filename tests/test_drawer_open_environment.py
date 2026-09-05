"""§10 Phase 4 applied to Drawer Open.

Implements: References/SelfEx-WM_Notes.tex §10's table for the Drawer Open task. Six rows are
checked exactly as Reach and Push check them; three change because the task does; and eight rows
exist only here.

    API test              Gymnasium environment checker passes.
    Reset determinism     Same seed -> identical initial state, drawer opening and target.
    Action bounds         Normalized actions never produce controls outside actuator limits.
    Zero action           A zero action does not produce numerical explosion or invalid state.
    Finite state          Observations and rewards remain finite over a long random rollout.
    Joint safety          Arm joints in radians AND the drawer slide in metres (differs).
    Target sampling       Goals lie on the handle's travel SEGMENT, not in a box (differs).
    Reward monotonicity   Reward rises toward the goal and falls past it (differs).
    Render test           rgb_array returns valid frames with the advertised shape.
    Seed separation       Different seeds produce different goals and arm poses.
    Handle interactable   Sampled actions reach the handle and move it (Drawer only).
    Object identity       Cabinet, drawer and handle are real compiled MuJoCo entities (Drawer only).
    Drawer starts closed  Every seed begins at opening 0 (Drawer only).
    No-op floor           The 10.4 % floor sampling from rest implies is measured (Drawer only).
    Drawer stays in range A kicked drawer comes to rest inside [0, travel] (Drawer only).
    Body order            The slide joint lands at qposadr 9 (Drawer only).
    Model geometry        Derived handle travel and goal bounds match the scene (Drawer only).
    XML/Python parity     panda_drawer_open.xml compiles to add_drawer_scene's model (Drawer only).

Two rows differ in kind rather than in threshold. "Target sampling" checks a one-dimensional segment
because the handle translates on one axis, so Push's three-sided box has nothing to check here.
"Reward monotonicity" sweeps BOTH sides of the goal: goals stop at 80 % of travel, so overshooting is
reachable, and a reward that kept rising past the goal would be a real defect Push cannot exhibit.

Two rows exist because this task's hardest failure is silent. A handle welded out of reach passes
every other row -- determinism, finiteness, monotonicity by hand-written qpos -- so
test_handle_is_interactable is what fails when the geometry is unsolvable, and
test_scene_objects_are_mujoco_initialized is what fails when a geom never compiled at all.

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

from menagerie_panda_drawer_open import MenageriePandaDrawerOpen  # noqa: E402
from render_scene import add_drawer_scene, load_model  # noqa: E402

MENAGERIE = REPO_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"

#: Menagerie's stock scene, loaded unmodified. Unlike Push there is no task MJCF in the load path:
#: the cabinet and drawer are attached from Python by add_drawer_scene.
XML_PATH = str(MENAGERIE / "scene.xml")

#: The reference MJCF, if install_env_wrapper.sh has linked it beside scene.xml. Only the parity
#: test needs it, and that test skips when it is absent.
DRAWER_XML = MENAGERIE / "panda_drawer_open.xml"

N_ACTUATED = 9  # arm + fingers; the drawer's slide joint is not among them

#: As Push. MuJoCo's joint limits are soft constraints, so a contact-rich task pushes joints past
#: their stops by small amounts as ordinary solver behaviour. §10's row is about the ACTION MAP --
#: §8.3 guarantees no valid action commands an out-of-range target -- not about the physics never
#: violating a soft constraint. 0.05 rad (2.9 deg) still fails loudly on real divergence.
CONTACT_TOL = 0.05

#: The drawer's limits are the same kind of soft constraint, but the joint is a slide measured in
#: METRES, so Push's radian tolerance does not transfer: 0.05 m is 42 % of the 0.12 m travel, which
#: would pass a drawer that had escaped its cabinet. 5 mm is 4 % of travel.
SLIDE_TOL = 5e-3

#: Every geom add_drawer_scene builds, in the order it builds them.
SCENE_GEOMS = (
    "cabinet_top",
    "cabinet_back",
    "cabinet_side_left",
    "cabinet_side_right",
    "drawer_body",
    "handle_stem",
    "handle",
)


def make_env(**kwargs) -> MenageriePandaDrawerOpen:
    kwargs.setdefault("xml_path", XML_PATH)
    return MenageriePandaDrawerOpen(**kwargs)


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
    """Covers the drawer too: it is closed by _reset_model, not by the keyframe's zero pad."""
    for seed in (0, 1, 42, 12345):
        obs_a, info_a = env.reset(seed=seed)
        qpos_a, qvel_a = env.data.qpos.copy(), env.data.qvel.copy()
        target_a = env.target.copy()

        obs_b, info_b = env.reset(seed=seed)

        np.testing.assert_array_equal(obs_a["state"], obs_b["state"])
        np.testing.assert_array_equal(qpos_a, env.data.qpos)
        np.testing.assert_array_equal(qvel_a, env.data.qvel)
        np.testing.assert_array_equal(target_a, env.target)
        np.testing.assert_array_equal(info_a["handle_position"], info_b["handle_position"])
        assert info_a["drawer_opening"] == info_b["drawer_opening"]


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
    """The arm, in radians. Push could only check these because its free joint has no limits."""
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

    # Reported rather than asserted joint by joint: if this fires, the magnitude is what separates
    # solver softness from an arm that has escaped its range, and it belongs in the failure message.
    over = {
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, j): round(float(worst[j]), 5)
        for j in range(N_ACTUATED)
        if worst[j] > CONTACT_TOL
    }
    assert not over, f"joints past their limits by more than {CONTACT_TOL} rad: {over}"


def test_drawer_joint_safety(env):
    """The drawer slide, in metres. Unlike Push's free joint this one HAS limits, so §10's joint
    safety row covers the task object here and did not for Push."""
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(1000):
        env.step(rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32))
        x = env.opening()
        worst = max(worst, -x, x - env.open_travel)

    assert worst <= SLIDE_TOL, f"drawer left [0, {env.open_travel}] by {worst:.5f} m"


# --------------------------------------------------------- Target sampling

def test_target_sampling(env):
    """The declared region is a segment, not a box: the handle translates on one axis, so every goal
    must sit on the line it sweeps, between rest and 80 % of the joint's travel."""
    env.reset(seed=0)
    closed = env._handle_at(0.0)
    lo = env._handle_at(env.target_opening_high)  # smallest x the goal may take

    for seed in range(500):
        _, info = env.reset(seed=seed)
        g = info["target_position"]
        assert lo[0] - 1e-9 <= g[0] <= closed[0] + 1e-9, f"goal off the travel segment: {g}"
        assert g[1] == pytest.approx(closed[1], abs=1e-9)
        assert g[2] == pytest.approx(closed[2], abs=1e-9)


def test_target_excludes_a_tolerance_band_at_both_ends(env):
    """Neither end of the sampling range may contain a goal that is satisfied without control.

    Below success_tol the drawer is already there at reset; within success_tol of the joint limit the
    hard stop halts it inside tolerance however it was driven. Both bands reward something other than
    the behaviour the task measures, so both are excluded.
    """
    assert env.target_opening_low >= env.success_tol
    assert env.target_opening_high <= env.open_travel - env.success_tol
    assert env.target_opening_high == pytest.approx(0.85 * env.open_travel)


def test_no_op_success_floor_is_zero(env):
    """The advantage the fixed start state buys, and the reason the low bound is not 0.

    A do-nothing policy scores 0 %, against Reach's 1.05 % and Push's 6 % (ENVIRONMENT_SPEC.md §13),
    so §13's success rate can be read directly rather than against a floor. Sampling from rest
    instead would put this at 8-10 %, the worst of the three tasks.
    """
    for seed in range(1000):
        _, info = env.reset(seed=seed)
        assert not info["success"], f"a closed drawer succeeded at seed {seed}"
        assert info["distance"] >= env.success_tol


# ----------------------------------------------------- Reward monotonicity

def test_reward_monotonicity(env):
    """r = exp(-beta d^2) on the HANDLE's distance, so moving the drawer toward the sampled goal
    must raise it -- and moving past the goal must lower it again.

    Driven by writing the slide joint's qpos directly rather than by pulling it: this asserts the
    reward function, not the controller's ability to hook the handle. The goal is pinned mid-travel
    rather than sampled, so both halves of the sweep exist for every run of the suite.
    """
    env.reset(seed=0)
    adr = env._drawer_qpos_adr
    goal = 0.8 * env.target_opening_high
    env.target = env._handle_at(goal)

    def sweep(a, b):
        out = []
        for opening in np.linspace(a, b, 20):
            env.data.qpos[adr] = opening
            mujoco.mj_forward(env.model, env.data)
            out.append(env._task_reward())
        return out

    approaching = sweep(0.0, goal)
    assert np.all(np.diff(approaching) > 0), f"reward not increasing toward the goal: {approaching}"
    assert approaching[-1] == pytest.approx(1.0, abs=1e-6)

    overshooting = sweep(goal, env.open_travel)
    assert np.all(np.diff(overshooting) < 0), f"reward not falling past the goal: {overshooting}"


def test_reward_spans_its_range(env):
    """beta is not frozen against a measured rollout the way Push's was. This pins the analytic
    choice: at the worst sampled start the reward must be near its floor, not already most of the
    way up -- the compression ENVIRONMENT_SPEC.md §13 rejected for Push at beta = 10.
    """
    worst = env.target_opening_high  # drawer starts closed, so d_0 is the goal opening
    # 0.2, not Push's 0.063: goals are uniform on a 1-D segment, so this task's worst start is only
    # 1.8x its median, where Push's 2-D distance distribution puts its worst at 2.1x with a longer
    # tail. beta cannot close that gap -- it is geometry. At beta=200 the worst start scores 0.125.
    assert np.exp(-env.beta * worst**2) < 0.2, "worst start already scores too well; beta too low"
    assert np.exp(-env.beta * (0.5 * worst) ** 2) > 0.25, "median start too harsh; beta too high"


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
    """As Push, with one asymmetry: the goal and the arm pose vary, but the handle does not -- the
    drawer is closed at every reset by design, so test_drawer_starts_closed covers it instead.

    Compared on target[0] alone, not the whole vector: the goal sits on the handle's travel line, so
    x is the only coordinate carrying the sample. Comparing all three would let y or z noise that
    should not exist count as separation.
    """
    targets, arms = [], []
    for seed in range(20):
        env.reset(seed=seed)
        targets.append(round(float(env.target[0]), 12))
        arms.append(tuple(np.round(env.data.qpos[:7], 12)))

    assert len(set(targets)) == len(targets), "goals repeat across seeds"
    assert len(set(arms)) == len(arms), "arm start poses repeat across seeds"


# ------------------------------------------ Drawer starts closed (Drawer only)

def test_drawer_starts_closed(env):
    """The start state, unlike the goal, carries no randomness at all.

    No success assertion here: goals are sampled from rest upward, so a goal inside success_tol of
    closed makes a closed drawer a legitimate success. That is the 10.4 % floor
    test_no_op_success_floor measures, not a bug.
    """
    for seed in range(200):
        _, info = env.reset(seed=seed)
        assert info["drawer_opening"] == 0.0
        assert env.opening() == 0.0


def test_start_distance_is_the_goal_opening(env):
    """The drawer starts closed, so d_0 is exactly how far open the goal asks it to be. This is what
    makes beta choosable analytically -- the d_0 distribution is the sampling distribution, with no
    physics in between, unlike Push where the cube's start also varies."""
    for seed in range(200):
        _, info = env.reset(seed=seed)
        goal_opening = np.linalg.norm(env.target - env._handle_at(0.0))
        assert info["distance"] == pytest.approx(goal_opening, abs=1e-9)
        assert info["distance"] <= env.target_opening_high + 1e-9


# ---------------------------------------- Drawer stays in range (Drawer only)

def test_kicked_drawer_comes_to_rest_in_range(env):
    """damping and frictionloss must actually stop the drawer, and the limits must hold it.

    At rest nothing loads this joint -- the slide axis is horizontal and gravity is vertical -- so a
    "does it hold position" test would pass even on a frictionless joint. Kicking it is what makes
    the assertion about the joint's dynamics rather than about the absence of forces.

    Physics is advanced directly rather than through env.step: §8.3 maps a zero action to the
    MIDPOINT of each ctrlrange, which would drive the arm out of its home pose and possibly into the
    drawer, failing this for a reason that has nothing to do with the joint.
    """
    env.reset(seed=0)
    env.data.qvel[env._drawer_dof_adr] = 0.5

    for _ in range(400 * env.n_substeps):
        mujoco.mj_step(env.model, env.data)
        x = env.opening()
        assert -SLIDE_TOL <= x <= env.open_travel + SLIDE_TOL, f"drawer escaped its range at {x}"

    assert abs(env.data.qvel[env._drawer_dof_adr]) < 1e-2, "drawer never came to rest"


# ----------------------------------------------------- Body order (Drawer only)

def test_slide_joint_lands_at_qposadr_9(env):
    """add_drawer_scene APPENDS to worldbody, so the drawer's joint must follow the arm's nine.

    If MjSpec ever prepended instead, the slide would take qpos 0 and shift the whole arm -- and
    every qpos[:9] slice in the wrapper, in scripts/buffer_metrics.py and in the diag_*.f32 stream
    would silently be reading the wrong joints. Nothing else in the suite would notice.
    """
    assert env._drawer_qpos_adr == 9
    assert env._drawer_dof_adr == 9
    assert env.model.nq == 10
    assert env.model.nv == 10
    assert env.model.njnt == 10


def test_keyframe_pad_is_the_closed_drawer(env):
    """scene.xml's `home` keyframe declares 9 qpos and the model has 10, so MuJoCo zero-pads the
    tenth. Zero happens to mean "closed" for a slide joint, unlike Push, where the pad covers a
    quaternion and zeros are not a valid rotation. _reset_model writes it anyway; this checks the
    pad it is writing over."""
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    assert env.data.qpos[env._drawer_qpos_adr] == 0.0


# ------------------------------------------- Model geometry (Drawer only)

def test_derived_geometry_matches_model(env):
    """open_travel is read off the compiled model instead of repeating 0.12 in Python, so a tuned
    scene cannot silently disagree. These are the values add_drawer_scene declares today."""
    assert env.open_travel == pytest.approx(0.12)
    closed = env._handle_at(0.0)
    fully_open = env._handle_at(env.open_travel)
    assert closed[0] == pytest.approx(0.48, abs=1e-6)
    assert fully_open[0] == pytest.approx(0.36, abs=1e-6)
    assert closed[0] - fully_open[0] == pytest.approx(env.open_travel, abs=1e-9)
    # 80 % of travel, so the furthest goal is 0.48 - 0.096.
    assert env.target_opening_high == pytest.approx(0.096, abs=1e-9)


def test_handle_at_is_side_effect_free(env):
    """_handle_at drives the drawer through forward kinematics and must restore qpos. If it did not,
    calling it would silently open the drawer mid-episode."""
    env.reset(seed=0)
    before = env.data.qpos.copy()
    env._handle_at(env.open_travel)
    np.testing.assert_array_equal(before, env.data.qpos)


def test_action_space_matches_reach_and_push(env):
    """The slide joint is unactuated, so all three tasks share an action space -- the same
    comparability ENVIRONMENT_SPEC.md §12 keeps between Reach and Push."""
    assert env.model.nu == 8
    assert env.action_space.shape == (8,)


def test_observation_dimension(env):
    """o_t = [q(9), qdot(9), p_ee(3), p_handle(3), pdot_handle(3), g(3)] = 30 -- block for block
    Push's vector with the handle in the cube's place, so both tasks observe the same entities."""
    obs, _ = env.reset(seed=0)
    assert obs["state"].shape == (30,)
    assert env.observation_space["state"].shape == (30,)


# ------------------------------------ Handle interactable (Drawer only)

def test_handle_is_interactable(env):
    """Sampled actions must be able to reach the handle and move it.

    Everything else in this suite would pass on a drawer the arm cannot touch: a handle welded out
    of reach still resets deterministically, still yields finite observations, and still produces a
    monotone reward when the joint is written by hand. This is the row that fails if the geometry is
    unsolvable, and it is the reason Push carries scripts/probe_push_reachability.py.

    Deliberately stochastic, so a failure is real information rather than a broken fixture: it means
    random exploration cannot find the affordance, which for a Plan2Explore task is the finding.
    Budgeted at 20 rollouts and stopped as soon as both conditions hold.
    """
    handle_ids = {
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("handle", "handle_stem")
    }
    rng = np.random.default_rng(0)

    touched, moved = False, 0.0
    for episode in range(20):
        env.reset(seed=episode)
        start = env._handle_position()

        for _ in range(env.max_episode_steps):
            env.step(rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32))

            for i in range(env.data.ncon):
                contact = env.data.contact[i]
                if contact.geom1 in handle_ids or contact.geom2 in handle_ids:
                    touched = True

            here = env._handle_position()
            moved = max(moved, abs(here[0] - start[0]))
            # The slide joint permits x only. If y or z ever move, the joint is not doing its job
            # and every distance in the task is measured against a handle that is free to wander.
            assert here[1] == pytest.approx(start[1], abs=1e-9)
            assert here[2] == pytest.approx(start[2], abs=1e-9)

        if touched and moved > 1e-3:
            break

    assert touched, "no sampled action ever brought an arm geom into contact with the handle"
    assert moved > 1e-3, f"the handle was touched but never moved; best was {moved:.6f} m"


# --------------------------------- MuJoCo object identity (Drawer only)

def test_scene_objects_are_mujoco_initialized(env):
    """The cabinet, drawer and handle must be real compiled MuJoCo entities, not names that happen
    to resolve. mj_name2id returning -1 is the failure the wrapper's own guards catch; this checks
    the layer above -- that the model and data are the types the bindings expect, and that every
    geom add_drawer_scene builds survived compilation into a named view."""
    assert isinstance(env.model, mujoco.MjModel)
    assert isinstance(env.data, mujoco.MjData)

    # A named view has no public class to import, so it is compared against an index-built one of
    # the same kind -- which is exactly what "this name resolved to a real entity" means.
    for name in ("cabinet", "drawer"):
        assert isinstance(env.model.body(name), type(env.model.body(0))), name
    for name in SCENE_GEOMS:
        assert isinstance(env.model.geom(name), type(env.model.geom(0))), name
    assert isinstance(env.model.joint("drawer_slide"), type(env.model.joint(0)))

    assert isinstance(env._handle_geom_id, int) and env._handle_geom_id >= 0
    assert isinstance(env._drawer_qpos_adr, int) and env._drawer_qpos_adr >= 0


def test_add_drawer_scene_builds_spec_objects():
    """The same check one level earlier, on what add_drawer_scene hands the compiler. A geom whose
    type was never set still compiles -- into a sphere -- so the builder is worth checking before
    the model is."""
    spec = mujoco.MjSpec.from_file(XML_PATH)
    add_drawer_scene(spec)

    body_cls = getattr(mujoco, "MjsBody", None)
    geom_cls = getattr(mujoco, "MjsGeom", None)
    if body_cls is None or geom_cls is None:
        pytest.skip("these mujoco bindings do not expose the MjsBody/MjsGeom types")

    built = {b.name: b for b in spec.bodies}
    for name in ("cabinet", "drawer"):
        assert name in built, f"add_drawer_scene did not create a body named {name!r}"
        assert isinstance(built[name], body_cls)

    geoms = {g.name: g for g in spec.geoms}
    for name in SCENE_GEOMS:
        assert name in geoms, f"add_drawer_scene did not create a geom named {name!r}"
        assert isinstance(geoms[name], geom_cls)
        assert geoms[name].type == mujoco.mjtGeom.mjGEOM_BOX, f"{name} is not a box"


# ------------------------------------- XML / Python parity (Drawer only)

@pytest.mark.skipif(
    not DRAWER_XML.exists(),
    reason="panda_drawer_open.xml is not linked beside scene.xml; run install_env_wrapper.sh",
)
def test_reference_xml_matches_add_drawer_scene():
    """The cost of keeping menagerie_tasks/panda_drawer_open.xml as a readable reference.

    add_drawer_scene is what runs; the MJCF is documentation. Two sources of truth drift silently,
    which is exactly what render_scene.py:117 warns about for Push's TABLE_*/CUBE_* constants. This
    compiles both and compares everything add_drawer_scene sets.
    """
    from_python = load_model(XML_PATH, add_bodies=add_drawer_scene)
    from_xml = mujoco.MjModel.from_xml_path(str(DRAWER_XML))

    for field in ("nq", "nv", "njnt", "nu", "nbody", "ngeom"):
        assert getattr(from_python, field) == getattr(from_xml, field), f"{field} differs"

    data_python, data_xml = mujoco.MjData(from_python), mujoco.MjData(from_xml)
    mujoco.mj_forward(from_python, data_python)
    mujoco.mj_forward(from_xml, data_xml)

    for name in SCENE_GEOMS:
        ids = []
        for model in (from_python, from_xml):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert gid >= 0, f"no geom {name!r} in one of the two models"
            ids.append(gid)
        py_id, xml_id = ids

        assert from_python.geom_type[py_id] == from_xml.geom_type[xml_id], f"{name}: type"
        for field in ("geom_size", "geom_rgba", "geom_friction"):
            np.testing.assert_allclose(
                getattr(from_python, field)[py_id],
                getattr(from_xml, field)[xml_id],
                atol=1e-12,
                err_msg=f"{name}: {field}",
            )
        # World position, not geom_pos: it composes the body's pose with the geom's offset, so a
        # body declared at the wrong place cannot hide behind a compensating local offset.
        np.testing.assert_allclose(
            data_python.geom_xpos[py_id],
            data_xml.geom_xpos[xml_id],
            atol=1e-12,
            err_msg=f"{name}: world position",
        )

    for name in ("cabinet", "drawer"):
        py_id = mujoco.mj_name2id(from_python, mujoco.mjtObj.mjOBJ_BODY, name)
        xml_id = mujoco.mj_name2id(from_xml, mujoco.mjtObj.mjOBJ_BODY, name)
        assert py_id >= 0 and xml_id >= 0, f"no body {name!r} in one of the two models"
        assert from_python.body_mass[py_id] == pytest.approx(
            from_xml.body_mass[xml_id]
        ), f"{name}: mass"

    py_jid = mujoco.mj_name2id(from_python, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    xml_jid = mujoco.mj_name2id(from_xml, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    assert py_jid >= 0 and xml_jid >= 0, "no drawer_slide joint in one of the two models"

    assert from_python.jnt_type[py_jid] == from_xml.jnt_type[xml_jid], "slide: type"
    assert from_python.jnt_limited[py_jid] == from_xml.jnt_limited[xml_jid], "slide: limited"
    assert from_python.jnt_qposadr[py_jid] == from_xml.jnt_qposadr[xml_jid], "slide: qposadr"
    np.testing.assert_allclose(
        from_python.jnt_axis[py_jid], from_xml.jnt_axis[xml_jid], atol=1e-12, err_msg="slide: axis"
    )
    np.testing.assert_allclose(
        from_python.jnt_range[py_jid], from_xml.jnt_range[xml_jid], atol=1e-12, err_msg="slide: range"
    )

    py_dof = from_python.jnt_dofadr[py_jid]
    xml_dof = from_xml.jnt_dofadr[xml_jid]
    assert from_python.dof_damping[py_dof] == pytest.approx(
        from_xml.dof_damping[xml_dof]
    ), "slide: damping"
    assert from_python.dof_frictionloss[py_dof] == pytest.approx(
        from_xml.dof_frictionloss[xml_dof]
    ), "slide: frictionloss"
