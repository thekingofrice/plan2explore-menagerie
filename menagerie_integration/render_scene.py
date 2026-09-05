"""Cosmetic render assets attached at load time with mujoco.MjSpec.

Implements: References/SelfEx-WM_Notes.tex §9 rendering, for §13.1's evaluation videos.

ENVIRONMENT_SPEC.md §1 deferred a fixed camera and a visible target as "purely cosmetic ... until
§10's render test needs a viewpoint, at which point mujoco.MjSpec can attach them to the loaded
model from Python". This is that. Neither task's MJCF changes, and the same code serves Reach (which
has no task file) and Push (which has one), so §16's other arms need no new mechanism.

Both additions are provably safe to attach mid-experiment: a site is massless with no collision
geometry, a camera has no physics at all. nq, nv, nu, the observation, the reward and the contact set
are identical either way, so a checkpoint trained without them evaluates identically with them.

Attached only when render_mode is not None. Training runs render nothing, so they never take the
MjSpec path at all.
"""

from __future__ import annotations

import mujoco
import numpy as np

CAMERA_NAME = "workspace"
TARGET_SITE = "target_marker"

#: Looks down at the region both tasks operate in, from front-right and above.
DEFAULT_CAMERA_EYE = (1.30, -0.95, 0.95)
DEFAULT_CAMERA_LOOK_AT = (0.45, 0.0, 0.30)

#: Radius of the goal marker, metres. Comfortably smaller than §8.1's 5 cm success tolerance, so a
#: viewer can tell "inside the tolerance" from "touching the marker".
TARGET_RADIUS = 0.02
TARGET_RGBA = (0.90, 0.20, 0.20, 0.55)

#: Parked below the floor until the first reset moves it. A marker at the origin would read as a
#: real goal in any frame captured before then.
SITE_PARK = (0.0, 0.0, -1.0)


def _look_at_quat(eye, look_at, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Orientation putting `look_at` on the camera's view axis.

    MuJoCo cameras look down their own -z with +y up, so the rotation's third column is the negated
    forward direction rather than the forward direction itself.
    """
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(look_at, dtype=float) - eye
    forward /= np.linalg.norm(forward)

    z_axis = -forward
    right = np.cross(np.asarray(up, dtype=float), z_axis)
    right /= np.linalg.norm(right)
    up_axis = np.cross(z_axis, right)

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, up_axis, z_axis]).ravel())
    return quat


#: Menagerie names the Panda's welded base link0. Elevating it is how Panda Push mounts the arm at
#: work-surface height, which MJCF cannot express: `<include>` splices at the element level, and an
#: included robot cannot be wrapped in a repositioned <body> because panda.xml carries <asset> and
#: <default> blocks that are not legal inside one.
BASE_BODY = "link0"


def load_model(
    xml_path: str,
    base_height: float = 0.0,
    render: bool = False,
    camera_eye=DEFAULT_CAMERA_EYE,
    camera_look_at=DEFAULT_CAMERA_LOOK_AT,
    target_radius: float = TARGET_RADIUS,
    add_bodies=None,
) -> mujoco.MjModel:
    """Compile `xml_path`, optionally adding task bodies, raising the robot base, and adding
    render assets.

    Falls through to a plain from_xml_path when none of the three is asked for, so a task that needs
    none never depends on MjSpec at all.

    `add_bodies` is a callable taking the spec, for a task that builds its scene in Python rather
    than from a task MJCF -- see add_drawer_scene. It runs FIRST, so the task's bodies append to
    worldbody directly after the robot, exactly where a task MJCF's own <worldbody> would put them.
    That ordering is what fixes their qpos addresses, so the two routes must not disagree on it.

    Loading through MjSpec keeps every relative include resolving from the file's own directory,
    exactly as a direct load would -- which is what panda_push.xml's `<include file="scene.xml"/>`
    depends on.
    """
    if base_height == 0.0 and not render and add_bodies is None:
        return mujoco.MjModel.from_xml_path(str(xml_path))

    spec = mujoco.MjSpec.from_file(str(xml_path))

    if add_bodies is not None:
        add_bodies(spec)

    if base_height:
        try:
            base = spec.body(BASE_BODY)
        except (KeyError, ValueError) as exc:
            names = [b.name for b in spec.bodies]
            raise ValueError(
                f"no body named {BASE_BODY!r} to raise; model has {names}"
            ) from exc
        base.pos = [base.pos[0], base.pos[1], base.pos[2] + base_height]

    if not render:
        return spec.compile()

    cam = spec.worldbody.add_camera()
    cam.name = CAMERA_NAME
    cam.pos = list(camera_eye)
    cam.quat = _look_at_quat(camera_eye, camera_look_at).tolist()

    site = spec.worldbody.add_site()
    site.name = TARGET_SITE
    site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    site.size = [target_radius, 0.0, 0.0]
    site.rgba = list(TARGET_RGBA)
    site.pos = list(SITE_PARK)

    return spec.compile()


#: §15's table and cube, as declared in menagerie_tasks/panda_push.xml. Restated here because
#: add_push_scene builds them from Python instead of from that file; the two must agree exactly,
#: which scripts/probe_push_mjspec.py checks field by field.
TABLE_POS = (0.5, 0.0, 0.11)
TABLE_SIZE = (0.30, 0.40, 0.11)
TABLE_RGBA = (0.55, 0.45, 0.35, 1.0)
CUBE_POS = (0.45, 0.0, 0.245)
CUBE_SIZE = (0.025, 0.025, 0.025)
CUBE_MASS = 0.05
CUBE_RGBA = (0.85, 0.2, 0.2, 1.0)
SCENE_FRICTION = (1.0, 0.005, 0.0001)


def add_push_scene(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add §15's table and free cube to a spec, the way Reach adds its camera and marker.

    The alternative is menagerie_tasks/panda_push.xml, which must be symlinked beside Menagerie's
    scene.xml for its <include> to resolve -- the arrangement Reach abandoned in d2ba577. Building
    the bodies here needs no task MJCF, no symlink inside the pinned clone, and no per-arm file.

    Table before cube, matching the MJCF's element order: body order fixes qpos addresses and
    contact ordering, so a swap would compile a different model.
    """
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = list(TABLE_POS)
    top = table.add_geom()
    top.name = "table_top"
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = list(TABLE_SIZE)
    top.rgba = list(TABLE_RGBA)
    top.friction = list(SCENE_FRICTION)

    cube = spec.worldbody.add_body()
    cube.name = "cube"
    cube.pos = list(CUBE_POS)
    try:
        joint = cube.add_freejoint()
    except AttributeError:  # older bindings expose only the generic adder
        joint = cube.add_joint()
        joint.type = mujoco.mjtJoint.mjJNT_FREE
    joint.name = "cube_free"

    body = cube.add_geom()
    body.name = "cube_geom"
    body.type = mujoco.mjtGeom.mjGEOM_BOX
    body.size = list(CUBE_SIZE)
    body.mass = CUBE_MASS
    body.rgba = list(CUBE_RGBA)
    body.friction = list(SCENE_FRICTION)

    return spec


#: Drawer Open's cabinet and drawer, as declared in menagerie_tasks/panda_drawer_open.xml. That file
#: is kept as the readable reference; this is what actually compiles. The two must agree exactly --
#: the same two-sources-of-truth risk the TABLE_*/CUBE_* block above carries for Push.
#:
#: Cabinet is welded in mid-air: a body with no joint is fixed to the world at whatever height it is
#: declared, so nothing has to hold it up. Cavity spans x [0.52, 0.72], y [-0.11, 0.11], z [0.30, 0.42].
CABINET_POS = (0.62, 0.0, 0.36)
CABINET_RGBA = (0.55, 0.45, 0.35, 1.0)
#: (name, pos, size) per panel: top, back, and two sides. No bottom and no front -- the slide joint
#: holds the drawer up, so a shelf would only add contacts.
CABINET_PANELS = (
    ("cabinet_top", (0.0, 0.0, 0.065), (0.10, 0.12, 0.005)),
    ("cabinet_back", (0.105, 0.0, 0.0), (0.005, 0.12, 0.06)),
    ("cabinet_side_left", (0.0, 0.115, 0.0), (0.10, 0.005, 0.06)),
    ("cabinet_side_right", (0.0, -0.115, 0.0), (0.10, 0.005, 0.06)),
)

#: Declared at the closed pose, coincident with the cabinet.
DRAWER_POS = (0.62, 0.0, 0.36)
#: Axis is -x so qpos IS the opening in metres: 0 closed, 0.12 fully out. menagerie_panda_drawer_open
#: reads the range off the compiled model rather than repeating 0.12, so this stays the only source.
DRAWER_AXIS = (-1.0, 0.0, 0.0)
DRAWER_RANGE = (0.0, 0.12)
DRAWER_DAMPING = 5.0
DRAWER_FRICTIONLOSS = 2.0
DRAWER_SIZE = (0.09, 0.10, 0.05)
#: Explicit: at MuJoCo's default density this slab would mass 3.6 kg, above the Panda's rated payload.
DRAWER_MASS = 0.5
DRAWER_RGBA = (0.45, 0.38, 0.30, 1.0)

#: Handle on a stem, on the face pointing at the robot. The 0.035 m gap between the drawer face
#: (local x = -0.09) and the handle's rear face (local x = -0.125) is the point: a finger slots in
#: beside the stem and pulling in -x catches the cube, so opening does not need a closed grasp. The
#: cube is 3 cm, well inside the Panda's 8 cm finger opening, so grasping stays available.
HANDLE_STEM_POS = (-0.1075, 0.0, 0.0)
HANDLE_STEM_SIZE = (0.0175, 0.008, 0.008)
HANDLE_STEM_RGBA = (0.30, 0.30, 0.32, 1.0)
HANDLE_POS = (-0.14, 0.0, 0.0)
HANDLE_SIZE = (0.015, 0.015, 0.015)
HANDLE_RGBA = (0.85, 0.2, 0.2, 1.0)


def add_drawer_scene(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add Drawer Open's cabinet and drawer to a spec, the way add_push_scene adds §15's.

    Pass as `load_model(..., add_bodies=add_drawer_scene)`. Unlike Push this is the only route --
    there is no symlink into the pinned Menagerie clone and no include-resolution constraint.

    Cabinet before drawer, matching the MJCF's element order: body order fixes qpos addresses, so a
    swap would compile a different model.
    """
    cabinet = spec.worldbody.add_body()
    cabinet.name = "cabinet"
    cabinet.pos = list(CABINET_POS)
    for name, pos, size in CABINET_PANELS:
        panel = cabinet.add_geom()
        panel.name = name
        panel.type = mujoco.mjtGeom.mjGEOM_BOX
        panel.pos = list(pos)
        panel.size = list(size)
        panel.rgba = list(CABINET_RGBA)
        panel.friction = list(SCENE_FRICTION)

    drawer = spec.worldbody.add_body()
    drawer.name = "drawer"
    drawer.pos = list(DRAWER_POS)

    joint = drawer.add_joint()
    joint.name = "drawer_slide"
    joint.type = mujoco.mjtJoint.mjJNT_SLIDE
    joint.axis = list(DRAWER_AXIS)
    joint.range = list(DRAWER_RANGE)
    # Set explicitly rather than trusting panda.xml's autolimits="true" to survive the spec
    # round-trip. Without it the range is ignored and the drawer slides out of the world.
    joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.damping = DRAWER_DAMPING
    joint.frictionloss = DRAWER_FRICTIONLOSS

    slab = drawer.add_geom()
    slab.name = "drawer_body"
    slab.type = mujoco.mjtGeom.mjGEOM_BOX
    slab.size = list(DRAWER_SIZE)
    slab.mass = DRAWER_MASS
    slab.rgba = list(DRAWER_RGBA)
    slab.friction = list(SCENE_FRICTION)

    for name, pos, size, rgba in (
        ("handle_stem", HANDLE_STEM_POS, HANDLE_STEM_SIZE, HANDLE_STEM_RGBA),
        ("handle", HANDLE_POS, HANDLE_SIZE, HANDLE_RGBA),
    ):
        geom = drawer.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.pos = list(pos)
        geom.size = list(size)
        geom.rgba = list(rgba)
        geom.friction = list(SCENE_FRICTION)

    return spec


def measure_scene(model: mujoco.MjModel, cube_geom: str, table_geom: str) -> tuple[float, float]:
    """Cube half-extent and table-top height, read off a compiled model.

    Both geoms are boxes, so geom_size holds half-extents per axis. One mj_forward puts world
    positions in geom_xpos, which handles the table being nested in a body without composing
    body_pos and geom_pos by hand.

    Lives here rather than on the wrapper because the wrapper needs these values BEFORE it can
    build its real model -- the base is raised by the table height.
    """
    ids = {}
    for name in (cube_geom, table_geom):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"model has no geom named {name!r}")
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"geom {name!r} must be a box for its half-extent to be read")
        ids[name] = gid

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cube_half = float(model.geom_size[ids[cube_geom]][2])
    table_top = float(
        data.geom_xpos[ids[table_geom]][2] + model.geom_size[ids[table_geom]][2]
    )
    return cube_half, table_top


def target_site_id(model: mujoco.MjModel) -> int:
    """Site index, or -1 when the model was loaded without render assets."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE)


def move_target_site(model: mujoco.MjModel, site_id: int, position) -> None:
    """Put the marker at `position`.

    Written to model.site_pos, not data.site_xpos: the latter is derived and mj_forward would
    overwrite it. The site hangs off worldbody, so its local frame is the world frame.
    """
    if site_id >= 0:
        model.site_pos[site_id] = np.asarray(position, dtype=float)
