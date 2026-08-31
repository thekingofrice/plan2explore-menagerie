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


def load_model_with_render_assets(
    xml_path: str,
    camera_eye=DEFAULT_CAMERA_EYE,
    camera_look_at=DEFAULT_CAMERA_LOOK_AT,
    target_radius: float = TARGET_RADIUS,
) -> mujoco.MjModel:
    """Compile `xml_path` with a fixed camera and a goal marker added.

    Loading through MjSpec rather than from_xml_path keeps every relative include resolving from the
    file's own directory, exactly as a direct load would -- which is what panda_push.xml's
    `<include file="scene.xml"/>` depends on.
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))

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
