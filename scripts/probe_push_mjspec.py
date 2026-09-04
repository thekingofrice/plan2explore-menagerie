#!/usr/bin/env python
"""Build §15's scene from Python instead of from panda_push.xml, and check the models match.

Reach dropped its task MJCF in d2ba577 and has loaded Menagerie's scene.xml directly ever since;
Push reintroduced one. This builds the same table and cube with MjSpec, the way render_scene already
attaches the camera and marker, so Push can drop its MJCF and its symlink into the pinned clone too.

--diff compiles both and compares every model field. An empty list means the two are the same model,
so switching costs no dynamics change and no restart. --steps then runs the random-action probe on
the MjSpec-built model, for comparison against the XML-built one's fault rate.

Usage:
    MUJOCO_GL=egl python scripts/probe_push_mjspec.py --diff
    MUJOCO_GL=egl python scripts/probe_push_mjspec.py --steps 700000 --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "menagerie_integration"))

from render_scene import add_push_scene, measure_scene  # noqa: E402

MENAGERIE = REPO_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
SCENE_XML = MENAGERIE / "scene.xml"
PUSH_XML = MENAGERIE / "panda_push.xml"  # the symlink install_env_wrapper.sh places

CUBE_GEOM, TABLE_GEOM = "cube_geom", "table_top"


def build(base_height: float = 0.0) -> mujoco.MjModel:
    """scene.xml + §15's bodies, with the base optionally raised, all from Python."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    add_push_scene(spec)
    if base_height:
        base = spec.body("link0")
        base.pos = [base.pos[0], base.pos[1], base.pos[2] + base_height]
    return spec.compile()


def compare() -> int:
    """Field-by-field diff of the MjSpec-built model against the panda_push.xml one."""
    if not PUSH_XML.is_file():
        print(f"missing {PUSH_XML}; run scripts/install_env_wrapper.sh", file=sys.stderr)
        return 1

    # Measured off a plain compile, exactly as the wrapper does, so both models are raised equally.
    _, table_top = measure_scene(
        mujoco.MjModel.from_xml_path(str(PUSH_XML)), CUBE_GEOM, TABLE_GEOM
    )

    from_xml = mujoco.MjModel.from_xml_path(str(PUSH_XML))
    spec_built = build()
    print(f"table_top={table_top}")
    print("nbody/ngeom/nq/nv/njnt/nu")
    for tag, m in (("xml ", from_xml), ("spec", spec_built)):
        print(f"  {tag}: {m.nbody} {m.ngeom} {m.nq} {m.nv} {m.njnt} {m.nu}")

    diffs = []
    for name in dir(from_xml):
        if name.startswith("_"):
            continue
        try:
            a, b = getattr(from_xml, name), getattr(spec_built, name)
        except Exception:  # noqa: BLE001 - a field that will not read is not a difference
            continue
        if isinstance(a, np.ndarray):
            if a.shape != b.shape or not np.array_equal(a, b):
                diffs.append(name)
        elif isinstance(a, (int, float, bool)) and a != b:
            diffs.append(name)

    print(f"differing fields ({len(diffs)}): {diffs}")
    if not diffs:
        print("\nidentical -- switching Push to the MjSpec scene changes no dynamics.")
    return 0


def probe(steps: int, seed: int, use_xml: bool) -> int:
    """Drive the real wrapper, changing only how its model is built.

    The reset logic -- home keyframe, joint jitter, cube placement with a valid quaternion, the
    200-step episode boundary -- is the wrapper's own, not a second copy that could drift from it.
    Only load_model is swapped, so the two arms of the comparison differ in the model and nothing
    else. Without this the probe would step a zeroed qpos, which gives the free joint a (0,0,0,0)
    quaternion: not a rotation, and not a state the environment ever produces.
    """
    import menagerie_panda_push as wrapper

    original = wrapper.load_model

    def spec_load(xml_path, base_height=0.0, render=False, **_):
        if render:  # the probe never renders; falling back keeps the branch honest if it ever does
            return original(xml_path, base_height=base_height, render=render)
        return build(base_height=base_height)

    if not use_xml:
        wrapper.load_model = spec_load
    try:
        env = wrapper.MenageriePandaPush(seed=seed)
        source = "panda_push.xml" if use_xml else "MjSpec"
        print(f"model from {source}: nq={env.model.nq} nv={env.model.nv} "
              f"ngeom={env.model.ngeom} n_substeps={env.n_substeps}", flush=True)

        rng = np.random.default_rng(seed)
        for step in range(steps):
            _, _, _, truncated, _ = env.step(
                rng.uniform(-1, 1, env.model.nu).astype(np.float32)
            )
            if truncated:
                env.reset()
            if step % 20_000 == 0:
                print(f"step {step} ncon={env.data.ncon} nefc={env.data.nefc}", flush=True)
    finally:
        wrapper.load_model = original

    print(f"survived {steps} steps ({steps * env.n_substeps} mj_step calls)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diff", action="store_true", help="compare the two models and exit")
    p.add_argument("--xml", action="store_true", help="probe the panda_push.xml model instead")
    p.add_argument("--steps", type=int, default=700_000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    return compare() if args.diff else probe(args.steps, args.seed, args.xml)


if __name__ == "__main__":
    raise SystemExit(main())
