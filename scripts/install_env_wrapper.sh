#!/usr/bin/env bash
# install_env_wrapper.sh
#
# Implements: References/SelfEx-WM_Notes.tex §11.1 Phase 5 Option A.
#
#   "Follow SheepRL's documented custom-environment pattern and add:
#        sheeprl/envs/menagerie_panda.py
#        sheeprl/configs/env/menagerie_panda_reach.yaml"
#
# Both files must live inside the SheepRL tree for `_target_: sheeprl.envs.menagerie_panda...` to
# import. That checkout is a pinned clone and is gitignored, so the canonical copies live in
# menagerie_integration/ and are symlinked into place here. One source of truth, and editing either
# location edits the same file.
#
# §3 is untouched: sheeprl/algos/p2e_dv3/ and sheeprl/algos/dreamer_v3/ are never written to.
# sheeprl/envs/ and sheeprl/configs/env/ are both on the allowed-modification list.
#
# Usage:
#     bash scripts/install_env_wrapper.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_clones

SRC_DIR="${REPO_ROOT}/menagerie_integration"
ENV_DEST="${SHEEPRL_DIR}/sheeprl/envs/menagerie_panda.py"
CFG_DEST="${SHEEPRL_DIR}/sheeprl/configs/env/menagerie_panda_reach.yaml"

# §15 Panda Push. The wrapper and config go into the SheepRL checkout like Reach's, but the task
# MJCF goes into the MENAGERIE checkout: panda_push.xml contains `<include file="scene.xml"/>`, and
# MuJoCo resolves that relative to the file being loaded. Only as a sibling of scene.xml does the
# include work -- from anywhere else MuJoCo compounds base directories and mis-resolves scene.xml's
# own `<include file="panda.xml"/>`. See ENVIRONMENT_SPEC.md §1.
#
# The MJCF's source is menagerie_tasks/, per §5's repo layout; the wrapper and config keep the
# menagerie_integration/ home they share with Reach's.
TASKS_DIR="${REPO_ROOT}/menagerie_tasks"
PUSH_ENV_DEST="${SHEEPRL_DIR}/sheeprl/envs/menagerie_panda_push.py"
PUSH_CFG_DEST="${SHEEPRL_DIR}/sheeprl/configs/env/menagerie_panda_push.yaml"
PUSH_XML_DEST="${MENAGERIE_DIR}/franka_emika_panda/panda_push.xml"

# ---------------------------------------------------------------------------
# link_file <source> <destination>
#
# Symlink, falling back to a copy on filesystems that refuse links. A copy is a silent fork, so it
# warns loudly and records that the file must be reinstalled after every edit.
# ---------------------------------------------------------------------------
link_file() {
  local src="$1" dest="$2"

  [[ -f "${src}" ]] || die "missing source file: ${src}"
  mkdir -p "$(dirname "${dest}")"

  if [[ -L "${dest}" ]]; then
    local current
    current="$(readlink -f "${dest}")"
    if [[ "${current}" == "$(readlink -f "${src}")" ]]; then
      log "already linked: ${dest#"${REPO_ROOT}/"}"
      return
    fi
    warn "replacing symlink that pointed at ${current}"
    rm -f "${dest}"
  elif [[ -e "${dest}" ]]; then
    # A real file here is either a stale copy from this script or something upstream shipped.
    # Neither should be clobbered without a trace.
    local backup="${dest}.bak.$(date -u +%Y%m%dT%H%M%SZ)"
    warn "${dest} exists and is not a symlink -- backing up to ${backup}"
    mv "${dest}" "${backup}"
  fi

  if ln -s "${src}" "${dest}" 2>/dev/null; then
    log "linked ${dest#"${SHEEPRL_DIR}/"} -> ${src#"${REPO_ROOT}/"}"
  else
    cp "${src}" "${dest}"
    warn "symlink failed; COPIED instead. ${dest#"${SHEEPRL_DIR}/"} is now a fork --"
    warn "re-run this script after every edit to ${src#"${REPO_ROOT}/"}"
  fi
}

log "Phase 5 Option A: installing the environment wrapper into the SheepRL checkout (notes §11.1)"

link_file "${SRC_DIR}/menagerie_panda.py" "${ENV_DEST}"
link_file "${SRC_DIR}/menagerie_panda_reach.yaml" "${CFG_DEST}"

# render_scene.py needs no link of its own: both wrappers reach it through
# Path(__file__).resolve().parent, which follows their symlinks back to menagerie_integration/.

link_file "${SRC_DIR}/menagerie_panda_push.py" "${PUSH_ENV_DEST}"
link_file "${SRC_DIR}/menagerie_panda_push.yaml" "${PUSH_CFG_DEST}"
link_file "${TASKS_DIR}/panda_push.xml" "${PUSH_XML_DEST}"

# The link above is the only thing this repository puts inside the pinned Menagerie clone. No
# tracked file is touched, so check_pins()'s `git rev-parse HEAD` still matches -- but the link
# would show as untracked and read like drift, so exclude it locally.
MENAGERIE_EXCLUDE="${MENAGERIE_DIR}/.git/info/exclude"
if [[ -d "$(dirname "${MENAGERIE_EXCLUDE}")" ]]; then
  if ! grep -qxF "franka_emika_panda/panda_push.xml" "${MENAGERIE_EXCLUDE}" 2>/dev/null; then
    echo "franka_emika_panda/panda_push.xml" >>"${MENAGERIE_EXCLUDE}"
    log "excluded franka_emika_panda/panda_push.xml from the Menagerie clone's git status"
  fi
else
  warn "no ${MENAGERIE_EXCLUDE}; the panda_push.xml link will show as untracked in Menagerie"
fi

# Prove the import path Hydra's _target_ will use actually resolves.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  log "verifying sheeprl.envs.menagerie_panda imports"
  venv_python -c "
from sheeprl.envs.menagerie_panda import MenageriePandaReach
from sheeprl.envs.menagerie_panda_push import MenageriePandaPush
for cls in (MenageriePandaReach, MenageriePandaPush):
    e = cls()
    print(f'  ok: {cls.__name__} obs {e.observation_space[\"state\"].shape}, '
          f'act {e.action_space.shape}, n_substeps {e.n_substeps}')
    e.close()
" || die "import failed -- a _target_ in one of the env configs will not resolve"
else
  warn "no virtualenv at ${VENV_DIR}; skipping the import check"
fi

cat <<EOF

Installed:
  ${ENV_DEST}
  ${CFG_DEST}

Run (notes §11.1):
  cd ${SHEEPRL_DIR}
  MUJOCO_GL=egl python sheeprl.py \\
    exp=p2e_dv3_exploration \\
    env=menagerie_panda_reach \\
    fabric.accelerator=${ACCELERATOR} \\
    fabric.devices=${DEVICES} \\
    seed=0
EOF
