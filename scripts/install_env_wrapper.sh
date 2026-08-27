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

# Prove the import path Hydra's _target_ will use actually resolves.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  log "verifying sheeprl.envs.menagerie_panda imports"
  venv_python -c "
from sheeprl.envs.menagerie_panda import MenageriePandaReach
e = MenageriePandaReach()
print(f'  ok: obs {e.observation_space[\"state\"].shape}, act {e.action_space.shape}, '
      f'n_substeps {e.n_substeps}')
" || die "import failed -- _target_ in menagerie_panda_reach.yaml will not resolve"
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
