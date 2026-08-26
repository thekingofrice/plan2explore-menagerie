#!/usr/bin/env bash
# common.sh
#
# Implements: References/SelfEx-WM_Notes.tex -- shared plumbing for all scripts/*.sh.
# Not a phase of its own; sourced by every phase script.

set -euo pipefail

# --- Repository paths -------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHEEPRL_DIR="${REPO_ROOT}/sheeprl"
MENAGERIE_DIR="${REPO_ROOT}/third_party/mujoco_menagerie"
VENV_DIR="${REPO_ROOT}/.venv311"
RESULTS_DIR="${REPO_ROOT}/results"
UPSTREAM_MD="${REPO_ROOT}/UPSTREAM.md"

# --- Upstream sources (notes 6) ---------------------------------------------
SHEEPRL_URL="https://github.com/Eclectic-Sheep/sheeprl.git"
MENAGERIE_URL="https://github.com/google-deepmind/mujoco_menagerie.git"

# Leave empty on the very first clone; scripts/record_upstream.sh writes the observed SHAs into
# UPSTREAM.md, and they are pasted back here to hard-pin every subsequent checkout.
SHEEPRL_PIN="${SHEEPRL_PIN:-}"
MENAGERIE_PIN="${MENAGERIE_PIN:-}"

# --- Python (notes 7) -------------------------------------------------------
# SheepRL declares requires-python = ">=3.8,<3.12", so the venv is built with 3.11.
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

# --- Compute (target is the Linux cluster's NVIDIA GPU) ---------------------
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"

# --- Logging ----------------------------------------------------------------
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Guards -----------------------------------------------------------------

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_venv() {
  [[ -x "${VENV_DIR}/bin/python" ]] \
    || die "virtualenv missing at ${VENV_DIR} -- see README 'Phase 1' (notes §7)"
}

require_clones() {
  [[ -d "${SHEEPRL_DIR}/.git" ]] \
    || die "SheepRL not cloned -- see README 'Phase 0' (notes §6)"
  [[ -d "${MENAGERIE_DIR}/.git" ]] \
    || die "MuJoCo Menagerie not cloned -- see README 'Phase 0' (notes §6)"
}

# Run the pinned interpreter. Used instead of `source .../activate` so the scripts stay safe
# under `set -u`.
venv_python() {
  require_venv
  "${VENV_DIR}/bin/python" "$@"
}

# Every training/eval run must first prove the algorithm was untouched (notes 3).
require_algorithm_untouched() {
  bash "${REPO_ROOT}/scripts/verify_algorithm_untouched.sh" \
    || die "algorithm-purity check failed -- refusing to run"
}

# --- Run identity -----------------------------------------------------------
# Deterministic, sortable run id used for log dirs and manifest filenames.
# Usage: run_id panda reach 0  ->  20260825T143012Z_panda_reach_seed0
run_id() {
  local robot="$1" task="$2" seed="$3"
  printf '%s_%s_%s_seed%s' "$(date -u +%Y%m%dT%H%M%SZ)" "${robot}" "${task}" "${seed}"
}
