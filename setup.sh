#!/usr/bin/env bash
#
# setup.sh — repo bootstrap for mimic-eicu-record-density
#
# What this does (idempotent — safe to re-run):
#   1. Create the directory skeleton documented in README
#   2. Create empty .gitkeep markers so empty dirs are tracked
#   3. Validate Python environment
#   4. Optionally install dependencies into a local venv
#   5. Run a pre-commit safety scan for accidentally staged patient data
#
# Usage:
#   bash setup.sh             # bootstrap only
#   bash setup.sh --install   # bootstrap + create venv + pip install
#   bash setup.sh --check     # bootstrap + run PhysioNet data safety scan only
#
# Exit codes:
#   0  success
#   1  environment validation failed
#   2  PhysioNet data hygiene scan failed (HARD STOP — do not push)

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours (only when stdout is a TTY)
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BLUE=''; RESET=''
fi

info()    { echo "${BLUE}[info]${RESET}    $*"; }
ok()      { echo "${GREEN}[ok]${RESET}      $*"; }
warn()    { echo "${YELLOW}[warn]${RESET}    $*"; }
err()     { echo "${RED}[error]${RESET}   $*" >&2; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
INSTALL=0
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    --check)   CHECK_ONLY=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *)
      err "Unknown argument: $arg"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Directory skeleton
# ---------------------------------------------------------------------------
if [[ "$CHECK_ONLY" -eq 0 ]]; then
  info "Creating directory skeleton..."
  mkdir -p cohort analysis figures docs outputs fixtures figures/published

  # .gitkeep markers (so empty dirs are tracked)
  for d in cohort analysis figures docs fixtures figures/published; do
    [[ -f "$d/.gitkeep" ]] || touch "$d/.gitkeep"
  done

  # outputs/ is .gitignore'd; do NOT put a .gitkeep there
  ok "Directory skeleton ready."
fi

# ---------------------------------------------------------------------------
# 2. Python environment validation
# ---------------------------------------------------------------------------
if [[ "$CHECK_ONLY" -eq 0 ]]; then
  info "Validating Python..."
  if ! command -v python3 >/dev/null 2>&1; then
    err "python3 not found. Install Python 3.10+ before continuing."
    exit 1
  fi

  PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  PYMAJOR=$(echo "$PYVER" | cut -d. -f1)
  PYMINOR=$(echo "$PYVER" | cut -d. -f2)
  if (( PYMAJOR < 3 )) || ( (( PYMAJOR == 3 )) && (( PYMINOR < 10 )) ); then
    err "Python 3.10+ required; found $PYVER."
    exit 1
  fi
  ok "Python $PYVER detected."
fi

# ---------------------------------------------------------------------------
# 3. Optional venv + install
# ---------------------------------------------------------------------------
if [[ "$INSTALL" -eq 1 ]]; then
  info "Creating venv at .venv ..."
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    ok "Created .venv"
  else
    info ".venv already exists; reusing."
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate
  info "Upgrading pip..."
  pip install --quiet --upgrade pip

  if [[ -f requirements.txt ]]; then
    info "Installing requirements.txt ..."
    pip install --quiet -r requirements.txt
    ok "Dependencies installed."
  else
    warn "requirements.txt not found; skipping install."
  fi
fi

# ---------------------------------------------------------------------------
# 4. PhysioNet data hygiene scan
# ---------------------------------------------------------------------------
info "Running PhysioNet data hygiene scan..."

# Identifier column names that, if found in a tracked file, suggest patient
# data may have been accidentally added. Add to this list if other identifier
# columns are introduced in future analyses.
PATTERNS=(
  "subject_id"
  "hadm_id"
  "stay_id"
  "patientunitstayid"
  "patienthealthsystemstayid"
  "uniquepid"
  "anchor_year"
  "anchor_age"
)

# Build a single grep -E pattern
PATTERN=$(IFS='|'; echo "${PATTERNS[*]}")

# Look ONLY in files that would be staged for commit (respects .gitignore).
# Skip if not inside a git working tree yet.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  HITS=$(git ls-files --cached --others --exclude-standard 2>/dev/null \
    | grep -E '\.(csv|tsv|parquet|json|txt|md|py|ipynb|yaml|yml)$' \
    | xargs grep -l -E "$PATTERN" 2>/dev/null || true)

  if [[ -n "$HITS" ]]; then
    err "PhysioNet identifier patterns detected in tracked files:"
    echo "$HITS" | sed 's/^/   /'
    err ""
    err "HARD STOP. Do NOT commit or push until these files are reviewed."
    err "These may contain patient identifiers that cannot be redistributed."
    err "Either remove the files, scrub the identifier columns, or add them"
    err "to .gitignore. Then re-run: bash setup.sh --check"
    exit 2
  fi
  ok "No PhysioNet identifier patterns found in tracked files."
else
  warn "Not yet a git repo; skipping the tracked-files scan."
  warn "Run 'git init' first, then re-run: bash setup.sh --check"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
ok "Bootstrap complete."
echo
echo "Next steps:"
echo "  1. Verify cohort/config.yaml points at your local PhysioNet data paths"
echo "  2. If not done yet:  git init && git add -A && git commit -m 'Initial commit'"
echo "  3. To install deps:  bash setup.sh --install"
echo "  4. Before every push: bash setup.sh --check"
echo
