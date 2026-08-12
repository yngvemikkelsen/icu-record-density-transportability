#!/usr/bin/env bash
#
# Environment setup for the ICU record-density analysis.
#
# Two paths. If uv is available it is used, because the scripts carry PEP 723
# inline dependency metadata and uv resolves them per script with no shared
# environment to drift. Otherwise a plain venv is created from requirements.txt.
#
#   ./setup.sh
#
# This script installs software only. It does not download, extract or touch
# any data: MIMIC-IV and eICU-CRD require credentialed PhysioNet access and
# their own data use agreements, and nothing here can or should automate that.

set -euo pipefail

echo "ICU record-density analysis — environment setup"
echo

if command -v uv >/dev/null 2>&1; then
    echo "uv found ($(uv --version))."
    echo "No setup is required: each script declares its own dependencies."
    echo
    echo "Run any script directly, for example:"
    echo "    uv run scripts/paper17_diagnose_exposure.py --help"
else
    echo "uv not found. Creating a virtual environment from requirements.txt."
    echo "(Installing uv instead is recommended: https://docs.astral.sh/uv/)"
    echo

    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
        echo "ERROR: Python 3.11 or later is required. Found $(python3 -V)."
        exit 1
    }

    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt

    echo "Done. Activate with:"
    echo "    source .venv/bin/activate"
    echo
    echo "Then run, for example:"
    echo "    python scripts/paper17_diagnose_exposure.py --help"
fi

echo
echo "-------------------------------------------------------------------"
echo "Data is NOT included and cannot be redistributed. Both databases"
echo "require credentialed access through PhysioNet:"
echo
echo "  MIMIC-IV v3.1   https://physionet.org/content/mimiciv/3.1/"
echo "  eICU-CRD v2.0   https://physionet.org/content/eicu-crd/2.0/"
echo
echo "Scripts take --mimic-root and --eicu-root pointing at your own local"
echo "extractions. Do not commit extracted data: .gitignore excludes the"
echo "parquet caches, but check 'git status' before committing."
echo "-------------------------------------------------------------------"
