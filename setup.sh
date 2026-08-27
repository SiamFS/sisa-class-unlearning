#!/usr/bin/env bash
# Bootstraps the project's virtual environment (see setup_env.py).
# Usage: ./setup.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/setup_env.py"
elif command -v python >/dev/null 2>&1; then
    python "$SCRIPT_DIR/setup_env.py"
else
    echo "Python was not found on PATH. Install Python 3.9+ and re-run this script." >&2
    exit 1
fi
