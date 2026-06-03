#!/usr/bin/env bash
# macOS launcher panel. Creates the venv and installs requirements with
# `uv` on first run (or whenever requirements.txt is newer than the
# install stamp), then opens the launcher panel.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
SIM="$ROOT/simulated_world"
VENV="$SIM/.venv"
PY="$VENV/bin/python"
REQ="$SIM/requirements.txt"
STAMP="$VENV/.requirements.stamp"

# uv is required for venv + install. Add the standard install paths to
# PATH first so Finder-launched .command (which doesn't source the user
# shell rc) can still find it.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] 'uv' not found on PATH."
    echo "Install with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    read -n 1 -s -r -p "Press any key to continue..."
    echo
    exit 1
fi

# Create venv if it doesn't exist.
if [ ! -x "$PY" ]; then
    echo "[setup] Creating Python 3.12 venv at $VENV ..."
    if ! uv venv --python 3.12 "$VENV"; then
        read -n 1 -s -r -p "Press any key to continue..."
        echo
        exit 1
    fi
fi

# Install / sync requirements if missing or stale.
if [ ! -f "$STAMP" ] || [ "$REQ" -nt "$STAMP" ]; then
    echo "[setup] Installing requirements from $(basename "$REQ") ..."
    if ! VIRTUAL_ENV="$VENV" uv pip install -r "$REQ"; then
        echo "[ERROR] uv pip install failed."
        read -n 1 -s -r -p "Press any key to continue..."
        echo
        exit 1
    fi
    touch "$STAMP"
fi

cd "$SIM"
"$PY" "$SIM/launcher.py" "$@"
status=$?
if [ "$status" -ne 0 ]; then
    read -n 1 -s -r -p "Press any key to continue..."
    echo
fi
exit "$status"
