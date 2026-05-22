#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
SIM="$ROOT/simulated_world"
PY="$SIM/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "[ERROR] Python venv not found at:"
    echo "  $PY"
    echo "Run uv venv inside simulated_world/ to recreate it."
    read -n 1 -s -r -p "Press any key to continue..."
    echo
    exit 1
fi

# File picker: CSV (extracted playback) OR .db3 / metadata.yaml (raw rosbag2)
INITIAL_DIR="$ROOT/data/example-playback-csv"
if [ ! -d "$INITIAL_DIR" ]; then
    INITIAL_DIR="$ROOT/data"
fi
if [ ! -d "$INITIAL_DIR" ]; then
    INITIAL_DIR="$ROOT"
fi

PICK="$(/usr/bin/osascript <<EOF
set initDir to POSIX file "$INITIAL_DIR"
try
    set theFile to choose file with prompt "Pick a playback CSV or a rosbag2 .db3 / metadata.yaml" default location initDir of type {"csv", "db3", "yaml", "yml", "public.comma-separated-values-text"}
on error number -128
    return ""
end try
return POSIX path of theFile
EOF
)"

if [ -z "$PICK" ]; then
    echo "No file selected, exiting."
    read -n 1 -s -r -p "Press any key to continue..."
    echo
    exit 1
fi

# If user picked a .db3 or metadata.yaml, the baker wants the *parent directory*.
SRC="$PICK"
NAME="$(basename "$PICK")"
EXT="${NAME##*.}"
PARENT="$(dirname "$PICK")"

shopt -s nocasematch
if [[ "$EXT" == "db3" ]] || [[ "$NAME" == "metadata.yaml" ]]; then
    SRC="$PARENT"
fi
shopt -u nocasematch

echo
echo "Picked:  $PICK"
echo "Passing: $SRC"
echo "Output:  $ROOT/data/Baked_Playbacks/<stem>_baked.mp4"
echo

cd "$SIM"
"$PY" "$SIM/bake_offscreen.py" "$SRC"
status=$?

echo
if [ "$status" -ne 0 ]; then
    echo "[Bake failed]"
else
    echo "[Done]"
fi
read -n 1 -s -r -p "Press any key to continue..."
echo
exit "$status"
