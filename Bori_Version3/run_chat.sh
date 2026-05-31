#!/bin/bash

# Bori (보리): Virtual Environment Wrapper & Launcher

# Find project root and virtual environment path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DISCOVERY_PATHS=(
    "$SCRIPT_DIR/../.venv"
    "$SCRIPT_DIR/../../.venv"
    "$SCRIPT_DIR/.venv"
)

VENV_ACTIVE=""
for path in "${VENV_DISCOVERY_PATHS[@]}"; do
    if [ -f "$path/bin/activate" ]; then
        VENV_ACTIVE="$path/bin/activate"
        break
    fi
done

if [ -n "$VENV_ACTIVE" ]; then
    echo "⚡ Activating virtual environment: $VENV_ACTIVE"
    source "$VENV_ACTIVE"
else
    echo "⚠️ Warning: Could not automatically locate '.venv' virtual environment."
    echo "Running with default system python..."
fi

# Run the unified download and chat script
python3 "$SCRIPT_DIR/bori_v3/scripts/download_and_test.py" "$@"
