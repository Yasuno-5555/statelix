#!/bin/bash

# Change directory to the project root (where this script is)
cd "$(dirname "$0")"

# Kill previous instances
pkill -f "statelix_py.app" || true

# Activate Virtual Environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Set PYTHONPATH to include current directory
export PYTHONPATH="$PYTHONPATH:$(pwd)"
# export QT_QPA_PLATFORM=xcb # Caused crash due to missing libxcb-cursor0
export QT_DEBUG_PLUGINS=0

# Run the Application
# Redirect output to log file for debugging since we hide terminal
python3 -m statelix_py.app > app.log 2>&1
