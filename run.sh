#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Starting NeuroGuard Clinic..."

if [ ! -d "$DIR/venv" ]; then
    echo "Virtual environment not found. Please create one."
    exit 1
fi

source "$DIR/venv/bin/activate"

# Export the base directory for safety
export PYTHONPATH="$DIR:$PYTHONPATH"

# Kill any existing instance of the server on port 5001
lsof -t -i:5001 | xargs kill -9 2>/dev/null || true

echo "Launching backend on port 5001..."
python "$DIR/backend/app.py"
