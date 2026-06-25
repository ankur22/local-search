#!/bin/bash
# Wrapper script to start the Flask app after ensuring Ollama is ready

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Wait for Ollama to be ready
python "$SCRIPT_DIR/check_ollama.py" || {
    echo "Failed to connect to Ollama. Make sure Ollama is running."
    exit 1
}

# Start the Flask app
exec python "$SCRIPT_DIR/app.py"

