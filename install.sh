#!/bin/bash
# Install dependencies for nah-memory-unified MCP server
# Run this script before starting the server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== NAH Memory Unified MCP Installation ==="

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Install dependencies
echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install asyncpg httpx numpy

# Verify
echo ""
echo "Installed packages:"
"$VENV_DIR/bin/pip" list | grep -E "asyncpg|httpx|numpy"

echo ""
echo "Installation complete!"
echo "Python: $VENV_DIR/bin/python"
