#!/bin/bash
# Complete setup for nah-memory-unified MCP server
# Run this script to install dependencies and update config

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
MCP_JSON="/Users/human/notahuman/.mcp.json"

echo "=== NAH Memory Unified Complete Setup ==="
echo ""

# Step 1: Install Python dependencies
echo "Step 1: Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install asyncpg httpx numpy

echo ""
echo "Installed packages:"
"$VENV_DIR/bin/pip" list | grep -E "asyncpg|httpx|numpy"

# Step 2: Update .mcp.json if needed
echo ""
echo "Step 2: Checking MCP configuration..."

if grep -q "nah-memory-unified" "$MCP_JSON"; then
    echo "✓ nah-memory-unified already configured in .mcp.json"
else
    echo "Adding nah-memory-unified to .mcp.json..."
    # Use Python to update JSON safely
    python3 << 'PYTHON_SCRIPT'
import json

mcp_path = "/Users/human/notahuman/.mcp.json"

with open(mcp_path, 'r') as f:
    config = json.load(f)

# Add new server before mem0-memory
new_server = {
    "command": "/Users/human/notahuman/infrastructure/memory/nah-memory-unified/venv/bin/python",
    "args": [
        "/Users/human/notahuman/infrastructure/memory/nah-memory-unified/nah_memory_mcp.py"
    ],
    "cwd": "/Users/human/notahuman",
    "env": {
        "DATABASE_URL": "postgresql://human:@localhost/notahuman",
        "LITELLM_BASE_URL": "http://localhost:4000",
        "EMBEDDING_MODEL": "embedding-primary",
        "EMBEDDING_MODEL_FALLBACK": "embedding-large",
        "EMBEDDING_DIM": "1536"
    }
}

# Insert before mem0-memory by rebuilding dict
new_servers = {}
for key, value in config["mcpServers"].items():
    if key == "mem0-memory":
        new_servers["nah-memory-unified"] = new_server
    new_servers[key] = value

config["mcpServers"] = new_servers

with open(mcp_path, 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Added nah-memory-unified to .mcp.json")
PYTHON_SCRIPT
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "The nah-memory-unified MCP server is now configured."
echo "Restart Claude Desktop to load the new MCP server."
echo ""
echo "To test the server manually:"
echo "  $VENV_DIR/bin/python $SCRIPT_DIR/nah_memory_mcp.py"
