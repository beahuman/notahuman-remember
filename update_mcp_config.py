#!/usr/bin/env python3
"""Update .mcp.json to add nah-memory-unified server configuration."""

import json
import sys

MCP_JSON_PATH = "/Users/human/notahuman/.mcp.json"


def main():
    print("Updating .mcp.json with nah-memory-unified configuration...")

    # Read current config
    with open(MCP_JSON_PATH, "r") as f:
        config = json.load(f)

    # Check if already configured
    if "nah-memory-unified" in config.get("mcpServers", {}):
        print("✓ nah-memory-unified already configured")
        return 0

    # New server configuration
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
            "EMBEDDING_DIM": "1536",
        },
    }

    # Insert before mem0-memory by rebuilding dict
    new_servers = {}
    for key, value in config["mcpServers"].items():
        if key == "mem0-memory":
            new_servers["nah-memory-unified"] = new_server
        new_servers[key] = value

    # If mem0-memory wasn't found, add at end
    if "nah-memory-unified" not in new_servers:
        new_servers["nah-memory-unified"] = new_server

    config["mcpServers"] = new_servers

    # Write updated config
    with open(MCP_JSON_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print("✓ Added nah-memory-unified to .mcp.json")
    print(f"  Total MCP servers: {len(new_servers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
