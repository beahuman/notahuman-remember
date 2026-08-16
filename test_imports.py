#!/usr/bin/env python3
"""Test that all imports work for nah-memory-unified MCP server."""

import sys
import os

# Add venv to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_SITE = os.path.join(SCRIPT_DIR, "venv", "lib", "python3.13", "site-packages")
sys.path.insert(0, VENV_SITE)

print("=== Testing NAH Memory Unified Imports ===\n")

# Test standard library
print("Testing standard library imports...")
try:
    import asyncio
    import json
    import logging
    from datetime import datetime
    from typing import Optional, List, Dict, Any

    print("  asyncio: OK")
    print("  json: OK")
    print("  logging: OK")
    print("  datetime: OK")
    print("  typing: OK")
except ImportError as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test third-party packages
print("\nTesting third-party imports...")
try:
    import asyncpg

    print(f"  asyncpg: OK (version {asyncpg.__version__})")
except ImportError as e:
    print(f"  asyncpg: FAILED - {e}")
    # Note: This runs in CLAUDE_CHAT_AGENT (agent), not the macOS application
    sys.exit(1)

try:
    import httpx

    print(f"  httpx: OK (version {httpx.__version__})")
except ImportError as e:
    print(f"  httpx: FAILED - {e}")
    sys.exit(1)

try:
    import numpy as np

    print(f"  numpy: OK (version {np.__version__})")
except ImportError as e:
    print(f"  numpy: FAILED - {e}")
    sys.exit(1)

print("\n=== All imports successful! ===")
print("\nThe nah-memory-unified MCP server should be ready to run.")
print("Restart Claude Desktop to load the new MCP server.")
