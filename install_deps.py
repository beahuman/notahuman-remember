#!/usr/bin/env python3
"""Install dependencies for nah-memory-unified MCP server."""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PIP = os.path.join(SCRIPT_DIR, "venv", "bin", "pip")

PACKAGES = ["asyncpg", "httpx", "numpy"]


def main():
    print("=== Installing NAH Memory Unified Dependencies ===")
    print(f"Using pip: {VENV_PIP}")
    print(f"Packages: {', '.join(PACKAGES)}")
    print()

    # Upgrade pip first
    print("Upgrading pip...")
    result = subprocess.run(
        [VENV_PIP, "install", "--upgrade", "pip"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Warning: pip upgrade failed: {result.stderr}")

    # Install packages
    print(f"\nInstalling {', '.join(PACKAGES)}...")
    result = subprocess.run([VENV_PIP, "install"] + PACKAGES, capture_output=True, text=True)

    if result.returncode == 0:
        print("SUCCESS!")
        print(result.stdout)
        return 0
    else:
        print(f"ERROR: {result.stderr}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
