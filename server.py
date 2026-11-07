#!/usr/bin/env python3
"""
Backward-compatibility wrapper for nah_remember.server.

For new installations, use the `nah-remember` command or import
directly from `nah_remember.server`.
"""
import sys
import os

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_src_dir) and _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from nah_remember.server import *  # noqa: F401, F403
from nah_remember.server import cli, main  # noqa: F401

if __name__ == "__main__":
    cli()
