#!/usr/bin/env python3
"""
Backward-compatibility wrapper for nah_remember.ranking.

For new installations, import directly from `nah_remember.ranking`
or `nah_remember`.
"""
import sys
import os

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_src_dir) and _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from nah_remember.ranking import *  # noqa: F401, F403
