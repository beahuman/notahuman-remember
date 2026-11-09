#!/bin/bash
# Wrapper to execute the nah-remember offline Dreamer consolidation daemon
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..:${SCRIPT_DIR}:${PYTHONPATH}"

exec /usr/bin/env python3 "${SCRIPT_DIR}/dreamer_daemon.py" "$@"
