#!/bin/bash
# Wrapper to execute the nah-remember offline Dreamer consolidation daemon
set -e

# Ensure Homebrew and standard paths are present for launchd execution
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..:${SCRIPT_DIR}:${PYTHONPATH}"

# Resolve python3: prefer /opt/homebrew/bin/python3 if present
if [ -x "/opt/homebrew/bin/python3" ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
else
    PYTHON_BIN="$(which python3)"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/dreamer_daemon.py" "$@"
