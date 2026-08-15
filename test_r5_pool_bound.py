"""R5 pool-bound tests (#564) — DB waits are bounded so a saturated pool fails fast instead of
hanging forever (the indefinite-hang -> deadlock pattern under 6-agent concurrency).

Run: graphrag-venv/bin/python -m pytest test_r5_pool_bound.py -q  (no DB / no OmniRoute).
"""

import asyncio
import os
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
import server  # noqa: E402


def test_timeout_constants_are_sane():
    assert server.POOL_ACQUIRE_TIMEOUT > 0
    assert server.COMMAND_TIMEOUT > 0


def test_acquire_is_called_with_timeout():
    """A real acquire on the hot path must pass timeout=POOL_ACQUIRE_TIMEOUT (no unbounded wait)."""
    s = object.__new__(server.NahMemoryMCPServer)
    s._format_vector = lambda e: "[" + ",".join(map(str, e)) + "]"
    s.generate_embedding = AsyncMock(return_value=[0.1] * 1024)  # valid → reaches acquire

    conn = MagicMock()
    conn.fetchrow = AsyncMock(
        return_value={"id": uuid.uuid4(), "user_id": "u", "created_at": "2026-06-25"}
    )
    conn.execute = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    s.pool = MagicMock()
    s.pool.acquire = MagicMock(return_value=cm)

    r = asyncio.run(s.memory_add(content="bound me"))
    assert r["success"] is True, r
    # the load-bearing assertion: acquire was bounded
    _, kwargs = s.pool.acquire.call_args
    assert kwargs.get("timeout") == server.POOL_ACQUIRE_TIMEOUT, s.pool.acquire.call_args


if __name__ == "__main__":
    test_timeout_constants_are_sane()
    test_acquire_is_called_with_timeout()
    print("R5 pool-bound: PASS")
