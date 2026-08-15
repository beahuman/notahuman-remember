"""R1 write-integrity regression test (M6 nah-remember FLAWLESS remediation, #564).

Proves the core invariant: memory_add NEVER persists a NULL-embedding row as success.
When generate_embedding returns None (OmniRoute down / 429 / 401 / key-leak), memory_add must
fail LOUD (success:false) and must NOT touch the pool / INSERT. This is the fix for the
verified 30% silent data-loss (LESSON-10 class).

Run: graphrag-venv/bin/python -m pytest test_r1_write_integrity.py -q
(no DB / no OmniRoute required — pool.acquire is asserted NOT called).
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
import server  # noqa: E402


def _make_server():
    """Instantiate without running __init__ (which would open a DB pool)."""
    srv = object.__new__(server.NahMemoryMCPServer)
    srv.pool = MagicMock(name="pool")  # if .acquire() is called the test fails (see asserts)
    srv._format_vector = lambda e: "[" + ",".join(map(str, e)) + "]"
    return srv


def test_embedding_none_returns_loud_failure_and_does_not_insert():
    srv = _make_server()
    srv.generate_embedding = AsyncMock(return_value=None)  # simulate embed failure

    result = asyncio.run(srv.memory_add(content="a real memory that must not be silently dropped"))

    # 1. Loud failure, not a fake success.
    assert result["success"] is False, result
    assert result["error"] == "embedding_unavailable", result
    assert result["has_embedding"] is False, result
    # 2. Content echoed back so the caller can retry without data loss.
    assert "content" in result and result["content"], result
    # 3. CRITICAL: no INSERT attempted — the pool was never acquired.
    srv.pool.acquire.assert_not_called()


def test_embedding_present_proceeds_past_the_guard():
    """Sanity: a valid embedding must NOT be short-circuited by the R1 guard.

    We stop at pool.acquire (raise a sentinel) to prove control flow reached the INSERT path
    without needing a live DB.
    """
    srv = _make_server()
    srv.generate_embedding = AsyncMock(return_value=[0.1] * 1024)

    class _Reached(Exception):
        pass

    def _boom(*_a, **_k):
        raise _Reached()

    srv.pool.acquire = _boom

    try:
        asyncio.run(srv.memory_add(content="valid content"))
    except _Reached:
        pass  # expected — we reached the INSERT path (guard did NOT short-circuit)
    else:
        raise AssertionError("memory_add did not reach the INSERT path for a valid embedding")


if __name__ == "__main__":
    test_embedding_none_returns_loud_failure_and_does_not_insert()
    test_embedding_present_proceeds_past_the_guard()
    print("R1 write-integrity: PASS")
