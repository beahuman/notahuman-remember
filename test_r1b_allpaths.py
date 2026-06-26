"""R1b write-integrity regression tests (#564) — extend the R1 invariant to ALL add/update paths.

R1 fixed memory_add (semantic_memories). R1b extends the same invariant to message_save, entity_add,
and memory_update, WITH the nuance that message_save/entity_add embeddings are OPTIONAL: a NULL is
only a failure when an embedding was REQUESTED (generate_embedding=True). memory_update always embeds,
and a failed embed there would WIPE the existing vector — the worst case — so it must refuse.

Run: graphrag-venv/bin/python -m pytest test_r1b_allpaths.py -q  (no DB / no Bifrost needed).
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
import server  # noqa: E402


def _srv():
    s = object.__new__(server.NahMemoryMCPServer)
    s.pool = MagicMock(name="pool")
    s._format_vector = lambda e: "[" + ",".join(map(str, e)) + "]"
    return s


# --- message_save -----------------------------------------------------------------


def test_message_save_requested_embed_fails_is_loud_no_insert():
    s = _srv()
    s.generate_embedding = AsyncMock(return_value=None)
    r = asyncio.run(
        s.message_save(
            session_id="x", message_index=0, role="user", content="c", generate_embedding=True
        )
    )
    assert r["success"] is False and r["error"] == "embedding_unavailable", r
    s.pool.acquire.assert_not_called()  # guard is BEFORE pool.acquire


def test_message_save_no_embed_requested_still_inserts():
    """Regression: NOT requesting an embedding must NOT be treated as a failure."""
    s = _srv()
    s.generate_embedding = AsyncMock(return_value=None)  # should never be consulted

    class _Reached(Exception):
        pass

    def _boom(*a, **k):
        raise _Reached()

    s.pool.acquire = _boom
    try:
        asyncio.run(
            s.message_save(
                session_id="x", message_index=0, role="user", content="c", generate_embedding=False
            )
        )
    except _Reached:
        pass  # reached INSERT path with no embedding — correct
    else:
        raise AssertionError("message_save with generate_embedding=False did not reach INSERT")
    s.generate_embedding.assert_not_awaited()


# --- entity_add -------------------------------------------------------------------


def test_entity_add_requested_embed_fails_is_loud_no_insert():
    s = _srv()
    s.generate_embedding = AsyncMock(return_value=None)
    r = asyncio.run(s.entity_add(name="n", entity_type="t", generate_embedding=True))
    assert r["success"] is False and r["error"] == "embedding_unavailable", r
    s.pool.acquire.assert_not_called()


# --- memory_update (worst case: would NULL an existing embedding) ------------------


def test_memory_update_embed_fails_refuses_update():
    s = _srv()
    s.generate_embedding = AsyncMock(return_value=None)

    # memory_update acquires the pool for a SELECT (current row + version) BEFORE embedding.
    # Mock that connection so the SELECT succeeds, then assert the UPDATE (conn.execute) is NEVER run.
    conn = MagicMock(name="conn")
    conn.fetchrow = AsyncMock(return_value={"content": "old"})
    conn.fetchval = AsyncMock(return_value=1)
    conn.execute = AsyncMock()

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    s.pool.acquire = MagicMock(return_value=acquire_cm)

    import uuid

    r = asyncio.run(s.memory_update(memory_id=str(uuid.uuid4()), content="new content"))
    assert r["success"] is False and r["error"] == "embedding_unavailable", r
    conn.execute.assert_not_awaited()  # CRITICAL: the UPDATE that would null the embedding never ran


if __name__ == "__main__":
    test_message_save_requested_embed_fails_is_loud_no_insert()
    test_message_save_no_embed_requested_still_inserts()
    test_entity_add_requested_embed_fails_is_loud_no_insert()
    test_memory_update_embed_fails_refuses_update()
    print("R1b all-paths write-integrity: PASS")
