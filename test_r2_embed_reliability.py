"""R2 embed-reliability tests (#564) — make R1's loud-fail RARE.

(1) 429 (rate limit) is retried with backoff instead of dropping the embedding.
(2) the embedding auth key is the dedicated VK (EMBEDDING_API_KEY), not the ambiguous
    NAH_MODEL_GATEWAY_KEY that can be stale in a gateway-spawned child.

Run: graphrag-venv/bin/python -m pytest test_r2_embed_reliability.py -q  (no DB / no OmniRoute).
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))
import server  # noqa: E402


def _srv():
    s = object.__new__(server.NahMemoryMCPServer)
    s.http_client = MagicMock()
    return s


def _resp(status, embedding=None):
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(
        return_value={"data": [{"embedding": embedding}]} if embedding else {"data": []}
    )
    r.raise_for_status = MagicMock()
    return r


def test_429_retries_then_succeeds():
    s = _srv()
    ok = _resp(200, [0.1] * 1024)
    s.http_client.post = AsyncMock(side_effect=[_resp(429), _resp(429), ok])
    with patch.object(server.asyncio, "sleep", new=AsyncMock()):  # no real backoff delay
        emb = asyncio.run(s.generate_embedding("hello"))
    assert emb == [0.1] * 1024, emb
    assert s.http_client.post.await_count == 3, (
        s.http_client.post.await_count
    )  # 2 retries + success


def test_persistent_429_returns_none_after_bounded_retries():
    """Persistent 429 → None (bounded, not infinite) → feeds R1's loud-fail (never NULL-as-success)."""
    s = _srv()
    s.http_client.post = AsyncMock(return_value=_resp(429))
    with patch.object(server.asyncio, "sleep", new=AsyncMock()):
        emb = asyncio.run(s.generate_embedding("hello"))
    assert emb is None, emb
    assert s.http_client.post.await_count == 5, s.http_client.post.await_count  # bounded at 5


def test_uses_dedicated_embedding_key_not_ambiguous_global():
    s = _srv()
    s.http_client.post = AsyncMock(return_value=_resp(200, [0.1] * 1024))
    asyncio.run(s.generate_embedding("hi"))
    _, kwargs = s.http_client.post.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {server.EMBEDDING_API_KEY}"


if __name__ == "__main__":
    test_429_retries_then_succeeds()
    test_persistent_429_returns_none_after_bounded_retries()
    test_uses_dedicated_embedding_key_not_ambiguous_global()
    print("R2 embed-reliability: PASS")
