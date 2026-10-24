#!/usr/bin/env python3
"""Backfill NULL embeddings in nah_remember.semantic_memories (Servo item 1, 2026-06-06).

Embeds every row whose `embedding` IS NULL via OmniRoute (mistral-embed, 1024-dim) using the
dedicated memory VK — the same path the live server uses — and UPDATEs the row. Idempotent:
re-running only touches rows that are still NULL.

Usage:
    venv/bin/python backfill_embeddings.py            # backfill all NULL rows
    venv/bin/python backfill_embeddings.py --dry-run  # report counts only, no writes
"""

import argparse
import asyncio
import os
import subprocess
import sys

import asyncpg
import httpx

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://human:@localhost/notahuman?search_path=nah_remember,public",
)
GATEWAY = os.getenv("EMBED_URL", "http://127.0.0.1:52343/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


def _vk() -> str:
    key = os.getenv("NAH_REMEMBER_EMBED_KEY") or os.getenv("VK_NAH_REMEMBER_FREE")
    if key:
        return key
    r = subprocess.run(
        [
            "python3",
            "/Users/human/notahuman/infrastructure/nah-nexus/vault-query.py",
            "get",
            "VK_NAH_REMEMBER_FREE",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    raise SystemExit("Could not resolve VK_NAH_REMEMBER_FREE (env or vault).")


def _normalize(vec):
    if len(vec) > EMBEDDING_DIM:
        return vec[:EMBEDDING_DIM]
    if len(vec) < EMBEDDING_DIM:
        return vec + [0.0] * (EMBEDDING_DIM - len(vec))
    return vec


def _to_pgvector(vec) -> str:
    return "[" + ",".join(str(x) for x in _normalize(vec)) + "]"


async def _embed(client: httpx.AsyncClient, vk: str, text: str, max_retries: int = 7):
    delay = 2.0
    for _ in range(max_retries):
        resp = await client.post(
            GATEWAY,
            headers={"Authorization": f"Bearer {vk}", "Content-Type": "application/json"},
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        if resp.status_code == 429:  # Mistral free-tier rate limit — back off and retry
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("data", [{}])[0].get("embedding")
        if not emb:
            raise RuntimeError(f"empty embedding response: {str(data)[:200]}")
        return emb
    raise RuntimeError("rate-limited (429) after retries")


async def main(dry_run: bool, reembed_all: bool = False):
    try:
        vk = _vk()
    except SystemExit:
        # The local embed-server (:52343) does not authenticate; a gateway URL does.
        # Missing VK is only fatal when the endpoint is not loopback.
        if "127.0.0.1" in GATEWAY or "localhost" in GATEWAY:
            vk = ""
        else:
            raise
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT count(*) FROM nah_remember.semantic_memories")
            where = "TRUE" if reembed_all else "embedding IS NULL"
            null_rows = await conn.fetch(
                "SELECT id, content FROM nah_remember.semantic_memories "
                f"WHERE {where} ORDER BY created_at"
            )
        print(f"semantic_memories: {total} total · {len(null_rows)} with NULL embedding")
        if dry_run:
            print("[dry-run] no writes performed.")
            return
        if not null_rows:
            print("Nothing to backfill — 0 NULL embeddings.")
            return

        ok = fail = 0
        async with httpx.AsyncClient(timeout=40.0) as client:
            for i, row in enumerate(null_rows, 1):
                try:
                    emb = await _embed(client, vk, row["content"] or "")
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE nah_remember.semantic_memories "
                            "SET embedding = $1::vector WHERE id = $2",
                            _to_pgvector(emb),
                            row["id"],
                        )
                    ok += 1
                    if i % 10 == 0 or i == len(null_rows):
                        print(f"  {i}/{len(null_rows)} embedded (ok={ok} fail={fail})")
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    print(f"  FAIL id={row['id']}: {str(exc)[:140]}")
                await asyncio.sleep(0.05)  # gentle rate-limit to OmniRoute

        async with pool.acquire() as conn:
            remaining = await conn.fetchval(
                "SELECT count(*) FROM nah_remember.semantic_memories WHERE embedding IS NULL"
            )
        print(f"\nDONE — embedded {ok}, failed {fail}, NULL remaining: {remaining}")
        if remaining:
            sys.exit(1)
    finally:
        await pool.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--all",
        action="store_true",
        help="re-embed every row (use when the embedding model changed)",
    )
    args = ap.parse_args()
    asyncio.run(main(args.dry_run, args.all))
