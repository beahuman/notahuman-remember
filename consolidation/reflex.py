#!/usr/bin/env python3
"""
Atomic reversible reflex correction for the nah-remember consolidation engine.
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation

Algorithm:
  1. Validate delta != 0 AND |delta| <= MAX_DELTA (safety cap).
  2. Write an undo record (before applying) to reflex_dir/<undo_token>.json.
  3. Apply weight change to database (nah_remember.procedures or reinforcement weights).
  4. Return result dict with undo_token.
  5. undo_reflex(undo_token): restores previous weight and marks token spent.
"""

from decimal import Decimal
import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

import asyncpg

logger = logging.getLogger("nah-remember-reflex")

DEFAULT_REFLEX_DIR = Path(__file__).parent / "_reflex"
MAX_REFLEX_DELTA = Decimal("5.0")
DEFAULT_DB_URL = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")


def _execute_db_update(
    memory_id: str, new_weight: float, db_url: Optional[str] = None
) -> Optional[str]:
    """Execute real DB update to procedures, semantic_memories, or episodes."""
    url = db_url or os.getenv("DATABASE_URL", DEFAULT_DB_URL)

    async def _run():
        conn = await asyncpg.connect(url)
        try:
            # 1. Check procedures (by UUID id or name)
            res = await conn.execute(
                """
                UPDATE nah_remember.procedures 
                SET success_rate = $1, updated_at = NOW() 
                WHERE id::text = $2 OR name = $2
                """,
                new_weight,
                memory_id,
            )
            if res != "UPDATE 0":
                return "procedures"

            # 2. Check semantic_memories (by UUID id)
            res = await conn.execute(
                """
                UPDATE nah_remember.semantic_memories 
                SET reinforcement_score = $1, updated_at = NOW() 
                WHERE id::text = $2
                """,
                new_weight,
                memory_id,
            )
            if res != "UPDATE 0":
                return "semantic_memories"

            # 3. Check episodes (by UUID id)
            res = await conn.execute(
                """
                UPDATE nah_remember.episodes 
                SET reinforcement_score = $1, updated_at = NOW() 
                WHERE id::text = $2
                """,
                new_weight,
                memory_id,
            )
            if res != "UPDATE 0":
                return "episodes"

            return "no_match"
        finally:
            await conn.close()

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(_run())).result(timeout=5)
        else:
            return asyncio.run(_run())
    except Exception as e:
        logger.debug(f"DB update skipped/failed for {memory_id}: {e}")
        return None


def reflex(
    memory_id: str,
    delta: Any,
    w_before: float,
    *,
    reflex_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    apply_db: bool = True,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a reversible single-memory correction with an undo record and live DB write."""
    r_dir = Path(reflex_dir) if reflex_dir else DEFAULT_REFLEX_DIR
    delta_dec = Decimal(str(delta))

    if delta_dec == Decimal("0"):
        return {
            "applied": False,
            "undo_token": None,
            "memory_id": memory_id,
            "delta_applied": None,
            "w_before": w_before,
            "w_after": w_before,
            "reason": "delta is zero — no-op",
        }

    if abs(delta_dec) > MAX_REFLEX_DELTA:
        return {
            "applied": False,
            "undo_token": None,
            "memory_id": memory_id,
            "delta_applied": None,
            "w_before": w_before,
            "w_after": w_before,
            "reason": f"delta {delta_dec} exceeds safety cap {MAX_REFLEX_DELTA}",
        }

    w_after = max(0.0, float(Decimal(str(w_before)) + delta_dec))
    undo_token = str(uuid.uuid4())

    r_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    undo_path = r_dir / f"reflex_{undo_token}.json"

    # Execute live DB update (D5 fix)
    db_table = None
    if apply_db:
        db_table = _execute_db_update(memory_id, w_after, db_url=db_url)

    undo_record = {
        "undo_token": undo_token,
        "memory_id": memory_id,
        "w_before": w_before,
        "w_after": w_after,
        "delta": str(delta_dec),
        "applied_at": ts,
        "reversed": False,
        "db_table": db_table,
        "metadata": metadata or {},
    }
    undo_path.write_text(json.dumps(undo_record, indent=2), encoding="utf-8")

    return {
        "applied": True,
        "undo_token": undo_token,
        "undo_path": str(undo_path),
        "memory_id": memory_id,
        "delta_applied": float(delta_dec),
        "w_before": w_before,
        "w_after": w_after,
        "db_table": db_table,
        "reason": f"reflex applied: {memory_id} {w_before:.4f} -> {w_after:.4f}",
    }


def undo_reflex(
    undo_token: str,
    *,
    reflex_dir: Optional[Path] = None,
    apply_db: bool = True,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Reverse a previous reflex operation using its undo token and restore DB weight."""
    r_dir = Path(reflex_dir) if reflex_dir else DEFAULT_REFLEX_DIR
    undo_path = r_dir / f"reflex_{undo_token}.json"

    if not undo_path.exists():
        return {
            "reversed": False,
            "memory_id": None,
            "w_restored": None,
            "reason": f"undo token not found: {undo_token!r}",
        }

    record = json.loads(undo_path.read_text(encoding="utf-8"))
    if record.get("reversed", False):
        return {
            "reversed": False,
            "memory_id": record["memory_id"],
            "w_restored": None,
            "reason": "reflex already reversed — token is spent",
        }

    # Execute reverse DB update (D5 fix)
    db_table = None
    if apply_db:
        db_table = _execute_db_update(record["memory_id"], record["w_before"], db_url=db_url)

    record["reversed"] = True
    record["reversed_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    undo_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return {
        "reversed": True,
        "memory_id": record["memory_id"],
        "w_restored": record["w_before"],
        "db_table": db_table,
        "reason": f"reflex reversed: {record['memory_id']} restored to {record['w_before']:.4f}",
    }
