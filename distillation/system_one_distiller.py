#!/usr/bin/env python3
"""
System 1 Distillation Engine for nah-remember.
Ecosystem: nah-neural / Subsystem: nah-remember / Module: distillation

Uses local Apple Silicon SLM (Qwen 2.5 1.5B 4-bit) via OmniRoute (:20128)
or direct MLX daemon (:52342) to distill closed episodes into concise
narratives and retrospective learnings at zero API cost.
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional
import uuid

import asyncpg
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("system-one-distiller")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")
OMNIROUTE_URL = os.getenv("NAH_OMNIROUTE_URL", "http://127.0.0.1:20128/v1")
MLX_DIRECT_URL = os.getenv("NAH_MLX_URL", "http://127.0.0.1:52342/v1")
MODEL_ALIAS = "nah-system-one-combo"
DIRECT_MODEL_PATH = "/Users/human/notahuman/infrastructure/nah-models/qwen2.5-1.5b-instruct-4bit"


# Retrieve bearer token from environment or nah-nexus vault
def get_auth_token() -> str:
    token = (
        os.getenv("OMNIROUTE_API_KEY")
        or os.getenv("NAH_MODEL_GATEWAY_KEY")
        or os.getenv("VK_NAH_REMEMBER_FREE")
        or os.getenv("VK_UNO_MAIN", "")
    )
    if not token:
        try:
            import subprocess

            res = subprocess.run(
                [
                    "python3",
                    "/Users/human/notahuman/infrastructure/nah-nexus/vault-query.py",
                    "get",
                    "VK_UNO_MAIN",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            token = res.stdout.strip()
        except Exception:
            token = ""
    return token or ""


class SystemOneDistiller:
    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.pool: Optional[asyncpg.Pool] = None
        self.auth_token = get_auth_token()

    async def setup(self):
        self.pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=4)

    async def cleanup(self):
        if self.pool:
            await self.pool.close()

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str = "You are an AI cognitive system distilling operational experience into actionable playbooks.",
    ) -> str:
        """Call local System 1 model with automatic fallback from OmniRoute to MLX."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # 1. Attempt OmniRoute
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{OMNIROUTE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL_ALIAS,
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": 250,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                logger.warning(
                    f"OmniRoute returned {resp.status_code}, falling back to direct MLX: {resp.text}"
                )
        except Exception as e:
            logger.warning(f"OmniRoute call failed ({e}), falling back to direct MLX daemon")

        # 2. Fallback to direct MLX daemon
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{MLX_DIRECT_URL}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": DIRECT_MODEL_PATH,
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": 250,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                logger.error(f"Direct MLX call returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Direct MLX call failed: {e}")

        return ""

    async def distill_episode(self, episode_id: uuid.UUID) -> Dict[str, Any]:
        """Fetch episode events and synthesize narrative + retrospective."""
        async with self.pool.acquire() as conn:
            ep = await conn.fetchrow(
                """
                SELECT id, title, goal, task_type, hunt_context, outcome, narrative, retrospective
                FROM nah_remember.episodes
                WHERE id = $1
                """,
                episode_id,
            )
            if not ep:
                return {"success": False, "error": f"Episode {episode_id} not found"}

            event_rows = await conn.fetch(
                """
                SELECT event_index, event_type, description, tool_name, tool_result
                FROM nah_remember.episode_events
                WHERE episode_id = $1
                ORDER BY event_index ASC
                LIMIT 20
                """,
                episode_id,
            )

        events_summary = []
        for ev in event_rows:
            events_summary.append(
                f"- Event {ev['event_index']} ({ev['event_type']}): {ev['description'] or ev['tool_name']}"
            )

        events_text = "\n".join(events_summary) or "No discrete events recorded."

        prompt = (
            f"Episode Title: {ep['title']}\n"
            f"Goal: {ep['goal']}\n"
            f"Task Type: {ep['task_type']}\n"
            f"Outcome: {ep['outcome']}\n\n"
            f"Chronological Events:\n{events_text}\n\n"
            "Task: Distill this completed episode into:\n"
            "1. Concise Narrative: 2-3 sentences explaining what transpired and final resolution.\n"
            "2. Retrospective Learnings: 2-3 key takeaways or procedural rules learned."
        )

        distilled = await self.generate_completion(prompt)
        if not distilled:
            return {"success": False, "error": "Model failed to produce distillation"}

        # Store in database
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE nah_remember.episodes
                SET retrospective = CASE WHEN retrospective IS NULL OR retrospective = '' THEN $2 ELSE retrospective END,
                    narrative = CASE WHEN narrative IS NULL OR narrative = '' THEN $3 ELSE narrative END,
                    updated_at = NOW()
                WHERE id = $1
                """,
                episode_id,
                distilled,
                distilled[:200],
            )

        return {
            "success": True,
            "episode_id": str(episode_id),
            "distillation": distilled,
        }

    async def run_batch(self, limit: int = 5) -> Dict[str, Any]:
        """Find closed episodes lacking retrospectives and distill them."""
        async with self.pool.acquire() as conn:
            ep_rows = await conn.fetch(
                """
                SELECT id
                FROM nah_remember.episodes
                WHERE ended_at IS NOT NULL
                  AND (retrospective IS NULL OR retrospective = '')
                ORDER BY ended_at DESC
                LIMIT $1
                """,
                limit,
            )

        distilled_count = 0
        results = []
        for r in ep_rows:
            res = await self.distill_episode(r["id"])
            if res.get("success"):
                distilled_count += 1
                results.append(res)

        return {
            "total_candidates": len(ep_rows),
            "distilled_count": distilled_count,
            "results": results,
        }


async def main():
    parser = argparse.ArgumentParser(description="nah-remember System 1 Distiller")
    parser.add_argument(
        "--batch", type=int, default=5, help="Number of undisitilled episodes to process"
    )
    parser.add_argument("--episode", type=str, help="Specific episode UUID to distill")
    args = parser.parse_args()

    distiller = SystemOneDistiller()
    await distiller.setup()
    try:
        if args.episode:
            ep_uuid = uuid.UUID(args.episode)
            res = await distiller.distill_episode(ep_uuid)
            print(json.dumps(res, indent=2))
        else:
            res = await distiller.run_batch(limit=args.batch)
            print(json.dumps(res, indent=2))
    finally:
        await distiller.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
