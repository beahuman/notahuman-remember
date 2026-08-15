#!/usr/bin/env python3
"""
nah-remember-dreamer.py — Phase D: Offline Dreaming Pipeline

Reads episodic snapshots + semantic memories, calls LLM to extract patterns,
and commits learned workflows (procedures) and failure patterns (negative constraints).

Architecture:
  Episodic Reader -> Analysis Engine (LLM via OmniRoute) -> Commit Writer
       |                    |                              |
       v                    v                              v
  context_snapshots    LLM extracts                   procedures /
  + messages           patterns +                      negative_constraints
                       preferences

Evidence-sufficiency gate:
  - n=1  -> log, don't promote
  - n=3+ -> promote to procedures
  - n=5+ -> high-confidence procedure
  - Failure pattern repeated -> promote to negative_constraints

Usage:
  python3 nah-remember-dreamer.py              # Full run
  python3 nah-remember-dreamer.py --dry-run    # Analyze only, no writes
  python3 nah-remember-dreamer.py --sessions 5 # Last N sessions only
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Optional

import asyncpg
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nah-remember-dreamer")

NAH_MODEL_GATEWAY_URL = os.getenv("NAH_MODEL_GATEWAY_URL", "http://localhost:20128")
NAH_MODEL_GATEWAY_KEY = os.getenv("NAH_MODEL_GATEWAY_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")
# Name the routing ALIAS, not a raw model. OmniRoute's policy resolves
# "nah-remember-dreamer" to free primary -> free -> free -> deepseek-v4-flash LAST.
# Was "deepseek-v4-flash" (raw PAID), which bypassed the policy and billed on every run.
LLM_MODEL = os.getenv("DREAMER_LLM_MODEL", "nah-remember-dreamer")


class DreamingPipeline:
    """Offline consolidation pipeline - episodic to procedural extraction."""

    def __init__(self):
        self.db_url = DATABASE_URL
        self.pool = None
        self.http_client = None
        self.dry_run = False
        self.sessions_limit = 10

    async def setup(self):
        self.pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=5)
        self.http_client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"Dreamer connected to {self.db_url}")

    async def cleanup(self):
        if self.pool:
            await self.pool.close()
        if self.http_client:
            await self.http_client.aclose()

    async def get_recent_sessions(self) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT s.id, s.session_id, s.application, s.status,
                       s.created_at, s.ended_at, s.total_messages, s.summary,
                       cs.context_data, cs.active_files, cs.decisions_made
                FROM nah_remember.sessions s
                LEFT JOIN nah_remember.context_snapshots cs
                    ON cs.session_id = s.session_id
                WHERE s.status = 'completed'
                  AND s.ended_at IS NOT NULL
                ORDER BY s.ended_at DESC
                LIMIT $1
            """,
                self.sessions_limit,
            )
            return [dict(r) for r in rows]

    async def get_semantic_memories(self, session_ids: List[str]) -> List[Dict]:
        if not session_ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, memory_type, category, tags, importance_score,
                       access_count, reinforcement_score, created_at, agent_id
                FROM nah_remember.semantic_memories
                WHERE agent_id = ANY($1::text[])
                ORDER BY importance_score DESC
                LIMIT 50
            """,
                session_ids,
            )
            return [dict(r) for r in rows]

    async def call_llm(self, prompt: str) -> Optional[str]:
        try:
            response = await self.http_client.post(
                f"{NAH_MODEL_GATEWAY_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {NAH_MODEL_GATEWAY_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def build_analysis_prompt(self, sessions, memories):
        session_lines = []
        for s in sessions:
            sid = str(s.get("session_id", ""))[:12]
            app = s.get("application", "unknown")
            session_lines.append(f"Session {sid}... ({app}):")
            if s.get("summary"):
                session_lines.append(f"  Summary: {str(s['summary'])[:500]}")
            if s.get("decisions_made"):
                session_lines.append(f"  Decisions: {str(s['decisions_made'])[:500]}")
            if s.get("context_data"):
                session_lines.append(f"  Context: {str(s['context_data'])[:500]}")

        memory_lines = []
        for m in memories[:20]:
            mt = m.get("memory_type", "unknown")
            mc = str(m.get("content", ""))[:300]
            sc = m.get("importance_score", 0.5)
            memory_lines.append(f"- [{mt}] {mc} (score: {sc})")

        session_text = "\n".join(session_lines)
        memory_text = "\n".join(memory_lines)

        return f"""You are analyzing agent session data to extract patterns. Identify:

1. RECURRING FAILURE PATTERNS: Actions or approaches that repeatedly led to errors.
2. SUCCESSFUL WORKFLOWS: Approaches that consistently worked well across sessions.
3. USER PREFERENCES: Repeated preferences or stylistic choices.
4. STALE CONTENT: Facts or procedures that appear outdated.

Evidence-sufficiency: only report patterns appearing in 3+ distinct sessions (n>=3).
For n=1 or n=2, label them as "insufficient evidence."

SESSIONS:
{session_text}

MEMORIES:
{memory_text}

Respond in JSON format:
{{
    "failure_patterns": [{{"pattern": "...", "trigger_context": "...", "severity": "warning|blocking", "evidence_sessions": N}}],
    "successful_workflows": [{{"workflow": "...", "steps": ["...", "..."], "evidence_sessions": N}}],
    "user_preferences": [{{"preference": "...", "context": "..."}}],
    "stale_content": [{{"memory_id_or_desc": "...", "reason": "..."}}]
}}"""

    async def commit_patterns(self, analysis):
        evidence_gate = 3
        high_confidence = 5

        for fp in analysis.get("failure_patterns", []):
            ev = fp.get("evidence_sessions", 0)
            if ev >= evidence_gate:
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would write constraint: {fp['pattern'][:80]}")
                    continue
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO nah_remember.negative_constraints
                            (pattern, trigger_context, severity, weight)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                    """,
                        fp["pattern"],
                        fp.get("trigger_context", ""),
                        fp.get("severity", "warning"),
                        0.9 if ev >= high_confidence else 0.7,
                    )
                logger.info(f"Wrote constraint: {fp['pattern'][:80]} (evidence: {ev})")

        for wf in analysis.get("successful_workflows", []):
            ev = wf.get("evidence_sessions", 0)
            if ev >= evidence_gate:
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would write procedure: {wf['workflow'][:80]}")
                    continue
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO nah_remember.procedures
                            (name, trigger_context, steps, success_rate, execution_count)
                        VALUES ($1, $2, $3::jsonb, $4, $5)
                        ON CONFLICT (name) DO UPDATE SET
                            success_rate = GREATEST(procedures.success_rate, $4),
                            execution_count = procedures.execution_count + 1
                    """,
                        wf["workflow"],
                        wf.get("trigger_context", ""),
                        json.dumps(wf.get("steps", [])),
                        0.9 if ev >= high_confidence else 0.7,
                        ev,
                    )
                logger.info(f"Wrote procedure: {wf['workflow'][:80]} (evidence: {ev})")

    async def run(self):
        logger.info(f"Dreamer starting (dry_run={self.dry_run}, sessions={self.sessions_limit})")

        sessions = await self.get_recent_sessions()
        if not sessions:
            logger.info("No completed sessions to analyze. Exiting.")
            return

        session_ids = [s["session_id"] for s in sessions if s.get("session_id")]
        memories = await self.get_semantic_memories(session_ids) if session_ids else []

        logger.info(f"Loaded {len(sessions)} sessions, {len(memories)} semantic memories")

        prompt = self.build_analysis_prompt(sessions, memories)
        response = await self.call_llm(prompt)

        if not response:
            logger.error("LLM returned no response. Skipping commit phase.")
            return

        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            try:
                analysis = json.loads(response[start_idx:end_idx])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON: {e}")
                return
        else:
            logger.error("No JSON found in LLM response")
            return

        await self.commit_patterns(analysis)

        logger.info(
            f"Dreamer complete. Extracted: "
            f"{len(analysis.get('failure_patterns', []))} failure patterns, "
            f"{len(analysis.get('successful_workflows', []))} workflows, "
            f"{len(analysis.get('user_preferences', []))} preferences, "
            f"{len(analysis.get('stale_content', []))} stale items"
        )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="nah-remember Dreaming Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, no writes")
    parser.add_argument("--sessions", type=int, default=10, help="Number of sessions to analyze")
    args = parser.parse_args()

    dreamer = DreamingPipeline()
    dreamer.dry_run = args.dry_run
    dreamer.sessions_limit = args.sessions

    try:
        await dreamer.setup()
        await dreamer.run()
    finally:
        await dreamer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
