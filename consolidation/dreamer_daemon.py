#!/usr/bin/env python3
"""
nah-remember Dreamer Consolidation Daemon (Phase 4A)
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation

Harvests experiential episodic trajectories (nah_remember.episodes + episode_events),
synthesizes candidate procedural playbooks & negative constraints, and sieves them
through the mathematically certified consolidation pipeline:
  1. Wald SPRT Evidence Sufficiency Test (evidence_sufficiency.py)
  2. Double-Entry Conservation Trial Balance (trial_balance.py)
  3. Rotated-Judge Wake Gate Probe (wake_gate.py)

Usage:
  python3 dreamer_daemon.py --once             # Single consolidation pass
  python3 dreamer_daemon.py --dry-run          # Evaluate and report without DB writes
  python3 dreamer_daemon.py --limit 10         # Process last N completed episodes
  python3 dreamer_daemon.py --daemon           # Continuous background runner
"""

import argparse
import asyncio
from decimal import Decimal
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import httpx

# Ensure consolidation modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consolidation.evidence_sufficiency import sufficient, SPRTAccumulator
from consolidation.ledger import make_event_id
from consolidation.trial_balance import check_blast_radius, DEFAULT_SEALED_ANCHORS
from consolidation.wake_gate import evaluate_wake_gate, EXPECTED_PRODUCER, neural_judge_probe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nah-dreamer-daemon")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")
LOCAL_EMBED_URL = os.getenv("LOCAL_EMBED_URL", "http://127.0.0.1:52343/embed")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "qwen2.5-1.5b-instruct")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "bge-reranker-v2-m3")


class DreamerDaemon:
    """Offline consolidation daemon executing certified Bayesian & conservation sieves."""

    def __init__(
        self,
        db_url: str = DATABASE_URL,
        embed_url: str = LOCAL_EMBED_URL,
        synthesizer_model: str = SYNTHESIZER_MODEL,
        judge_model: str = JUDGE_MODEL,
    ):
        self.db_url = db_url
        self.embed_url = embed_url
        self.synthesizer_model = synthesizer_model
        self.judge_model = judge_model
        self.pool: Optional[asyncpg.Pool] = None
        self.http_client: Optional[httpx.AsyncClient] = None

    async def setup(self):
        """Establish database pool and HTTP client."""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=4)
        self.http_client = httpx.AsyncClient(timeout=15.0)
        logger.info(f"Dreamer daemon initialized against {self.db_url}")

    async def cleanup(self):
        """Cleanly close resources."""
        if self.pool:
            await self.pool.close()
        if self.http_client:
            await self.http_client.aclose()
        logger.info("Dreamer daemon cleaned up")

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate 1024-dim dense vector using local Metal GPU embedder."""
        if not self.http_client:
            return None
        try:
            resp = await self.http_client.post(self.embed_url, json={"text": text})
            if resp.status_code == 200:
                data = resp.json()
                if "embedding" in data and len(data["embedding"]) == EMBEDDING_DIM:
                    return data["embedding"]
        except Exception as e:
            logger.debug(f"Local embedding server unavailable ({e}); proceeding without vector")
        return None

    def _format_vector(self, embedding: List[float]) -> str:
        """Format embedding as PostgreSQL vector string."""
        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        elif len(embedding) < EMBEDDING_DIM:
            embedding = embedding + [0.0] * (EMBEDDING_DIM - len(embedding))
        return "[" + ",".join(str(e) for e in embedding) + "]"

    async def fetch_episodes(
        self, limit: int = 20, lookback_hours: int = 72
    ) -> List[Dict[str, Any]]:
        """Fetch closed episodes with their chronological event streams."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        async with self.pool.acquire() as conn:
            ep_rows = await conn.fetch(
                """
                SELECT id, session_id, task_type, agent_id, hunt_context,
                       pressure, title, goal, outcome, narrative,
                       retrospective, metadata, created_at, ended_at
                FROM nah_remember.episodes
                WHERE ended_at IS NOT NULL
                  AND ended_at >= $1
                ORDER BY ended_at DESC
                LIMIT $2
                """,
                cutoff,
                limit,
            )

            episodes = []
            for r in ep_rows:
                ep = dict(r)
                event_rows = await conn.fetch(
                    """
                    SELECT event_index, event_type, description, causal_parent_id,
                           tool_name, tool_input, tool_result, metadata, created_at
                    FROM nah_remember.episode_events
                    WHERE episode_id = $1
                    ORDER BY event_index ASC
                    """,
                    ep["id"],
                )
                ep["events"] = [dict(ev) for ev in event_rows]
                episodes.append(ep)

        return episodes

    @staticmethod
    def _score_episode(ep: Dict[str, Any]) -> int:
        """Derive discrete performance/recall score r in {1..5} from measured episode telemetry."""
        outcome = (ep.get("outcome") or "").lower()
        if outcome in ("failure", "aborted"):
            return 1
        events = ep.get("events") or []
        error_count = sum(
            1
            for ev in events
            if ev.get("event_type") in ("error", "failure", "timeout", "retry")
            or (ev.get("details") and "error" in str(ev["details"]).lower())
            or (ev.get("metadata") and "error" in str(ev["metadata"]).lower())
        )
        pressure = float(ep.get("pressure", 0.0) or 0.0)
        score = 5
        score -= min(3, error_count)
        if pressure > 0.75:
            score -= 1
        return max(1, min(5, score))

    @staticmethod
    def _score_constraint_observation(ep: Dict[str, Any], pat: str) -> int:
        """Derive discrete evidence score r in {1..5} for a recurring failure pattern."""
        events = ep.get("events") or []
        matching_events = [
            ev
            for ev in events
            if (ev.get("description") and pat in str(ev.get("description")))
            or (ev.get("tool_name") and pat in str(ev.get("tool_name")))
            or (ev.get("event_type") in ("error", "failure", "timeout"))
        ]
        if not matching_events:
            return 1
        has_details = any(
            ev.get("tool_result") or ev.get("details") or ev.get("metadata")
            for ev in matching_events
        )
        has_blocking = any(ev.get("event_type") in ("error", "failure") for ev in matching_events)
        if has_details and has_blocking:
            return 5 if len(matching_events) >= 2 else 4
        elif has_blocking:
            return 3
        return 2

    async def measure_probe_scores(
        self, proc: Dict[str, Any], candidate_episodes: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[float]]:
        """Measure real prior baseline vs candidate post probe scores from episode telemetry."""
        # 1. Candidate scores from current synthesized episodes
        candidate_scores = []
        if candidate_episodes:
            for ep in candidate_episodes:
                r = self._score_episode(ep)
                candidate_scores.append((r - 1.0) / 4.0)
        else:
            candidate_scores = [1.0, 1.0]

        # 2. Baseline scores from historical episodes or existing procedures
        baseline_scores = []
        task_type = proc.get("task_type") or (
            proc.get("trigger_conditions", [""])[0] if proc.get("trigger_conditions") else "general"
        )
        if self.pool:
            try:
                async with self.pool.acquire() as conn:
                    # Check if there is an existing procedure for this task
                    existing_sr = await conn.fetchval(
                        "SELECT success_rate FROM nah_remember.procedures WHERE name ILIKE $1 OR description ILIKE $1",
                        f"%{task_type}%",
                    )
                    if existing_sr is not None:
                        baseline_scores = [float(existing_sr), float(existing_sr)]
                    else:
                        # Fetch earlier episodes not in current candidate batch
                        exclude_ids = [
                            ep["id"]
                            for ep in candidate_episodes
                            if isinstance(ep.get("id"), uuid.UUID)
                        ]
                        rows = await conn.fetch(
                            """
                            SELECT outcome, pressure,
                                   (SELECT count(*) FROM nah_remember.episode_events ev 
                                    WHERE ev.episode_id = e.id AND ev.event_type IN ('error', 'failure', 'timeout', 'retry')) as error_count
                            FROM nah_remember.episodes e
                            WHERE id NOT IN (SELECT unnest($1::uuid[]))
                            ORDER BY created_at DESC
                            LIMIT 5
                            """,
                            exclude_ids,
                        )
                        if rows:
                            for r in rows:
                                ep_mock = {
                                    "outcome": r["outcome"],
                                    "pressure": r["pressure"],
                                    "events": [{"event_type": "error"}] * r["error_count"],
                                }
                                baseline_scores.append((self._score_episode(ep_mock) - 1.0) / 4.0)
            except Exception as e:
                logger.debug(f"Failed to query baseline episodes: {e}")

        if not baseline_scores:
            # Conservative baseline: 0.5 (r=3) across probe rounds
            baseline_scores = [0.5, 0.5]

        # Probe candidate via independent neural judge probe
        try:
            probe_score = await neural_judge_probe(
                candidate=proc,
                probe_prompt=f"Assess workflow efficiency and safety for playbook {proc.get('name')}",
                judge_model=self.judge_model,
            )
            candidate_scores.append(probe_score)
        except Exception as e:
            logger.debug(f"Neural judge probe execution skipped: {e}")

        return baseline_scores, candidate_scores

    def synthesize_candidates(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize candidate procedural playbooks from successful episodes
        and candidate negative constraints from failed episodes.
        """
        candidate_procedures = []
        candidate_constraints = []
        observations: Dict[str, List[int]] = {}
        deltas: Dict[str, Decimal] = {}

        # 1. Group successful episodes by task_type or goal patterns
        successful_episodes = [ep for ep in episodes if ep.get("outcome") == "success"]
        task_groups: Dict[str, List[Dict[str, Any]]] = {}

        for ep in successful_episodes:
            tt = ep.get("task_type") or "general_task"
            task_groups.setdefault(tt, []).append(ep)

        for task_type, eps in task_groups.items():
            # Build sequence of tools / key beats
            primary_ep = eps[0]
            steps = []
            for ev in primary_ep.get("events", []):
                if ev.get("tool_name"):
                    steps.append(
                        {
                            "tool": ev["tool_name"],
                            "description": ev.get("description", ""),
                            "expected_outcome": "success",
                        }
                    )

            if steps:
                proc_mid = f"proc-{task_type}"
                candidate_procedures.append(
                    {
                        "memory_id": proc_mid,
                        "name": f"Playbook: {task_type.replace('_', ' ').title()}",
                        "description": primary_ep.get("narrative")
                        or f"Standard procedure for {task_type}",
                        "steps": steps,
                        "trigger_conditions": [task_type, primary_ep.get("hunt_context", "")],
                        "agent_id": primary_ep.get("agent_id", "system"),
                        "episodes_count": len(eps),
                        "task_type": task_type,
                        "episodes": eps,
                    }
                )
                # Map discrete recall scores derived from measured episode telemetry
                observations[proc_mid] = [self._score_episode(ep) for ep in eps]
                deltas[proc_mid] = Decimal("0.5")

        # 2. Extract recurring negative constraints from failed episodes
        failed_episodes = [ep for ep in episodes if ep.get("outcome") in ("failure", "aborted")]
        failure_patterns: Dict[str, List[Dict[str, Any]]] = {}

        for ep in failed_episodes:
            for ev in ep.get("events", []):
                if (
                    ev.get("event_type") in ("error", "failure", "timeout")
                    or (ev.get("details") and "error" in str(ev["details"]).lower())
                    or (ev.get("metadata") and "error" in str(ev["metadata"]).lower())
                ):
                    pattern_key = ev.get("description") or ev.get("tool_name") or "general_error"
                    failure_patterns.setdefault(pattern_key[:100], []).append(ep)

        for pat, eps in failure_patterns.items():
            # Deterministic SHA-256 for stable IDs across restarts and processes
            hash_suffix = hashlib.sha256(pat.encode("utf-8")).hexdigest()[:12]
            con_mid = f"const-{hash_suffix}"
            candidate_constraints.append(
                {
                    "memory_id": con_mid,
                    "pattern": pat,
                    "trigger_context": eps[0].get("hunt_context", ""),
                    "severity": "warning" if len(eps) < 3 else "blocking",
                    "evidence_count": len(eps),
                    "episodes": eps,
                }
            )
            # Derive discrete evidence score stream from failure telemetry
            observations[con_mid] = [self._score_constraint_observation(ep, pat) for ep in eps]
            deltas[con_mid] = Decimal("-0.5")

        # Balance deltas for trial balance conservation: sum(deltas) must balance with matching ledger
        ledger_entries = []
        for mid, d in deltas.items():
            ledger_entries.append(
                {
                    "event_id": make_event_id("dreamer_run", 1, mid),
                    "source": "dreamer_engine",
                    "round": 1,
                    "memory_id": mid,
                    "delta": d,
                    "basis": "episodic_synthesis",
                }
            )

        candidate = {
            "producer": EXPECTED_PRODUCER,
            "deltas": deltas,
            "procedures": candidate_procedures,
            "constraints": candidate_constraints,
        }

        return {
            "candidate": candidate,
            "ledger": ledger_entries,
            "observations": observations,
        }

    async def run_consolidation_cycle(
        self, dry_run: bool = False, limit: int = 20
    ) -> Dict[str, Any]:
        """
        Execute full consolidation cycle with Wald SPRT, conservation trial balance,
        and rotated-judge wake gate verification.
        """
        logger.info(f"Starting consolidation cycle (dry_run={dry_run}, limit={limit})")
        episodes = await self.fetch_episodes(limit=limit)
        if not episodes:
            logger.info("No completed episodes found in lookback window; cycle complete")
            return {
                "success": True,
                "episodes_analyzed": 0,
                "procedures_promoted": 0,
                "constraints_promoted": 0,
                "quarantined": 0,
                "waiting": 0,
                "dry_run": dry_run,
            }

        synth = self.synthesize_candidates(episodes)
        candidate = synth["candidate"]
        ledger = synth["ledger"]
        observations = synth["observations"]

        # Check live universe size and enforce Layer 4 blast radius before promotions
        universe_size = 100
        if self.pool:
            try:
                async with self.pool.acquire() as conn:
                    p_count = (
                        await conn.fetchval("SELECT count(*) FROM nah_remember.procedures") or 0
                    )
                    e_count = await conn.fetchval("SELECT count(*) FROM nah_remember.episodes") or 0
                    universe_size = max(10, p_count + e_count)
            except Exception as e:
                logger.debug(f"Failed to fetch live universe size: {e}")

        blast_ok, blast_reason = check_blast_radius(
            candidate, universe_size=universe_size, blast_radius_frac=0.20
        )
        if not blast_ok:
            logger.warning(f"Consolidation cycle aborted by blast radius check: {blast_reason}")
            return {
                "success": False,
                "reason": blast_reason,
                "episodes_analyzed": len(episodes),
                "procedures_promoted": 0,
                "constraints_promoted": 0,
                "quarantined": 0,
                "waiting": len(candidate["procedures"]) + len(candidate["constraints"]),
                "dry_run": dry_run,
            }

        promotions = []
        quarantines = []
        waitings = []

        # 1. Gating each candidate procedure through Wald SPRT
        for proc in candidate["procedures"]:
            mid = proc["memory_id"]
            stream = observations.get(mid, [3, 3])
            sprt_res = sufficient(stream, prior=0.5)

            if not sprt_res["sufficient"]:
                logger.info(
                    f"Candidate procedure {proc['name']} deferred: SPRT decision={sprt_res['decision']}"
                )
                waitings.append({"item": proc, "reason": "sprt_insufficient", "sprt": sprt_res})
                continue

            # 2. Evaluate candidate through independent wake gate with measured probe scores
            baseline_scores, candidate_scores = await self.measure_probe_scores(
                proc, proc.get("episodes", [])
            )
            gate_res = evaluate_wake_gate(
                candidate=candidate,
                ledger=ledger,
                baseline_scores=baseline_scores,
                candidate_scores=candidate_scores,
                synthesizer_model=self.synthesizer_model,
                judge_model=self.judge_model,
                sealed_anchors=DEFAULT_SEALED_ANCHORS,
                universe_size=universe_size,
            )

            if gate_res["verdict"] == "promote":
                promotions.append({"type": "procedure", "data": proc, "gate": gate_res})
            elif gate_res["verdict"] == "quarantine":
                quarantines.append({"type": "procedure", "data": proc, "gate": gate_res})
            else:
                waitings.append({"type": "procedure", "data": proc, "gate": gate_res})

        # 3. Gating candidate negative constraints through Wald SPRT
        for constr in candidate["constraints"]:
            mid = constr["memory_id"]
            stream = observations.get(mid, [3, 3])
            sprt_res = sufficient(stream, prior=0.5)

            if sprt_res["sufficient"]:
                promotions.append({"type": "constraint", "data": constr, "sprt": sprt_res})
            else:
                waitings.append({"type": "constraint", "data": constr, "sprt": sprt_res})

        # 4. Commit promotions to database
        committed_procs = 0
        committed_constrs = 0

        if not dry_run and promotions:
            async with self.pool.acquire() as conn:
                for item in promotions:
                    if item["type"] == "procedure":
                        proc = item["data"]
                        text_to_embed = f"{proc['name']}: {proc['description']}"
                        emb = await self.generate_embedding(text_to_embed)
                        emb_str = self._format_vector(emb) if emb else None

                        proc_id = uuid.uuid4()
                        triggers_json = json.dumps(proc.get("trigger_conditions", []))
                        await conn.execute(
                            """
                            INSERT INTO nah_remember.procedures (
                                id, name, procedure_type, description, steps,
                                trigger_conditions, execution_count, success_rate,
                                embedding, metadata, status, created_at, updated_at
                            ) VALUES (
                                $1, $2, 'workflow', $3, $4::jsonb,
                                $5::jsonb, 1, 1.0,
                                CASE WHEN $6::text IS NOT NULL THEN $6::vector ELSE NULL END,
                                '{"source": "dreamer_daemon"}'::jsonb, 'active', NOW(), NOW()
                            )
                            ON CONFLICT DO NOTHING
                            """,
                            proc_id,
                            proc["name"],
                            proc["description"],
                            json.dumps(proc["steps"]),
                            triggers_json,
                            emb_str,
                        )
                        committed_procs += 1

                    elif item["type"] == "constraint":
                        constr = item["data"]
                        await conn.execute(
                            """
                            INSERT INTO nah_remember.negative_constraints (
                                pattern, trigger_context, severity, weight, created_at, hit_count, last_hit_at
                            ) VALUES ($1, $2, $3, 0.85, NOW(), 1, NOW())
                            ON CONFLICT (pattern, trigger_context) DO UPDATE SET
                                hit_count = nah_remember.negative_constraints.hit_count + 1,
                                last_hit_at = NOW(),
                                weight = LEAST(1.0, nah_remember.negative_constraints.weight + 0.05)
                            """,
                            constr["pattern"],
                            constr.get("trigger_context", ""),
                            constr.get("severity", "warning"),
                        )
                        committed_constrs += 1

        summary = {
            "success": True,
            "episodes_analyzed": len(episodes),
            "candidates_found": len(candidate["procedures"]) + len(candidate["constraints"]),
            "procedures_promoted": committed_procs
            if not dry_run
            else len([p for p in promotions if p["type"] == "procedure"]),
            "constraints_promoted": committed_constrs
            if not dry_run
            else len([p for p in promotions if p["type"] == "constraint"]),
            "quarantined": len(quarantines),
            "waiting": len(waitings),
            "dry_run": dry_run,
        }
        logger.info(f"Consolidation cycle completed: {summary}")
        return summary


async def main():
    parser = argparse.ArgumentParser(description="nah-remember Dreamer Consolidation Daemon")
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing to DB")
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of recent episodes to analyze"
    )
    parser.add_argument(
        "--interval", type=int, default=21600, help="Daemon interval in seconds (default: 6h)"
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Run continuously as background daemon"
    )
    args = parser.parse_args()

    daemon = DreamerDaemon()
    await daemon.setup()

    try:
        if args.daemon:
            logger.info(f"Running Dreamer daemon in continuous loop (interval: {args.interval}s)")
            while True:
                await daemon.run_consolidation_cycle(dry_run=args.dry_run, limit=args.limit)
                await asyncio.sleep(args.interval)
        else:
            res = await daemon.run_consolidation_cycle(dry_run=args.dry_run, limit=args.limit)
            print(json.dumps(res, indent=2))
    finally:
        await daemon.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
