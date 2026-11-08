#!/usr/bin/env python3
"""
NAH Neural Memory Core: Retrieval + Query Routing with Bet 2 Ranking
Subsystem: nah-remember (v2.0.0)
Architecture:
- Ephemeral layer: Recent messages (< 48h, high recency weight)
- Episodic layer: True episodes (narrative, timeline beats, cause/effect, outcomes) + situational context
- Semantic layer: Semantic memories + entity knowledge graph
- Procedural layer: Learned playbooks, procedures, dynamic success rates
- Meta layer: Entity relationships, memory evolution, negative constraints

Ranking formula (Bet 2 Hybrid + Situational Affinity):
  MatchScore = similarity * 0.4 + significance * 0.3 + recency * 0.2 + access * 0.1 (+ situational_affinity * 0.2)
  CompositeScore = MatchScore * (1.0 + beta * reinforcement_score)
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
import uuid
import json

import asyncpg

logger = logging.getLogger(__name__)

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
DEFAULT_BETA = float(os.getenv("BET2_BETA", "1.2"))
DEFAULT_EPHEMERAL_HOURS = int(os.getenv("NAH_EPHEMERAL_WINDOW_HOURS", "48"))


class MemoryLayer(str, Enum):
    """Memory layers for query routing"""

    EPHEMERAL = "ephemeral"  # Recent messages (< 48h)
    EPISODIC = "episodic"  # True episodes & situational context
    SEMANTIC = "semantic"  # Semantic memories & entity graph
    PROCEDURAL = "procedural"  # Playbooks & executable skills
    META = "meta"  # Lineage, relationships, constraints
    ALL = "all"  # All layers


class RankedResult:
    """Ranked result with Bet 2 composite scoring and situational affinity"""

    def __init__(
        self,
        memory_type: str,
        memory_id: str,
        content: str,
        layer: MemoryLayer,
        significance_score: float = 0.5,
        access_count: int = 0,
        reinforcement_score: float = 0.0,
        created_at: Optional[datetime] = None,
        recency_window_days: int = 30,
        metadata: Optional[Dict] = None,
        similarity: Optional[float] = None,
        situational_affinity: float = 0.0,
    ):
        self.memory_type = memory_type
        self.memory_id = memory_id
        self.content = content
        self.layer = layer
        self.significance_score = min(1.0, max(0.0, float(significance_score or 0.5)))
        self.reinforcement_score = min(1.0, max(0.0, float(reinforcement_score or 0.0)))
        self.access_count = max(0, int(access_count or 0))
        self.created_at = created_at or datetime.now(timezone.utc)
        self.metadata = metadata or {}
        self.similarity = min(1.0, max(0.0, float(similarity or 0.0)))
        self.situational_affinity = min(1.0, max(0.0, float(situational_affinity or 0.0)))

        # Normalize recency score (0.0 - 1.0)
        now = datetime.now(timezone.utc)
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).days
        self.recency_score = max(0.0, 1.0 - (age_days / max(1, recency_window_days)))

        # Normalize access count logarithmically
        self.access_count_score = min(1.0, (1.0 + self.access_count) / 10.0)
        self.composite_score = 0.0

    def compute_composite_score(
        self,
        similarity_weight: float = 0.4,
        significance_weight: float = 0.3,
        recency_weight: float = 0.2,
        access_weight: float = 0.1,
        beta: float = DEFAULT_BETA,
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute Bet 2 composite ranking score.
        Formula:
          MatchScore = (sim * w_sim + sig * w_sig + rec * w_rec + acc * w_acc) + (affinity * 0.2)
          CompositeScore = MatchScore * (1.0 + beta * reinforcement_score)

        Contract:
          composite_score >= 0.0. CompositeScore is an unbounded relevance ranking score
          (not a probability) scaled by usage reinforcement. Floor at 0.0 prevents negative scores.
        """
        # If weights dictionary passed directly, unpack it
        if weights:
            similarity_weight = weights.get("similarity", similarity_weight)
            significance_weight = weights.get("significance", significance_weight)
            recency_weight = weights.get("recency", recency_weight)
            access_weight = weights.get("access", weights.get("access_count", access_weight))
            beta = weights.get("beta", beta)

        # Gracefully handle legacy keyword arguments if passed
        if "access_count_weight" in kwargs:
            access_weight = kwargs["access_count_weight"]
        if "reinforcement_weight" in kwargs and kwargs["reinforcement_weight"] is not None:
            # Legacy caller expecting additive reinforcement
            pass

        total_weight = similarity_weight + significance_weight + recency_weight + access_weight
        if abs(total_weight - 1.0) > 0.02 and total_weight > 0:
            similarity_weight /= total_weight
            significance_weight /= total_weight
            recency_weight /= total_weight
            access_weight /= total_weight

        # Clamp inputs to safe intervals (Fix for R4)
        sim = min(1.0, max(0.0, float(self.similarity)))
        sig = min(1.0, max(0.0, float(self.significance_score)))
        rec = min(1.0, max(0.0, float(self.recency_score)))
        acc = min(1.0, max(0.0, float(self.access_count_score)))
        aff = min(1.0, max(0.0, float(self.situational_affinity)))
        reinf = min(1.0, max(0.0, float(self.reinforcement_score)))
        beta_clamped = max(0.0, float(beta))

        base_match = (
            sim * similarity_weight
            + sig * significance_weight
            + rec * recency_weight
            + acc * access_weight
            + (aff * 0.2)
        )

        # Bet 2 usage reinforcement multiplier with non-negative lower bound
        score = max(0.0, base_match * (1.0 + (beta_clamped * reinf)))
        self.composite_score = float(score)
        return self.composite_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            "memory_type": self.memory_type,
            "memory_id": self.memory_id,
            "content": self.content[:1000] + "..." if len(self.content) > 1000 else self.content,
            "layer": self.layer.value,
            "composite_score": float(self.composite_score or self.compute_composite_score()),
            "significance_score": float(self.significance_score),
            "reinforcement_score": float(self.reinforcement_score),
            "access_count": int(self.access_count),
            "recency_score": float(self.recency_score),
            "similarity": float(self.similarity),
            "situational_affinity": float(self.situational_affinity),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }


class NahRememberPhase3:
    """nah-neural Memory Retrieval: Bet 2 Hybrid Ranking + 5 Cognitive Layers"""

    def __init__(self, pool: asyncpg.pool.Pool):
        self.pool = pool
        logger.info("nah-neural Phase 3 Engine initialized with Bet 2 ranking")

    # ==================== SITUATIONAL AFFINITY ====================

    @staticmethod
    def calculate_situational_affinity(
        target_task_type: Optional[str],
        target_hunt_context: Optional[str],
        target_agent_id: Optional[str],
        target_pressure: Optional[float],
        spore_task_type: Optional[str],
        spore_hunt_context: Optional[str],
        spore_agent_id: Optional[str],
        spore_pressure: Optional[float],
    ) -> float:
        """
        Calculate condition-signal affinity between query signal and memory engram.
        Returns 0.0 - 1.0.
        """
        score = 0.0
        if (
            target_task_type
            and spore_task_type
            and target_task_type.lower() == spore_task_type.lower()
        ):
            score += 0.40
        if (
            target_hunt_context
            and spore_hunt_context
            and target_hunt_context.lower() == spore_hunt_context.lower()
        ):
            score += 0.25
        if target_agent_id and spore_agent_id and target_agent_id.lower() == spore_agent_id.lower():
            score += 0.20
        if target_pressure is not None and spore_pressure is not None:
            delta = abs(float(target_pressure) - float(spore_pressure))
            score += max(0.0, 1.0 - delta) * 0.15
        return min(1.0, max(0.0, score))

    # ==================== LAYER QUERIES ====================

    async def query_ephemeral(
        self,
        query_embedding: List[float],
        embedding_dim: int = EMBEDDING_DIM,
        hours: int = DEFAULT_EPHEMERAL_HOURS,
        limit: int = 10,
    ) -> List[RankedResult]:
        """Query ephemeral layer: recent messages (< hours old, default 48h)"""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            embedding_str = self._format_vector(query_embedding, embedding_dim)

            query = """
                SELECT
                    'message' as memory_type,
                    m.message_id::text as memory_id,
                    m.content,
                    m.importance_score,
                    0 as access_count,
                    0.0 as reinforcement_score,
                    m.created_at,
                    1 - (m.embedding <=> $1::vector) as similarity
                FROM nah_remember.messages m
                WHERE m.created_at > $2
                    AND m.embedding IS NOT NULL
                ORDER BY m.embedding <=> $1::vector
                LIMIT $3
            """

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, embedding_str, cutoff, limit)

            return [
                RankedResult(
                    memory_type=r["memory_type"],
                    memory_id=r["memory_id"],
                    content=r["content"],
                    layer=MemoryLayer.EPHEMERAL,
                    significance_score=r["importance_score"],
                    access_count=r["access_count"],
                    reinforcement_score=r["reinforcement_score"],
                    created_at=r["created_at"],
                    similarity=r["similarity"],
                )
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Ephemeral query failed: {e}")
            return []

    async def query_episodic(
        self,
        query_embedding: List[float],
        embedding_dim: int = EMBEDDING_DIM,
        task_type: Optional[str] = None,
        hunt_context: Optional[str] = None,
        agent_id: Optional[str] = None,
        pressure: Optional[float] = None,
        limit: int = 10,
    ) -> List[RankedResult]:
        """Query episodic layer: true episodes with situational affinity matching"""
        results = []
        try:
            embedding_str = self._format_vector(query_embedding, embedding_dim)

            async with self.pool.acquire() as conn:
                # 1. Query true episodes
                episode_query = """
                    SELECT
                        'episode' as memory_type,
                        e.id::text as memory_id,
                        coalesce(e.narrative, e.title || ': ' || e.goal) as content,
                        0.8 as importance_score,
                        e.access_count,
                        coalesce(e.reinforcement_score, 0.0) as reinforcement_score,
                        e.created_at,
                        e.task_type,
                        e.hunt_context,
                        e.agent_id,
                        e.pressure,
                        e.outcome,
                        CASE 
                            WHEN e.embedding IS NOT NULL THEN (1 - (e.embedding <=> $1::vector))
                            ELSE 0.5 
                        END as similarity
                    FROM nah_remember.episodes e
                    WHERE ($3::text IS NULL OR e.task_type = $3)
                      AND ($4::text IS NULL OR e.agent_id = $4)
                      AND ($5::text IS NULL OR e.hunt_context = $5)
                    ORDER BY 
                        CASE WHEN e.embedding IS NOT NULL THEN (e.embedding <=> $1::vector) ELSE 1.0 END ASC,
                        e.created_at DESC
                    LIMIT $2
                """
                episode_rows = await conn.fetch(
                    episode_query, embedding_str, limit, task_type, agent_id, hunt_context
                )

                for r in episode_rows:
                    affinity = self.calculate_situational_affinity(
                        task_type,
                        hunt_context,
                        agent_id,
                        pressure,
                        r["task_type"],
                        r["hunt_context"],
                        r["agent_id"],
                        r["pressure"],
                    )
                    results.append(
                        RankedResult(
                            memory_type=r["memory_type"],
                            memory_id=r["memory_id"],
                            content=r["content"] or "",
                            layer=MemoryLayer.EPISODIC,
                            significance_score=r["importance_score"],
                            access_count=r["access_count"],
                            reinforcement_score=r["reinforcement_score"],
                            created_at=r["created_at"],
                            similarity=r["similarity"],
                            situational_affinity=affinity,
                            metadata={"outcome": r["outcome"], "task_type": r["task_type"]},
                        )
                    )

                # 2. Context snapshots as situational context fallback
                if len(results) < limit:
                    remaining = limit - len(results)
                    snapshot_query = """
                        SELECT
                            'context_snapshot' as memory_type,
                            cs.id::text as memory_id,
                            cs.context_summary as content,
                            0.6 as importance_score,
                            0 as access_count,
                            0.0 as reinforcement_score,
                            cs.created_at
                        FROM nah_remember.context_snapshots cs
                        WHERE cs.context_summary IS NOT NULL
                        ORDER BY cs.created_at DESC
                        LIMIT $1
                    """
                    snapshot_rows = await conn.fetch(snapshot_query, remaining)
                    for r in snapshot_rows:
                        results.append(
                            RankedResult(
                                memory_type=r["memory_type"],
                                memory_id=r["memory_id"],
                                content=r["content"] or "",
                                layer=MemoryLayer.EPISODIC,
                                significance_score=r["importance_score"],
                                access_count=r["access_count"],
                                reinforcement_score=r["reinforcement_score"],
                                created_at=r["created_at"],
                                similarity=0.5,
                            )
                        )

            return results
        except Exception as e:
            logger.error(f"Episodic query failed: {e}")
            return []

    async def query_semantic(
        self,
        query_embedding: List[float],
        embedding_dim: int = EMBEDDING_DIM,
        limit: int = 10,
    ) -> List[RankedResult]:
        """Query semantic layer: semantic memories + knowledge graph entities"""
        results = []
        try:
            embedding_str = self._format_vector(query_embedding, embedding_dim)

            async with self.pool.acquire() as conn:
                # 1. Semantic memories
                memory_query = """
                    SELECT
                        'semantic_memory' as memory_type,
                        sm.id::text as memory_id,
                        sm.content,
                        sm.importance_score,
                        sm.access_count,
                        coalesce(sm.reinforcement_score, 0.0) as reinforcement_score,
                        sm.created_at,
                        1 - (sm.embedding <=> $1::vector) as similarity
                    FROM nah_remember.semantic_memories sm
                    WHERE sm.embedding IS NOT NULL
                    ORDER BY sm.embedding <=> $1::vector
                    LIMIT $2
                """
                memory_rows = await conn.fetch(memory_query, embedding_str, limit)

                # 2. Entities (world-model concepts/entities)
                entity_query = """
                    SELECT
                        'entity' as memory_type,
                        e.id::text as memory_id,
                        coalesce(e.description, e.name) as content,
                        e.importance_score,
                        e.access_count,
                        0.0 as reinforcement_score,
                        e.created_at,
                        1 - (e.embedding <=> $1::vector) as similarity
                    FROM nah_remember.entities e
                    WHERE e.embedding IS NOT NULL
                    ORDER BY e.embedding <=> $1::vector
                    LIMIT $2
                """
                entity_rows = await conn.fetch(entity_query, embedding_str, limit)

            for r in memory_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"],
                        layer=MemoryLayer.SEMANTIC,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        reinforcement_score=r["reinforcement_score"],
                        created_at=r["created_at"],
                        similarity=r["similarity"],
                    )
                )

            for r in entity_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.SEMANTIC,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        reinforcement_score=r["reinforcement_score"],
                        created_at=r["created_at"],
                        similarity=r["similarity"],
                    )
                )

            return results
        except Exception as e:
            logger.error(f"Semantic query failed: {e}")
            return []

    async def query_procedural(
        self,
        query_embedding: List[float],
        embedding_dim: int = EMBEDDING_DIM,
        limit: int = 10,
    ) -> List[RankedResult]:
        """Query procedural layer: executable procedures and playbooks"""
        results = []
        try:
            embedding_str = self._format_vector(query_embedding, embedding_dim)

            async with self.pool.acquire() as conn:
                procedure_query = """
                    SELECT
                        'procedure' as memory_type,
                        p.id::text as memory_id,
                        coalesce(p.description, p.name) as content,
                        0.7 as importance_score,
                        p.execution_count as access_count,
                        coalesce(p.success_rate, 0.0) as reinforcement_score,
                        p.created_at,
                        1 - (p.embedding <=> $1::vector) as similarity,
                        p.metadata
                    FROM nah_remember.procedures p
                    WHERE p.embedding IS NOT NULL
                    ORDER BY p.embedding <=> $1::vector
                    LIMIT $2
                """
                rows = await conn.fetch(procedure_query, embedding_str, limit)

            for r in rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.PROCEDURAL,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        reinforcement_score=r["reinforcement_score"],
                        created_at=r["created_at"],
                        similarity=r["similarity"],
                        metadata=r["metadata"] or {},
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Procedural query failed: {e}")
            return []

    async def query_meta(self, limit: int = 10) -> List[RankedResult]:
        """Query meta layer: entity relationships + evolution tracking"""
        results = []
        try:
            async with self.pool.acquire() as conn:
                rel_query = """
                    SELECT
                        'relationship' as memory_type,
                        r.id::text as memory_id,
                        r.description as content,
                        CASE WHEN r.strength > 0.7 THEN 0.8 ELSE 0.5 END as importance_score,
                        0 as access_count,
                        0.0 as reinforcement_score,
                        r.created_at
                    FROM nah_remember.relationships r
                    ORDER BY r.strength DESC
                    LIMIT $1
                """
                rel_rows = await conn.fetch(rel_query, limit)

                history_query = """
                    SELECT
                        'memory_evolution' as memory_type,
                        mh.id::text as memory_id,
                        mh.new_content as content,
                        0.6 as importance_score,
                        0 as access_count,
                        0.0 as reinforcement_score,
                        mh.created_at
                    FROM nah_remember.memory_history mh
                    WHERE mh.change_type = 'update'
                    ORDER BY mh.created_at DESC
                    LIMIT $1
                """
                history_rows = await conn.fetch(history_query, limit)

            for r in rel_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.META,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        reinforcement_score=r["reinforcement_score"],
                        created_at=r["created_at"],
                    )
                )

            for r in history_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.META,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        reinforcement_score=r["reinforcement_score"],
                        created_at=r["created_at"],
                    )
                )

            return results
        except Exception as e:
            logger.error(f"Meta query failed: {e}")
            return []

    # ==================== EPISODIC PRIMITIVES ====================

    async def open_episode(
        self,
        session_id: Optional[str] = None,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        hunt_context: Optional[str] = None,
        initial_pressure: float = 0.5,
        title: str = "Agent Episode",
        goal: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Open a new episodic memory container. Returns episode UUID string."""
        ep_uuid = uuid.uuid4()
        meta_json = json.dumps(metadata or {})
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO nah_remember.episodes (
                    id, session_id, agent_id, title, goal, task_type,
                    pressure, hunt_context, outcome, metadata, started_at, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'in_progress', $9::jsonb, NOW(), NOW(), NOW())
                """,
                ep_uuid,
                session_id,
                agent_id,
                title,
                goal or title,
                task_type,
                float(initial_pressure),
                hunt_context,
                meta_json,
            )
        logger.info(f"Opened episode {ep_uuid} for agent {agent_id} (task: {task_type})")
        return str(ep_uuid)

    async def append_episode_event(
        self,
        episode_id: str,
        event_type: str,
        description: str = "",
        details: Optional[Dict[str, Any]] = None,
        pressure: Optional[float] = None,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append a discrete causal event to an active episode. Returns event sequence index."""
        valid_types = {"action", "observation", "error", "pivot", "breakthrough", "decision"}
        norm_type = event_type.lower() if event_type.lower() in valid_types else "action"
        desc = description or (json.dumps(details) if details else f"Event {norm_type}")
        meta_json = json.dumps(details or {})
        t_input_json = json.dumps(tool_input) if tool_input else None
        t_res_json = json.dumps(tool_result) if tool_result else None

        ep_uuid = uuid.UUID(episode_id)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(MAX(event_index), -1) + 1 as next_idx FROM nah_remember.episode_events WHERE episode_id = $1",
                ep_uuid,
            )
            next_idx = row["next_idx"] if row else 0

            await conn.execute(
                """
                INSERT INTO nah_remember.episode_events (
                    episode_id, event_index, event_type, description,
                    tool_name, tool_input, tool_result, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, NOW())
                """,
                ep_uuid,
                next_idx,
                norm_type,
                desc,
                tool_name,
                t_input_json,
                t_res_json,
                meta_json,
            )

            if pressure is not None:
                await conn.execute(
                    "UPDATE nah_remember.episodes SET pressure = $1, updated_at = NOW() WHERE id = $2",
                    float(pressure),
                    ep_uuid,
                )

        return next_idx + 1

    async def close_episode(
        self,
        episode_id: str,
        outcome: str = "success",
        final_summary: Optional[str] = None,
        retrospective: Optional[str] = None,
        lessons_learned: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Close an episode and mark its outcome."""
        norm_outcome = outcome.lower()
        if norm_outcome not in ("success", "failure", "partial", "aborted", "in_progress"):
            norm_outcome = "success" if "success" in norm_outcome else "failure"

        retro = retrospective or ""
        if lessons_learned:
            retro = (retro + "\nLessons: " + "; ".join(lessons_learned)).strip()

        emb_str = self._format_vector(embedding) if embedding else None
        ep_uuid = uuid.UUID(episode_id)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE nah_remember.episodes
                SET outcome = $1,
                    narrative = COALESCE($2, narrative),
                    retrospective = COALESCE($3, retrospective),
                    embedding = CASE WHEN $4::text IS NOT NULL THEN $4::vector ELSE embedding END,
                    ended_at = NOW(),
                    updated_at = NOW()
                WHERE id = $5
                """,
                norm_outcome,
                final_summary,
                retro if retro else None,
                emb_str,
                ep_uuid,
            )
        logger.info(f"Closed episode {ep_uuid} with outcome {norm_outcome}")
        return True

    # ==================== UNIFIED RANKED SEARCH ====================

    async def remember_search_ranked(
        self,
        query_embedding: List[float],
        memory_type: Literal[
            "ephemeral", "episodic", "semantic", "procedural", "meta", "all"
        ] = "all",
        limit: int = 10,
        embedding_dim: int = EMBEDDING_DIM,
        similarity_weight: float = 0.4,
        significance_weight: float = 0.3,
        recency_weight: float = 0.2,
        access_weight: float = 0.1,
        beta: float = DEFAULT_BETA,
        task_type: Optional[str] = None,
        hunt_context: Optional[str] = None,
        agent_id: Optional[str] = None,
        pressure: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Unified ranked search across memory layers using Bet 2 formula.
        """
        try:
            # Normalize weights to sum to 1.0
            total_w = similarity_weight + significance_weight + recency_weight + access_weight
            if total_w > 0:
                similarity_weight = similarity_weight / total_w
                significance_weight = significance_weight / total_w
                recency_weight = recency_weight / total_w
                access_weight = access_weight / total_w

            all_results = []
            layers_queried = []

            # 1. Ephemeral
            if memory_type in ("ephemeral", "all"):
                ephemeral = await self.query_ephemeral(query_embedding, embedding_dim, limit=limit)
                all_results.extend(ephemeral)
                if ephemeral:
                    layers_queried.append("ephemeral")

            # 2. Episodic
            if memory_type in ("episodic", "all"):
                episodic = await self.query_episodic(
                    query_embedding,
                    embedding_dim=embedding_dim,
                    task_type=task_type,
                    hunt_context=hunt_context,
                    agent_id=agent_id,
                    pressure=pressure,
                    limit=limit,
                )
                all_results.extend(episodic)
                if episodic:
                    layers_queried.append("episodic")

            # 3. Semantic
            if memory_type in ("semantic", "all"):
                semantic = await self.query_semantic(query_embedding, embedding_dim, limit=limit)
                all_results.extend(semantic)
                if semantic:
                    layers_queried.append("semantic")

            # 4. Procedural
            if memory_type in ("procedural", "all"):
                procedural = await self.query_procedural(
                    query_embedding, embedding_dim, limit=limit
                )
                all_results.extend(procedural)
                if procedural:
                    layers_queried.append("procedural")

            # 5. Meta
            if memory_type in ("meta", "all"):
                meta = await self.query_meta(limit=limit)
                all_results.extend(meta)
                if meta:
                    layers_queried.append("meta")

            # Compute Bet 2 composite scores
            for result in all_results:
                result.compute_composite_score(
                    similarity_weight=similarity_weight,
                    significance_weight=significance_weight,
                    recency_weight=recency_weight,
                    access_weight=access_weight,
                    beta=beta,
                )

            # Sort descending by composite score
            all_results.sort(key=lambda r: r.composite_score, reverse=True)
            all_results = all_results[:limit]

            return {
                "success": True,
                "query_type": memory_type,
                "total_results": len(all_results),
                "layers_queried": layers_queried,
                "weights": {
                    "similarity": similarity_weight,
                    "significance": significance_weight,
                    "recency": recency_weight,
                    "access": access_weight,
                    "beta": beta,
                },
                "results": [r.to_dict() for r in all_results],
            }

        except Exception as e:
            logger.error(f"Ranked search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_type": memory_type,
                "total_results": 0,
                "results": [],
            }

    # ==================== HELPER METHODS ====================

    @staticmethod
    def _format_vector(embedding: List[float], dim: int = EMBEDDING_DIM) -> str:
        """Format embedding as PostgreSQL vector string"""
        if len(embedding) > dim:
            embedding = embedding[:dim]
        elif len(embedding) < dim:
            embedding = embedding + [0.0] * (dim - len(embedding))
        return "[" + ",".join(str(e) for e in embedding) + "]"


def add_phase3_tools_to_mcp(
    mcp_handler_tools: Dict[str, Any],
    phase3: NahRememberPhase3,
    embedding_fn: Optional[Any] = None,
):
    """Inject Phase 3 tools into MCP handler."""

    async def remember_search_ranked_wrapper(
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        memory_type: str = "all",
        limit: int = 10,
        embedding_dim: int = EMBEDDING_DIM,
        similarity_weight: float = 0.4,
        significance_weight: float = 0.3,
        recency_weight: float = 0.2,
        access_weight: float = 0.1,
        beta: float = DEFAULT_BETA,
        task_type: Optional[str] = None,
        hunt_context: Optional[str] = None,
        agent_id: Optional[str] = None,
        pressure: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not query_embedding and query and embedding_fn:
            query_embedding = await embedding_fn(query)
        if not query_embedding:
            query_embedding = [0.0] * embedding_dim

        return await phase3.remember_search_ranked(
            query_embedding,
            memory_type=memory_type,
            limit=limit,
            embedding_dim=embedding_dim,
            similarity_weight=similarity_weight,
            significance_weight=significance_weight,
            recency_weight=recency_weight,
            access_weight=access_weight,
            beta=beta,
            task_type=task_type,
            hunt_context=hunt_context,
            agent_id=agent_id,
            pressure=pressure,
        )

    async def episode_open_wrapper(
        title: str = "Agent Episode",
        goal: str = "",
        session_id: Optional[str] = None,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        hunt_context: Optional[str] = None,
        initial_pressure: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ep_id = await phase3.open_episode(
            session_id=session_id,
            task_type=task_type,
            agent_id=agent_id,
            hunt_context=hunt_context,
            initial_pressure=initial_pressure,
            title=title,
            goal=goal,
            metadata=metadata,
        )
        return {"success": True, "episode_id": ep_id}

    async def episode_append_event_wrapper(
        episode_id: str,
        event_type: str,
        description: str = "",
        details: Optional[Dict[str, Any]] = None,
        pressure: Optional[float] = None,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        evt_idx = await phase3.append_episode_event(
            episode_id=episode_id,
            event_type=event_type,
            description=description,
            details=details,
            pressure=pressure,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_result=tool_result,
        )
        return {"success": True, "episode_id": episode_id, "event_index": evt_idx}

    async def episode_close_wrapper(
        episode_id: str,
        outcome: str = "success",
        final_summary: Optional[str] = None,
        retrospective: Optional[str] = None,
        lessons_learned: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        closed = await phase3.close_episode(
            episode_id=episode_id,
            outcome=outcome,
            final_summary=final_summary,
            retrospective=retrospective,
            lessons_learned=lessons_learned,
            embedding=embedding,
        )
        return {"success": closed, "episode_id": episode_id, "outcome": outcome}

    mcp_handler_tools["nah_remember_search_ranked"] = remember_search_ranked_wrapper
    mcp_handler_tools["nah_remember_episode_open"] = episode_open_wrapper
    mcp_handler_tools["nah_remember_episode_append_event"] = episode_append_event_wrapper
    mcp_handler_tools["nah_remember_episode_close"] = episode_close_wrapper
    logger.info("nah-neural Phase 3 tools registered with MCP handler")
