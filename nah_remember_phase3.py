#!/usr/bin/env python3
"""
NAH Remember Phase 3: Retrieval + Query Routing with Ranking
Implements cross-layer query interface (ephemeral → episodic → semantic → meta)
with composite scoring and type filtering.

Architecture:
- Ephemeral layer: Recent messages (< 24h, high recency weight)
- Episodic layer: Context snapshots + entity relationships (structured context)
- Semantic layer: Semantic memories + procedures (long-term knowledge)
- Meta layer: Entity graph + memory history (metadata + evolution tracking)

Ranking formula: significance_score × 0.3 + access_count × 0.2 + recency × 0.2 + reinforcement_score × 0.3
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Literal
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)

# Single source of truth for embedding dimension. Must match server.py EMBEDDING_DIM and the
# live pgvector columns: mistral-embed natively returns 1024-dim. Overridable via env for
# alternate embedders. Defaulting here prevents the 1536/1024 drift that silently breaks
# phase3 distance queries (`embedding <=> $1::vector`) against the 1024-dim columns.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


class MemoryLayer(str, Enum):
    """Memory layers for query routing"""

    EPHEMERAL = "ephemeral"  # Recent messages (< 24h)
    EPISODIC = "episodic"  # Context snapshots + entities
    SEMANTIC = "semantic"  # Semantic memories + procedures
    META = "meta"  # Entity relationships + history
    ALL = "all"  # All layers


class RankedResult:
    """Ranked result with composite scoring"""

    def __init__(
        self,
        memory_type: str,
        memory_id: str,
        content: str,
        layer: MemoryLayer,
        significance_score: float = 0.5,
        access_count: int = 0,
        reinforcement_score: float = 0.0,
        created_at: datetime = None,
        recency_window_days: int = 30,
        metadata: Optional[Dict] = None,
        similarity: Optional[float] = None,
    ):
        self.memory_type = memory_type
        self.memory_id = memory_id
        self.content = content
        self.layer = layer
        self.significance_score = significance_score
        self.reinforcement_score = reinforcement_score
        self.access_count = access_count
        self.created_at = created_at or datetime.utcnow()
        self.metadata = metadata or {}
        self.similarity = similarity or 0.0

        # Calculate recency score (0.0 - 1.0, normalized to recency_window_days)
        # Handle both timezone-aware and naive datetimes
        now = datetime.now(timezone.utc)
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).days
        self.recency_score = max(0.0, 1.0 - (age_days / recency_window_days))

        # Normalize access count logarithmically (avoid overwhelming impact)
        self.access_count_score = min(1.0, (1.0 + access_count) / 10.0)
        self.reinforcement_score = min(1.0, max(0.0, float(reinforcement_score)))

    def compute_composite_score(
        self,
        significance_weight: float = 0.3,
        access_weight: float = 0.2,
        recency_weight: float = 0.2,
        reinforcement_weight: float = 0.3,
    ) -> float:
        """
        Compute composite ranking score.

        Formula: significance_score × 0.3 + access_count × 0.2 + recency × 0.2 + reinforcement_score × 0.3
        """
        # Convert Decimal types to float (PostgreSQL returns Decimal for numeric fields)
        sig_score = (
            float(self.significance_score)
            if hasattr(self.significance_score, "__float__")
            else self.significance_score
        )
        access_score = (
            float(self.access_count_score)
            if hasattr(self.access_count_score, "__float__")
            else self.access_count_score
        )
        recency_score = (
            float(self.recency_score)
            if hasattr(self.recency_score, "__float__")
            else self.recency_score
        )
        reinf_score = (
            float(self.reinforcement_score)
            if hasattr(self.reinforcement_score, "__float__")
            else self.reinforcement_score
        )

        if (
            abs(significance_weight + access_weight + recency_weight + reinforcement_weight - 1.0)
            > 0.01
        ):
            logger.warning(
                f"Weights do not sum to 1.0: {significance_weight + access_weight + recency_weight}"
            )

        return (
            sig_score * significance_weight
            + access_score * access_weight
            + recency_score * recency_weight
            + reinf_score * reinforcement_weight
        )

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            "memory_type": self.memory_type,
            "memory_id": self.memory_id,
            "content": self.content[:1000] + "..." if len(self.content) > 1000 else self.content,
            "layer": self.layer.value,
            "composite_score": float(self.compute_composite_score()),
            "significance_score": float(self.significance_score),
            "access_count": int(self.access_count),
            "recency_score": float(self.recency_score),
            "similarity": float(self.similarity),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class NahRememberPhase3:
    """Phase 3: Ranked retrieval + query routing across memory layers"""

    def __init__(self, pool: asyncpg.pool.Pool):
        """
        Initialize Phase 3 with database pool.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool
        logger.info("NAH Remember Phase 3 initialized")

    # ==================== LAYER QUERIES ====================

    async def query_ephemeral(
        self,
        query_embedding: List[float],
        embedding_dim: int = EMBEDDING_DIM,
        hours: int = 24,
        limit: int = 10,
    ) -> List[RankedResult]:
        """
        Query ephemeral layer: recent messages (< hours old)

        Ephemeral = recent conversation context, high recency weight
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            embedding_str = self._format_vector(query_embedding, embedding_dim)

            query = """
                SELECT
                    'message' as memory_type,
                    m.message_id::text as memory_id,
                    m.content,
                    m.importance_score,
                    0 as access_count,
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

            results = [
                RankedResult(
                    memory_type=r["memory_type"],
                    memory_id=r["memory_id"],
                    content=r["content"],
                    layer=MemoryLayer.EPHEMERAL,
                    significance_score=r["importance_score"],
                    access_count=r["access_count"],
                    created_at=r["created_at"],
                    similarity=r["similarity"],
                )
                for r in rows
            ]

            logger.info(f"Ephemeral query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Ephemeral query failed: {e}")
            return []

    async def query_episodic(
        self, query_embedding: List[float], embedding_dim: int = EMBEDDING_DIM, limit: int = 10
    ) -> List[RankedResult]:
        """
        Query episodic layer: context snapshots + entities

        Episodic = structured context, relationships, project state
        """
        try:
            embedding_str = self._format_vector(query_embedding, embedding_dim)
            results = []

            async with self.pool.acquire() as conn:
                # Query context snapshots
                snapshot_query = """
                    SELECT
                        'context_snapshot' as memory_type,
                        cs.id::text as memory_id,
                        cs.context_summary as content,
                        0.6 as importance_score,
                        0 as access_count,
                        cs.created_at
                    FROM nah_remember.context_snapshots cs
                    WHERE cs.context_summary IS NOT NULL
                    ORDER BY cs.created_at DESC
                    LIMIT $1
                """
                snapshot_rows = await conn.fetch(snapshot_query, limit)

                # Query entities
                entity_query = """
                    SELECT
                        'entity' as memory_type,
                        e.id::text as memory_id,
                        e.description as content,
                        e.importance_score,
                        e.access_count,
                        e.created_at,
                        1 - (e.embedding <=> $1::vector) as similarity
                    FROM nah_remember.entities e
                    WHERE e.embedding IS NOT NULL
                    ORDER BY e.embedding <=> $1::vector
                    LIMIT $2
                """
                entity_rows = await conn.fetch(entity_query, embedding_str, limit)

            # Process snapshots
            for r in snapshot_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.EPISODIC,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        created_at=r["created_at"],
                    )
                )

            # Process entities
            for r in entity_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.EPISODIC,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        created_at=r["created_at"],
                        similarity=r.get("similarity", 0.0),
                    )
                )

            logger.info(f"Episodic query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Episodic query failed: {e}")
            return []

    async def query_semantic(
        self, query_embedding: List[float], embedding_dim: int = EMBEDDING_DIM, limit: int = 10
    ) -> List[RankedResult]:
        """
        Query semantic layer: semantic memories + procedures

        Semantic = long-term knowledge, learned patterns, procedures
        """
        try:
            embedding_str = self._format_vector(query_embedding, embedding_dim)
            results = []

            async with self.pool.acquire() as conn:
                # Query semantic memories
                memory_query = """
                    SELECT
                        'semantic_memory' as memory_type,
                        sm.id::text as memory_id,
                        sm.content,
                        sm.importance_score,
                        sm.access_count,
                        sm.created_at,
                        1 - (sm.embedding <=> $1::vector) as similarity
                    FROM nah_remember.semantic_memories sm
                    WHERE sm.embedding IS NOT NULL
                    ORDER BY sm.embedding <=> $1::vector
                    LIMIT $2
                """
                memory_rows = await conn.fetch(memory_query, embedding_str, limit)

                # Query procedures
                procedure_query = """
                    SELECT
                        'procedure' as memory_type,
                        p.id::text as memory_id,
                        p.description as content,
                        0.7 as importance_score,
                        p.execution_count as access_count,
                        p.created_at,
                        1 - (p.embedding <=> $1::vector) as similarity
                    FROM nah_remember.procedures p
                    WHERE p.embedding IS NOT NULL
                    ORDER BY p.embedding <=> $1::vector
                    LIMIT $2
                """
                procedure_rows = await conn.fetch(procedure_query, embedding_str, limit)

            # Process semantic memories
            for r in memory_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"],
                        layer=MemoryLayer.SEMANTIC,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        created_at=r["created_at"],
                        similarity=r["similarity"],
                    )
                )

            # Process procedures
            for r in procedure_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.SEMANTIC,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        created_at=r["created_at"],
                        similarity=r["similarity"],
                    )
                )

            logger.info(f"Semantic query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Semantic query failed: {e}")
            return []

    async def query_meta(self, limit: int = 10) -> List[RankedResult]:
        """
        Query meta layer: entity relationships + memory history

        Meta = entity graph topology, evolution tracking, lineage
        """
        try:
            results = []

            async with self.pool.acquire() as conn:
                # Query relationships (high-level structure)
                rel_query = """
                    SELECT
                        'relationship' as memory_type,
                        r.id::text as memory_id,
                        r.description as content,
                        CASE WHEN r.strength > 0.7 THEN 0.8 ELSE 0.5 END as importance_score,
                        0 as access_count,
                        r.created_at
                    FROM nah_remember.relationships r
                    ORDER BY r.strength DESC
                    LIMIT $1
                """
                rel_rows = await conn.fetch(rel_query, limit)

                # Query memory history (evolution)
                history_query = """
                    SELECT
                        'memory_evolution' as memory_type,
                        mh.id::text as memory_id,
                        mh.new_content as content,
                        0.6 as importance_score,
                        0 as access_count,
                        mh.created_at
                    FROM nah_remember.memory_history mh
                    WHERE mh.change_type = 'update'
                    ORDER BY mh.created_at DESC
                    LIMIT $1
                """
                history_rows = await conn.fetch(history_query, limit)

            # Process relationships
            for r in rel_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.META,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        created_at=r["created_at"],
                    )
                )

            # Process history
            for r in history_rows:
                results.append(
                    RankedResult(
                        memory_type=r["memory_type"],
                        memory_id=r["memory_id"],
                        content=r["content"] or "",
                        layer=MemoryLayer.META,
                        significance_score=r["importance_score"],
                        access_count=r["access_count"],
                        created_at=r["created_at"],
                    )
                )

            logger.info(f"Meta query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Meta query failed: {e}")
            return []

    # ==================== UNIFIED RANKED SEARCH ====================

    async def remember_search_ranked(
        self,
        query_embedding: List[float],
        memory_type: Literal["ephemeral", "episodic", "semantic", "meta", "all"] = "all",
        limit: int = 10,
        embedding_dim: int = EMBEDDING_DIM,
        significance_weight: float = 0.4,
        access_weight: float = 0.3,
        recency_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Unified ranked search across memory layers.

        Args:
            query_embedding: Query vector (pre-embedded)
            memory_type: Filter by layer (ephemeral|episodic|semantic|meta|all)
            limit: Results per layer
            embedding_dim: Embedding dimension (default EMBEDDING_DIM env = 1024, mistral-embed)
            significance_weight: Weight for significance score (0.0-1.0)
            access_weight: Weight for access count (0.0-1.0)
            recency_weight: Weight for recency (0.0-1.0)

        Returns:
            {
                "success": bool,
                "query_type": str,
                "total_results": int,
                "results": [RankedResult.to_dict(), ...],
                "layers_queried": [str, ...]
            }
        """
        try:
            # Validate weights sum to ~1.0
            total_weight = significance_weight + access_weight + recency_weight
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Weights sum to {total_weight}, normalizing...")
                significance_weight /= total_weight
                access_weight /= total_weight
                recency_weight /= total_weight

            all_results = []
            layers_queried = []

            # Route by memory type
            if memory_type == "ephemeral" or memory_type == "all":
                ephemeral = await self.query_ephemeral(query_embedding, embedding_dim, limit=limit)
                all_results.extend(ephemeral)
                if ephemeral:
                    layers_queried.append("ephemeral")

            if memory_type == "episodic" or memory_type == "all":
                episodic = await self.query_episodic(query_embedding, embedding_dim, limit=limit)
                all_results.extend(episodic)
                if episodic:
                    layers_queried.append("episodic")

            if memory_type == "semantic" or memory_type == "all":
                semantic = await self.query_semantic(query_embedding, embedding_dim, limit=limit)
                all_results.extend(semantic)
                if semantic:
                    layers_queried.append("semantic")

            if memory_type == "meta" or memory_type == "all":
                meta = await self.query_meta(limit=limit)
                all_results.extend(meta)
                if meta:
                    layers_queried.append("meta")

            # Compute composite scores
            for result in all_results:
                result.composite_score = result.compute_composite_score(
                    significance_weight, access_weight, recency_weight
                )

            # Sort by composite score
            all_results.sort(key=lambda r: r.composite_score, reverse=True)

            # Limit final results
            all_results = all_results[:limit]

            logger.info(
                f"Ranked search complete: {len(all_results)} results from layers {layers_queried}"
            )

            return {
                "success": True,
                "query_type": memory_type,
                "total_results": len(all_results),
                "layers_queried": layers_queried,
                "weights": {
                    "significance": significance_weight,
                    "access_count": access_weight,
                    "recency": recency_weight,
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
        # Truncate or pad to correct dimension
        if len(embedding) > dim:
            embedding = embedding[:dim]
        elif len(embedding) < dim:
            embedding = embedding + [0.0] * (dim - len(embedding))

        return "[" + ",".join(str(e) for e in embedding) + "]"


# ==================== INTEGRATION WITH MCP SERVER ====================


def add_phase3_tools_to_mcp(mcp_handler_tools: Dict[str, Any], phase3: NahRememberPhase3):
    """
    Inject Phase 3 tools into MCP handler.

    This function is called by the main MCP server to register Phase 3 functionality.
    """

    async def remember_search_ranked_wrapper(
        query_embedding: List[float],
        memory_type: str = "all",
        limit: int = 10,
        embedding_dim: int = EMBEDDING_DIM,
        significance_weight: float = 0.4,
        access_weight: float = 0.3,
        recency_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """MCP wrapper for ranked search"""
        return await phase3.remember_search_ranked(
            query_embedding,
            memory_type=memory_type,
            limit=limit,
            embedding_dim=embedding_dim,
            significance_weight=significance_weight,
            access_weight=access_weight,
            recency_weight=recency_weight,
        )

    mcp_handler_tools["nah_remember_search_ranked"] = remember_search_ranked_wrapper
    logger.info("Phase 3 tools registered with MCP handler")


if __name__ == "__main__":
    print("NAH Remember Phase 3 module loaded")
    print("Use add_phase3_tools_to_mcp() to register with MCP handler")
