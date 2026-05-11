#!/usr/bin/env python3
"""
NAH Memory Unified MCP Server
Consolidates mem0, rag-memory, and episodic-memory into a single PostgreSQL-backed system.

Storage: PostgreSQL nah_remember schema with pgvector embeddings
Embedding: LiteLLM proxy with OpenAI text-embedding-3-small/large aliases

Tables:
- sessions: Session tracking across applications
- messages: Conversation messages with embeddings
- context_snapshots: Working state checkpoints
- entities: Knowledge graph entities
- relationships: Entity relationships
- semantic_memories: Long-term semantic memories
- memory_history: Version tracking for semantic memories
- procedures: Learned workflows
- procedure_executions: Execution history
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional
import logging
from datetime import datetime
import uuid

import asyncpg
import httpx

# Import Phase 3
from nah_remember_phase3 import NahRememberPhase3, add_phase3_tools_to_mcp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Embedding configuration
# LiteLLM is the only embedding path so the server stays decoupled from local model runners.
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-primary")
EMBEDDING_MODEL_FALLBACK = os.getenv("EMBEDDING_MODEL_FALLBACK", "embedding-large")
LITELLM_API_KEY = os.getenv("LITELLM_MASTER_KEY", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


class NahMemoryMCPServer:
    """Unified Memory MCP Server with JSON-RPC Implementation"""

    # Default DSN when DATABASE_URL is empty, unset, or an unresolved template
    DEFAULT_DB_URL = "postgresql://human:@localhost/notahuman"

    @staticmethod
    def _resolve_db_url() -> str:
        """Get database URL, falling back to default if env var is empty or unresolved."""
        url = os.getenv("DATABASE_URL", "")
        # Handle empty, unset, or unresolved ${...} template strings
        if not url or url.startswith("${"):
            return NahMemoryMCPServer.DEFAULT_DB_URL
        return url

    def __init__(self):
        self.db_url = self._resolve_db_url()
        self.pool = None
        self.http_client = None
        logger.info(f"NAH Memory Unified MCP Server initializing (db: {self.db_url})...")

    async def setup(self):
        """Initialize connection pool and HTTP client"""
        try:
            self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=20)
            self.http_client = httpx.AsyncClient(timeout=30.0)
            logger.info(f"PostgreSQL pool created: {self.db_url}")
            logger.info(f"LiteLLM endpoint: {LITELLM_BASE_URL}")
            logger.info(
                f"Embedding model: {EMBEDDING_MODEL} (fallback: {EMBEDDING_MODEL_FALLBACK})"
            )
        except Exception as e:
            logger.error(f"Setup error: {e}")
            raise

    async def ensure_pool(self):
        """Lazy pool initialization — retry if pool is None or closed.

        When launched on-demand via mcp-gateway, the initial setup() may fail
        due to transient issues (PostgreSQL not ready, env vars not resolved).
        This method retries pool creation on each tool call if needed.
        """
        if self.pool is not None:
            return
        # Re-resolve db_url in case __init__ ran with stale env
        self.db_url = self._resolve_db_url()
        logger.warning(f"Pool is None — attempting lazy initialization with db_url={self.db_url!r}")
        try:
            self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=20)
            logger.info(f"Lazy pool initialization succeeded: {self.db_url}")
        except Exception as e:
            logger.error(f"Lazy pool initialization failed (db_url={self.db_url!r}): {e}")
            raise RuntimeError(f"Database connection unavailable (db_url={self.db_url!r}): {e}")
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)

    async def cleanup(self):
        """Cleanup resources"""
        if self.pool:
            await self.pool.close()
        if self.http_client:
            await self.http_client.aclose()

    # ==================== EMBEDDING FUNCTIONS ====================

    async def generate_embedding(
        self, text: str, use_fallback: bool = False
    ) -> Optional[List[float]]:
        """Generate an embedding through LiteLLM."""

        async def _generate_via_litellm(model: str) -> Optional[List[float]]:
            response = await self.http_client.post(
                f"{LITELLM_BASE_URL}/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {LITELLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": text},
            )
            response.raise_for_status()
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                embedding = data["data"][0].get("embedding", [])
                logger.debug(
                    f"Generated LiteLLM embedding with {len(embedding)} dimensions using {model}"
                )
                return embedding
            logger.warning(f"Unexpected LiteLLM embedding response format: {data}")
            return None

        litellm_model = EMBEDDING_MODEL_FALLBACK if use_fallback else EMBEDDING_MODEL

        try:
            embedding = await _generate_via_litellm(litellm_model)
            if embedding:
                return embedding
        except httpx.HTTPStatusError as e:
            if not use_fallback and e.response.status_code >= 500:
                logger.warning(f"Primary LiteLLM embedding failed, trying fallback: {e}")
                return await self.generate_embedding(text, use_fallback=True)
            logger.error(f"LiteLLM embedding generation failed: {e}")
            return None
        except Exception as e:
            if not use_fallback:
                logger.warning(f"Primary LiteLLM embedding failed, trying fallback: {e}")
                return await self.generate_embedding(text, use_fallback=True)
            logger.error(f"LiteLLM embedding generation failed: {e}")
            return None

        return None

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Resize an embedding to the configured PostgreSQL vector dimension."""
        if len(embedding) > EMBEDDING_DIM:
            return embedding[:EMBEDDING_DIM]
        if len(embedding) < EMBEDDING_DIM:
            return embedding + [0.0] * (EMBEDDING_DIM - len(embedding))
        return embedding

    def _format_vector(self, embedding: List[float]) -> str:
        """Format embedding as PostgreSQL vector string."""
        normalized = self._normalize_embedding(embedding)
        return f"[{','.join(str(x) for x in normalized)}]"

    # ==================== SESSION MANAGEMENT ====================

    async def session_start(
        self,
        session_id: str,
        application: str,
        user_id: str = "default_user",
        project_context: Optional[str] = None,
        working_directory: Optional[str] = None,
        git_branch: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Start or resume a session"""
        try:
            metadata_json = json.dumps(metadata) if metadata else "{}"

            query = """
                INSERT INTO nah_remember.sessions
                (session_id, application, user_id, project_context, working_directory, git_branch, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (session_id) DO UPDATE SET
                    status = 'active',
                    updated_at = NOW()
                RETURNING id, session_id, application, user_id, status, created_at, updated_at
            """

            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(
                    query,
                    session_id,
                    application,
                    user_id,
                    project_context,
                    working_directory,
                    git_branch,
                    metadata_json,
                )

            logger.info(f"Session started/resumed: {session_id}")
            return {
                "success": True,
                "session_id": result["session_id"],
                "application": result["application"],
                "status": result["status"],
                "created_at": result["created_at"].isoformat() if result["created_at"] else None,
            }
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            raise

    async def session_end(self, session_id: str, summary: Optional[str] = None) -> Dict:
        """End a session"""
        try:
            query = """
                UPDATE nah_remember.sessions
                SET status = 'ended', ended_at = NOW(), summary = COALESCE($2, summary), updated_at = NOW()
                WHERE session_id = $1
                RETURNING session_id, status, ended_at, total_messages
            """

            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(query, session_id, summary)

            if result:
                logger.info(f"Session ended: {session_id}")
                return {
                    "success": True,
                    "session_id": result["session_id"],
                    "status": result["status"],
                    "total_messages": result["total_messages"],
                }
            else:
                return {"success": False, "error": "Session not found"}
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            raise

    async def session_get_context(self, session_id: str, message_limit: int = 50) -> Dict:
        """Get full session context for restoration"""
        try:
            query = "SELECT * FROM nah_remember.get_session_context($1, $2)"

            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(query, session_id, message_limit)

            return {
                "success": True,
                "session_info": result["session_info"],
                "recent_messages": result["recent_messages"],
                "latest_snapshot": result["latest_snapshot"],
                "relevant_entities": result["relevant_entities"],
            }
        except Exception as e:
            logger.error(f"Error getting session context: {e}")
            raise

    # ==================== MESSAGE MANAGEMENT ====================

    async def message_save(
        self,
        session_id: str,
        role: str,
        content: str,
        message_index: Optional[int] = None,
        tool_calls: Optional[Dict] = None,
        tool_results: Optional[Dict] = None,
        importance_score: float = 0.5,
        metadata: Optional[Dict] = None,
        generate_embedding: bool = True,
    ) -> Dict:
        """Save a message to the session"""
        try:
            # Get next message index if not provided
            if message_index is None:
                async with self.pool.acquire() as conn:
                    result = await conn.fetchval(
                        "SELECT COALESCE(MAX(message_index), 0) + 1 FROM nah_remember.messages WHERE session_id = $1",
                        session_id,
                    )
                    message_index = result or 1

            # Generate embedding if requested
            embedding = None
            if generate_embedding and content:
                embedding = await self.generate_embedding(content)

            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            tool_results_json = json.dumps(tool_results) if tool_results else None
            metadata_json = json.dumps(metadata) if metadata else "{}"
            token_count = len(content.split()) * 1.3  # Rough estimate

            query = """
                INSERT INTO nah_remember.messages
                (session_id, message_index, role, content, tool_calls, tool_results,
                 token_count, importance_score, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9::jsonb, $10::vector)
                RETURNING message_id, message_index, role, created_at
            """

            embedding_str = self._format_vector(embedding) if embedding else None

            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(
                    query,
                    session_id,
                    message_index,
                    role,
                    content,
                    tool_calls_json,
                    tool_results_json,
                    int(token_count),
                    importance_score,
                    metadata_json,
                    embedding_str,
                )

                # Update session message count
                await conn.execute(
                    "UPDATE nah_remember.sessions SET total_messages = total_messages + 1, updated_at = NOW() WHERE session_id = $1",
                    session_id,
                )

            logger.info(f"Message saved: {result['message_id']} (index {message_index})")
            return {
                "success": True,
                "message_id": str(result["message_id"]),
                "message_index": result["message_index"],
                "role": result["role"],
                "has_embedding": embedding is not None,
            }
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise

    async def message_search(
        self, query_text: str, session_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """Search messages by semantic similarity"""
        try:
            embedding = await self.generate_embedding(query_text)
            if not embedding:
                return {"success": False, "error": "Failed to generate query embedding"}

            embedding_str = self._format_vector(embedding)

            if session_id:
                query = """
                    SELECT message_id, session_id, role, content,
                           1 - (embedding <=> $1::vector) as similarity,
                           created_at
                    FROM nah_remember.messages
                    WHERE session_id = $2 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """
                async with self.pool.acquire() as conn:
                    results = await conn.fetch(query, embedding_str, session_id, limit)
            else:
                query = """
                    SELECT message_id, session_id, role, content,
                           1 - (embedding <=> $1::vector) as similarity,
                           created_at
                    FROM nah_remember.messages
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """
                async with self.pool.acquire() as conn:
                    results = await conn.fetch(query, embedding_str, limit)

            return {
                "success": True,
                "results": [
                    {
                        "message_id": str(r["message_id"]),
                        "session_id": r["session_id"],
                        "role": r["role"],
                        "content": r["content"][:500] + "..."
                        if len(r["content"]) > 500
                        else r["content"],
                        "similarity": float(r["similarity"]),
                        "created_at": r["created_at"].isoformat(),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            raise

    # ==================== CONTEXT SNAPSHOTS ====================

    async def context_save(
        self,
        session_id: str,
        snapshot_type: str,
        context_data: Dict,
        context_summary: Optional[str] = None,
        key_entities: Optional[List[str]] = None,
        action_items: Optional[List[str]] = None,
        decisions_made: Optional[List[str]] = None,
    ) -> Dict:
        """Save a context snapshot"""
        try:
            context_json = json.dumps(context_data)
            working_dir = context_data.get("working_directory")
            active_files = context_data.get("active_files", [])
            git_state = json.dumps(context_data.get("git_state", {}))
            todo_state = json.dumps(context_data.get("todo_state", {}))
            task_master_state = json.dumps(context_data.get("task_master_state", {}))

            query = """
                INSERT INTO nah_remember.context_snapshots
                (session_id, snapshot_type, context_data, working_directory, active_files,
                 git_state, todo_state, task_master_state, context_summary,
                 key_entities, action_items, decisions_made)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9, $10, $11, $12)
                RETURNING id, snapshot_type, created_at
            """

            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(
                    query,
                    session_id,
                    snapshot_type,
                    context_json,
                    working_dir,
                    active_files,
                    git_state,
                    todo_state,
                    task_master_state,
                    context_summary,
                    key_entities,
                    action_items,
                    decisions_made,
                )

            logger.info(f"Context snapshot saved: {result['id']} ({snapshot_type})")
            return {
                "success": True,
                "snapshot_id": result["id"],
                "snapshot_type": result["snapshot_type"],
                "created_at": result["created_at"].isoformat(),
            }
        except Exception as e:
            logger.error(f"Error saving context: {e}")
            raise

    # ==================== ENTITY MANAGEMENT ====================

    async def entity_add(
        self,
        name: str,
        entity_type: Optional[str] = None,
        description: Optional[str] = None,
        source_session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        generate_embedding: bool = True,
    ) -> Dict:
        """Add or update an entity"""
        try:
            # Generate embedding from name + description
            embedding = None
            if generate_embedding:
                embed_text = f"{name}: {description}" if description else name
                embedding = await self.generate_embedding(embed_text)

            metadata_json = json.dumps(metadata) if metadata else "{}"
            embedding_str = self._format_vector(embedding) if embedding else None

            query = """
                INSERT INTO nah_remember.entities
                (name, entity_type, description, source_session_id, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector)
                ON CONFLICT (name, entity_type) DO UPDATE SET
                    description = COALESCE($3, nah_remember.entities.description),
                    metadata = $5::jsonb,
                    embedding = COALESCE($6::vector, nah_remember.entities.embedding),
                    updated_at = NOW(),
                    access_count = nah_remember.entities.access_count + 1,
                    last_accessed_at = NOW()
                RETURNING id, name, entity_type, created_at, updated_at
            """

            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(
                    query,
                    name,
                    entity_type,
                    description,
                    source_session_id,
                    metadata_json,
                    embedding_str,
                )

            logger.info(f"Entity added/updated: {name}")
            return {
                "success": True,
                "entity_id": str(result["id"]),
                "name": result["name"],
                "entity_type": result["entity_type"],
                "has_embedding": embedding is not None,
            }
        except Exception as e:
            logger.error(f"Error adding entity: {e}")
            raise

    async def entity_get(self, name: str) -> Dict:
        """Get entity by name"""
        try:
            query = """
                UPDATE nah_remember.entities
                SET access_count = access_count + 1, last_accessed_at = NOW()
                WHERE name = $1
                RETURNING id, name, entity_type, description, metadata,
                          created_at, updated_at, access_count, importance_score
            """

            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(query, name)

            if result:
                return {
                    "success": True,
                    "entity": {
                        "id": str(result["id"]),
                        "name": result["name"],
                        "entity_type": result["entity_type"],
                        "description": result["description"],
                        "metadata": result["metadata"],
                        "access_count": result["access_count"],
                        "importance_score": result["importance_score"],
                    },
                }
            else:
                return {"success": False, "error": "Entity not found"}
        except Exception as e:
            logger.error(f"Error getting entity: {e}")
            raise

    async def entity_search(
        self, query_text: str, entity_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """Search entities by semantic similarity"""
        try:
            embedding = await self.generate_embedding(query_text)
            if not embedding:
                # Fallback to text search
                return await self._entity_text_search(query_text, entity_type, limit)

            embedding_str = self._format_vector(embedding)

            if entity_type:
                query = """
                    SELECT id, name, entity_type, description,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM nah_remember.entities
                    WHERE entity_type = $2 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """
                async with self.pool.acquire() as conn:
                    results = await conn.fetch(query, embedding_str, entity_type, limit)
            else:
                query = """
                    SELECT id, name, entity_type, description,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM nah_remember.entities
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """
                async with self.pool.acquire() as conn:
                    results = await conn.fetch(query, embedding_str, limit)

            return {
                "success": True,
                "results": [
                    {
                        "id": str(r["id"]),
                        "name": r["name"],
                        "entity_type": r["entity_type"],
                        "description": r["description"],
                        "similarity": float(r["similarity"]),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            raise

    async def _entity_text_search(
        self, query_text: str, entity_type: Optional[str], limit: int
    ) -> Dict:
        """Fallback text search for entities"""
        if entity_type:
            query = """
                SELECT id, name, entity_type, description
                FROM nah_remember.entities
                WHERE entity_type = $2 AND (name ILIKE $1 OR description ILIKE $1)
                LIMIT $3
            """
            async with self.pool.acquire() as conn:
                results = await conn.fetch(query, f"%{query_text}%", entity_type, limit)
        else:
            query = """
                SELECT id, name, entity_type, description
                FROM nah_remember.entities
                WHERE name ILIKE $1 OR description ILIKE $1
                LIMIT $2
            """
            async with self.pool.acquire() as conn:
                results = await conn.fetch(query, f"%{query_text}%", limit)

        return {
            "success": True,
            "results": [
                {
                    "id": str(r["id"]),
                    "name": r["name"],
                    "entity_type": r["entity_type"],
                    "description": r["description"],
                    "similarity": None,  # Text search, no similarity score
                }
                for r in results
            ],
        }

    # ==================== RELATIONSHIP MANAGEMENT ====================

    async def relationship_add(
        self,
        source_name: str,
        target_name: str,
        relationship_type: str,
        description: Optional[str] = None,
        strength: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Add a relationship between entities"""
        try:
            metadata_json = json.dumps(metadata) if metadata else "{}"

            # Get or create source entity
            async with self.pool.acquire() as conn:
                source = await conn.fetchrow(
                    "SELECT id FROM nah_remember.entities WHERE name = $1", source_name
                )
                if not source:
                    # Create source entity
                    source = await conn.fetchrow(
                        "INSERT INTO nah_remember.entities (name) VALUES ($1) RETURNING id",
                        source_name,
                    )

                target = await conn.fetchrow(
                    "SELECT id FROM nah_remember.entities WHERE name = $1", target_name
                )
                if not target:
                    target = await conn.fetchrow(
                        "INSERT INTO nah_remember.entities (name) VALUES ($1) RETURNING id",
                        target_name,
                    )

                # Create relationship
                query = """
                    INSERT INTO nah_remember.relationships
                    (source_entity_id, target_entity_id, relationship_type, description, strength, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    ON CONFLICT (source_entity_id, target_entity_id, relationship_type) DO UPDATE SET
                        description = COALESCE($4, nah_remember.relationships.description),
                        strength = $5,
                        metadata = $6::jsonb,
                        updated_at = NOW()
                    RETURNING id, relationship_type, created_at
                """

                result = await conn.fetchrow(
                    query,
                    source["id"],
                    target["id"],
                    relationship_type,
                    description,
                    strength,
                    metadata_json,
                )

            logger.info(
                f"Relationship added: {source_name} --[{relationship_type}]--> {target_name}"
            )
            return {
                "success": True,
                "relationship_id": str(result["id"]),
                "source": source_name,
                "target": target_name,
                "relationship_type": result["relationship_type"],
            }
        except Exception as e:
            logger.error(f"Error adding relationship: {e}")
            raise

    async def relationship_get(self, entity_name: str) -> Dict:
        """Get all relationships for an entity"""
        try:
            query = """
                SELECT r.id, r.relationship_type, r.description, r.strength,
                       se.name as source_name, te.name as target_name,
                       CASE WHEN se.name = $1 THEN 'outgoing' ELSE 'incoming' END as direction
                FROM nah_remember.relationships r
                JOIN nah_remember.entities se ON r.source_entity_id = se.id
                JOIN nah_remember.entities te ON r.target_entity_id = te.id
                WHERE se.name = $1 OR te.name = $1
                ORDER BY r.created_at DESC
            """

            async with self.pool.acquire() as conn:
                results = await conn.fetch(query, entity_name)

            return {
                "success": True,
                "entity": entity_name,
                "relationships": [
                    {
                        "id": str(r["id"]),
                        "source": r["source_name"],
                        "target": r["target_name"],
                        "type": r["relationship_type"],
                        "description": r["description"],
                        "strength": r["strength"],
                        "direction": r["direction"],
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            raise

    # ==================== SEMANTIC MEMORY ====================

    async def memory_add(
        self,
        content: str,
        user_id: str = "default_user",
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        importance_score: float = 0.5,
    ) -> Dict:
        """Add a semantic memory (replaces mem0_add_memory)"""
        try:
            embedding = await self.generate_embedding(content)
            embedding_str = self._format_vector(embedding) if embedding else None
            metadata_json = json.dumps(metadata) if metadata else "{}"
            tags_arr = tags or []

            query = """
                INSERT INTO nah_remember.semantic_memories
                (user_id, agent_id, content, memory_type, category,
                 tags, metadata, importance_score, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9::vector)
                RETURNING id, user_id, created_at
            """

            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(
                    query,
                    user_id,
                    agent_id,
                    content,
                    memory_type,
                    category,
                    tags_arr,
                    metadata_json,
                    importance_score,
                    embedding_str,
                )

                # Record in history
                await conn.execute(
                    """
                    INSERT INTO nah_remember.memory_history
                    (memory_id, version, new_content, change_type, changed_by)
                    VALUES ($1, 1, $2, 'create', $3)
                """,
                    result["id"],
                    content,
                    agent_id or user_id,
                )

            logger.info(f"Semantic memory added: {result['id']}")
            return {
                "success": True,
                "memory_id": str(result["id"]),
                "user_id": result["user_id"],
                "has_embedding": embedding is not None,
            }
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise

    async def memory_search(
        self,
        query: str,
        user_id: str = "default_user",
        agent_id: Optional[str] = None,
        limit: int = 10,
    ) -> Dict:
        """Search semantic memories (replaces mem0_search_memory)"""
        try:
            embedding = await self.generate_embedding(query)
            if not embedding:
                return {"success": False, "error": "Failed to generate query embedding"}

            embedding_str = self._format_vector(embedding)

            if agent_id:
                sql = """
                    SELECT id, content, memory_type, category, tags, metadata,
                           1 - (embedding <=> $1::vector) as similarity,
                           created_at, access_count
                    FROM nah_remember.semantic_memories
                    WHERE user_id = $2 AND agent_id = $3 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $4
                """
                async with self.pool.acquire() as conn:
                    results = await conn.fetch(sql, embedding_str, user_id, agent_id, limit)
            else:
                sql = """
                    SELECT id, content, memory_type, category, tags, metadata,
                           1 - (embedding <=> $1::vector) as similarity,
                           created_at, access_count
                    FROM nah_remember.semantic_memories
                    WHERE user_id = $2 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """
                async with self.pool.acquire() as conn:
                    results = await conn.fetch(sql, embedding_str, user_id, limit)

            # Update access counts
            if results:
                async with self.pool.acquire() as conn:
                    for r in results:
                        await conn.execute(
                            "UPDATE nah_remember.semantic_memories SET access_count = access_count + 1, last_accessed_at = NOW() WHERE id = $1",
                            r["id"],
                        )

            return {
                "success": True,
                "results": [
                    {
                        "memory_id": str(r["id"]),
                        "content": r["content"],
                        "memory_type": r["memory_type"],
                        "category": r["category"],
                        "tags": r["tags"],
                        "similarity": float(r["similarity"]),
                        "created_at": r["created_at"].isoformat(),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise

    async def memory_get_all(
        self, user_id: str = "default_user", agent_id: Optional[str] = None
    ) -> Dict:
        """Get all memories for a user/agent"""
        try:
            if agent_id:
                query = """
                    SELECT id, content, memory_type, category, tags, metadata,
                           created_at, access_count, importance_score
                    FROM nah_remember.semantic_memories
                    WHERE user_id = $1 AND agent_id = $2
                    ORDER BY importance_score DESC, created_at DESC
                """
                async with self.pool.acquire() as conn:
                    results = await conn.fetch(query, user_id, agent_id)
            else:
                query = """
                    SELECT id, content, memory_type, category, tags, metadata,
                           created_at, access_count, importance_score
                    FROM nah_remember.semantic_memories
                    WHERE user_id = $1
                    ORDER BY importance_score DESC, created_at DESC
                """
                async with self.pool.acquire() as conn:
                    results = await conn.fetch(query, user_id)

            return {
                "success": True,
                "memories": [
                    {
                        "memory_id": str(r["id"]),
                        "content": r["content"],
                        "memory_type": r["memory_type"],
                        "category": r["category"],
                        "tags": r["tags"],
                        "access_count": r["access_count"],
                        "importance_score": r["importance_score"],
                        "created_at": r["created_at"].isoformat(),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error getting memories: {e}")
            raise

    async def memory_update(
        self,
        memory_id: str,
        content: str,
        user_id: str = "default_user",
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Update a semantic memory"""
        try:
            # Get current version
            async with self.pool.acquire() as conn:
                current = await conn.fetchrow(
                    "SELECT content FROM nah_remember.semantic_memories WHERE id = $1",
                    uuid.UUID(memory_id),
                )
                if not current:
                    return {"success": False, "error": "Memory not found"}

                # Get current version number
                version = await conn.fetchval(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM nah_remember.memory_history WHERE memory_id = $1",
                    uuid.UUID(memory_id),
                )

                # Generate new embedding
                embedding = await self.generate_embedding(content)
                embedding_str = self._format_vector(embedding) if embedding else None
                metadata_json = json.dumps(metadata) if metadata else None

                # Update memory
                if metadata_json:
                    await conn.execute(
                        """
                        UPDATE nah_remember.semantic_memories
                        SET content = $2, embedding = $3::vector, metadata = $4::jsonb, updated_at = NOW()
                        WHERE id = $1
                    """,
                        uuid.UUID(memory_id),
                        content,
                        embedding_str,
                        metadata_json,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE nah_remember.semantic_memories
                        SET content = $2, embedding = $3::vector, updated_at = NOW()
                        WHERE id = $1
                    """,
                        uuid.UUID(memory_id),
                        content,
                        embedding_str,
                    )

                # Record history
                await conn.execute(
                    """
                    INSERT INTO nah_remember.memory_history
                    (memory_id, version, previous_content, new_content, change_type, changed_by)
                    VALUES ($1, $2, $3, $4, 'update', $5)
                """,
                    uuid.UUID(memory_id),
                    version,
                    current["content"],
                    content,
                    user_id,
                )

            logger.info(f"Memory updated: {memory_id} (version {version})")
            return {"success": True, "memory_id": memory_id, "version": version}
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            raise

    async def memory_delete(self, memory_id: str, user_id: str = "default_user") -> Dict:
        """Delete a semantic memory"""
        try:
            async with self.pool.acquire() as conn:
                # Get content for history
                current = await conn.fetchrow(
                    "SELECT content FROM nah_remember.semantic_memories WHERE id = $1",
                    uuid.UUID(memory_id),
                )
                if not current:
                    return {"success": False, "error": "Memory not found"}

                version = await conn.fetchval(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM nah_remember.memory_history WHERE memory_id = $1",
                    uuid.UUID(memory_id),
                )

                # Record deletion in history before deleting
                await conn.execute(
                    """
                    INSERT INTO nah_remember.memory_history
                    (memory_id, version, previous_content, new_content, change_type, changed_by)
                    VALUES ($1, $2, $3, '', 'delete', $4)
                """,
                    uuid.UUID(memory_id),
                    version,
                    current["content"],
                    user_id,
                )

                # Delete memory (history entries cascade)
                await conn.execute(
                    "DELETE FROM nah_remember.semantic_memories WHERE id = $1", uuid.UUID(memory_id)
                )

            logger.info(f"Memory deleted: {memory_id}")
            return {"success": True, "memory_id": memory_id, "deleted": True}
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise

    async def memory_get_history(self, memory_id: str, user_id: str = "default_user") -> Dict:
        """Get memory version history"""
        try:
            query = """
                SELECT version, previous_content, new_content, change_type, changed_by, created_at
                FROM nah_remember.memory_history
                WHERE memory_id = $1
                ORDER BY version DESC
            """

            async with self.pool.acquire() as conn:
                results = await conn.fetch(query, uuid.UUID(memory_id))

            return {
                "success": True,
                "memory_id": memory_id,
                "history": [
                    {
                        "version": r["version"],
                        "previous_content": r["previous_content"],
                        "new_content": r["new_content"],
                        "change_type": r["change_type"],
                        "changed_by": r["changed_by"],
                        "created_at": r["created_at"].isoformat(),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error getting memory history: {e}")
            raise

    # ==================== UNIFIED SEARCH ====================

    async def unified_search(
        self, query: str, memory_types: Optional[List[str]] = None, limit_per_type: int = 5
    ) -> Dict:
        """Search across all memory types"""
        try:
            if memory_types is None:
                memory_types = ["messages", "entities", "semantic", "procedures"]

            embedding = await self.generate_embedding(query)
            if not embedding:
                return {"success": False, "error": "Failed to generate query embedding"}

            embedding_str = self._format_vector(embedding)

            results = []

            async with self.pool.acquire() as conn:
                if "messages" in memory_types:
                    results.extend(
                        await conn.fetch(
                            """
                        SELECT
                            'messages' as memory_type,
                            message_id::text as id,
                            content,
                            1 - (embedding <=> $1::vector) as similarity,
                            metadata,
                            created_at
                        FROM nah_remember.messages
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2
                    """,
                            embedding_str,
                            limit_per_type,
                        )
                    )

                if "entities" in memory_types:
                    results.extend(
                        await conn.fetch(
                            """
                        SELECT
                            'entities' as memory_type,
                            id::text as id,
                            COALESCE(description, name) as content,
                            1 - (embedding <=> $1::vector) as similarity,
                            metadata,
                            created_at
                        FROM nah_remember.entities
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2
                    """,
                            embedding_str,
                            limit_per_type,
                        )
                    )

                if "semantic" in memory_types:
                    results.extend(
                        await conn.fetch(
                            """
                        SELECT
                            COALESCE(memory_type, 'semantic') as memory_type,
                            id::text as id,
                            content,
                            1 - (embedding <=> $1::vector) as similarity,
                            metadata,
                            created_at
                        FROM nah_remember.semantic_memories
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2
                    """,
                            embedding_str,
                            limit_per_type,
                        )
                    )

                if "procedures" in memory_types:
                    results.extend(
                        await conn.fetch(
                            """
                        SELECT
                            'procedures' as memory_type,
                            id::text as id,
                            COALESCE(description, name) as content,
                            1 - (embedding <=> $1::vector) as similarity,
                            metadata,
                            created_at
                        FROM nah_remember.procedures
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2
                    """,
                            embedding_str,
                            limit_per_type,
                        )
                    )

            return {
                "success": True,
                "query": query,
                "results": [
                    {
                        "memory_type": r["memory_type"],
                        "id": r["id"],
                        "content": r["content"][:500] + "..."
                        if r["content"] and len(r["content"]) > 500
                        else r["content"],
                        "similarity": float(r["similarity"]) if r["similarity"] else None,
                        "metadata": r["metadata"],
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error in unified search: {e}")
            raise

    # ==================== COHERENCE CHECK ====================

    async def coherence_check(self, query: str, limit: int = 5) -> Dict:
        """
        Phase 3 coherence check: cross-query nah-remember and nah-rag for the same concept.

        Surfaces:
        - nah-remember results (semantic search)
        - GraphRAG entity matches (name + description)
        - Coverage: which entities appear in one system but not the other
        - Freshness gap: nah-remember updated_at vs GraphRAG index build time
        - Score delta: nah-remember composite score vs GraphRAG frequency/degree

        Args:
            query: Natural language query
            limit: Max results from each system (default 5)

        Returns:
            Coherence report dict with divergence analysis
        """
        import os
        import pandas as pd
        from datetime import timezone

        try:
            # 1. Generate embedding for query
            embedding = await self.generate_embedding(query)
            if not embedding:
                return {"success": False, "error": "Failed to generate embedding for query"}

            embedding_str = self._format_vector(embedding)

            # 2. Query nah-remember: semantic memories + entities
            remember_results = []
            async with self.pool.acquire() as conn:
                # Semantic memories
                sem_rows = await conn.fetch(
                    """
                    SELECT
                        'semantic_memory' as source,
                        id::text as id,
                        content,
                        importance_score,
                        access_count,
                        updated_at,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM nah_remember.semantic_memories
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """,
                    embedding_str,
                    limit,
                )

                # Entities
                ent_rows = await conn.fetch(
                    """
                    SELECT
                        'entity' as source,
                        id::text as id,
                        COALESCE(description, name) as content,
                        importance_score,
                        access_count,
                        updated_at,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM nah_remember.entities
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """,
                    embedding_str,
                    limit,
                )

                # Also get entity names for GraphRAG lookup
                entity_names = await conn.fetch(
                    """
                    SELECT name FROM nah_remember.entities
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """,
                    embedding_str,
                    limit * 2,
                )

            for r in sem_rows:
                remember_results.append(
                    {
                        "source": r["source"],
                        "id": r["id"],
                        "content": (r["content"] or "")[:300],
                        "importance_score": float(r["importance_score"])
                        if r["importance_score"]
                        else 0.5,
                        "access_count": int(r["access_count"]) if r["access_count"] else 0,
                        "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                        "similarity": float(r["similarity"]) if r["similarity"] else 0.0,
                    }
                )

            for r in ent_rows:
                remember_results.append(
                    {
                        "source": r["source"],
                        "id": r["id"],
                        "content": (r["content"] or "")[:300],
                        "importance_score": float(r["importance_score"])
                        if r["importance_score"]
                        else 0.5,
                        "access_count": int(r["access_count"]) if r["access_count"] else 0,
                        "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                        "similarity": float(r["similarity"]) if r["similarity"] else 0.0,
                    }
                )

            # Sort by similarity, take top limit
            remember_results.sort(key=lambda x: x["similarity"], reverse=True)
            remember_results = remember_results[:limit]

            # 3. Query GraphRAG entities parquet
            graphrag_results = []
            graphrag_entity_names = set()
            graphrag_index_timestamp = None
            graphrag_total_entities = 0

            graphrag_output_path = os.environ.get(
                "GRAPHRAG_OUTPUT_PATH", "/Users/human/notahuman/data-management/graphrag/output"
            )
            entities_parquet = os.path.join(graphrag_output_path, "entities.parquet")
            stats_json = os.path.join(graphrag_output_path, "stats.json")

            if os.path.exists(entities_parquet):
                try:
                    entities_df = pd.read_parquet(entities_parquet)
                    graphrag_total_entities = len(entities_df)

                    # Get index timestamp from stats.json
                    if os.path.exists(stats_json):
                        import json as _json

                        with open(stats_json) as f:
                            stats_data = _json.load(f)
                        # stats.json may have 'timestamp' or file mtime fallback
                        graphrag_index_timestamp = stats_data.get("timestamp") or stats_data.get(
                            "end_time"
                        )

                    if not graphrag_index_timestamp:
                        # Fall back to file mtime
                        mtime = os.path.getmtime(entities_parquet)
                        graphrag_index_timestamp = datetime.fromtimestamp(
                            mtime, tz=timezone.utc
                        ).isoformat()

                    # Build set of all GraphRAG entity titles (uppercased for matching)
                    all_graphrag_titles = set(entities_df["title"].str.upper().tolist())

                    # Text search: find entities matching the query keywords
                    query_words = query.lower().split()
                    mask = entities_df["title"].str.lower().apply(
                        lambda t: any(w in t for w in query_words)
                    ) | entities_df["description"].str.lower().apply(
                        lambda d: any(w in str(d).lower() for w in query_words)
                    )
                    matched = entities_df[mask].head(limit)

                    for _, row in matched.iterrows():
                        title = str(row.get("title", ""))
                        graphrag_entity_names.add(title.upper())
                        graphrag_results.append(
                            {
                                "source": "graphrag",
                                "id": str(row.get("id", "")),
                                "title": title,
                                "type": str(row.get("type", "")),
                                "content": (str(row.get("description", "")) or "")[:300],
                                "frequency": int(row.get("frequency", 0))
                                if pd.notna(row.get("frequency"))
                                else 0,
                                "degree": int(row.get("degree", 0))
                                if pd.notna(row.get("degree"))
                                else 0,
                            }
                        )

                except Exception as e:
                    logger.warning(f"GraphRAG parquet read failed: {e}")
                    graphrag_results = [{"error": str(e)}]
            else:
                graphrag_results = [{"error": f"GraphRAG output not found at {entities_parquet}"}]

            # 4. Coverage analysis
            # Which nah-remember entities appear in GraphRAG?
            remember_entity_names = {
                r["content"].upper()[:50] for r in remember_results if r["source"] == "entity"
            }
            top_entity_names = {str(r["name"]).upper() for r in entity_names}

            in_both = top_entity_names & graphrag_entity_names if graphrag_entity_names else set()
            only_in_remember = (
                top_entity_names - graphrag_entity_names
                if graphrag_entity_names
                else top_entity_names
            )
            only_in_graphrag = graphrag_entity_names - top_entity_names

            coverage_score = (
                len(in_both) / max(len(top_entity_names), 1) if top_entity_names else 0.0
            )

            # 5. Freshness gap
            freshness_gap = None
            if graphrag_index_timestamp and remember_results:
                most_recent_remember = max(
                    (r["updated_at"] for r in remember_results if r["updated_at"]), default=None
                )
                if most_recent_remember:
                    try:
                        graphrag_dt = datetime.fromisoformat(
                            str(graphrag_index_timestamp).replace("Z", "+00:00")
                        )
                        remember_dt = datetime.fromisoformat(most_recent_remember)
                        if remember_dt.tzinfo is None:
                            remember_dt = remember_dt.replace(tzinfo=timezone.utc)
                        if graphrag_dt.tzinfo is None:
                            graphrag_dt = graphrag_dt.replace(tzinfo=timezone.utc)
                        gap_days = (remember_dt - graphrag_dt).days
                        freshness_gap = {
                            "graphrag_index_time": graphrag_index_timestamp,
                            "most_recent_remember_update": most_recent_remember,
                            "gap_days": gap_days,
                            "assessment": (
                                "IN_SYNC"
                                if abs(gap_days) <= 1
                                else "REMEMBER_AHEAD"
                                if gap_days > 0
                                else "GRAPHRAG_AHEAD"
                            ),
                        }
                    except Exception as e:
                        freshness_gap = {"error": f"Could not compute gap: {e}"}

            # 6. Score delta (normalized comparison)
            # nah-remember uses composite score; GraphRAG uses frequency + degree
            score_comparison = {}
            if (
                remember_results
                and graphrag_results
                and not any("error" in r for r in graphrag_results)
            ):
                avg_remember_similarity = sum(r["similarity"] for r in remember_results) / len(
                    remember_results
                )
                avg_graphrag_freq = sum(r.get("frequency", 0) for r in graphrag_results) / len(
                    graphrag_results
                )
                max_possible_freq = max(r.get("frequency", 0) for r in graphrag_results) or 1
                normalized_graphrag = avg_graphrag_freq / max_possible_freq
                score_comparison = {
                    "avg_remember_similarity": round(avg_remember_similarity, 4),
                    "avg_graphrag_frequency": round(avg_graphrag_freq, 2),
                    "normalized_graphrag_score": round(normalized_graphrag, 4),
                    "score_delta": round(avg_remember_similarity - normalized_graphrag, 4),
                    "note": "Positive delta = nah-remember rates this concept higher than GraphRAG frequency",
                }

            return {
                "success": True,
                "query": query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "nah_remember": {"results": remember_results, "count": len(remember_results)},
                "graphrag": {
                    "results": graphrag_results,
                    "count": len(graphrag_results),
                    "total_entities_in_index": graphrag_total_entities,
                    "index_timestamp": graphrag_index_timestamp,
                },
                "coherence": {
                    "coverage_score": round(coverage_score, 4),
                    "entities_in_both": sorted(list(in_both)),
                    "only_in_nah_remember": sorted(list(only_in_remember)),
                    "only_in_graphrag": sorted(list(only_in_graphrag)),
                    "freshness_gap": freshness_gap,
                    "score_comparison": score_comparison,
                    "assessment": (
                        "COHERENT"
                        if coverage_score >= 0.5
                        else "PARTIAL"
                        if coverage_score >= 0.2
                        else "DIVERGENT"
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Coherence check failed: {e}")
            return {"success": False, "error": str(e), "query": query}

    # ==================== STATS ====================

    async def get_stats(self) -> Dict:
        """Get memory system statistics"""
        try:
            stats = {}
            async with self.pool.acquire() as conn:
                # Table counts
                for table in [
                    "sessions",
                    "messages",
                    "context_snapshots",
                    "entities",
                    "relationships",
                    "semantic_memories",
                    "procedures",
                ]:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM nah_remember.{table}")
                    stats[table] = count

                # Embedding coverage
                for table in ["messages", "entities", "semantic_memories"]:
                    total = await conn.fetchval(f"SELECT COUNT(*) FROM nah_remember.{table}")
                    with_embedding = await conn.fetchval(
                        f"SELECT COUNT(*) FROM nah_remember.{table} WHERE embedding IS NOT NULL"
                    )
                    stats[f"{table}_embedding_coverage"] = f"{with_embedding}/{total}"

            return {
                "success": True,
                "stats": stats,
                "litellm_endpoint": LITELLM_BASE_URL,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": EMBEDDING_DIM,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise


# ==================== MCP JSON-RPC HANDLER ====================


class MCPHandler:
    """JSON-RPC handler for MCP protocol"""

    def __init__(self, server: NahMemoryMCPServer, phase3: Optional[NahRememberPhase3] = None):
        self.server = server
        self.phase3 = phase3
        self.tools = {
            # Session management
            "nah_session_start": self.server.session_start,
            "nah_session_end": self.server.session_end,
            "nah_session_context": self.server.session_get_context,
            # Message management
            "nah_message_save": self.server.message_save,
            "nah_message_search": self.server.message_search,
            # Context snapshots
            "nah_context_save": self.server.context_save,
            # Entity management
            "nah_entity_add": self.server.entity_add,
            "nah_entity_get": self.server.entity_get,
            "nah_entity_search": self.server.entity_search,
            # Relationship management
            "nah_relationship_add": self.server.relationship_add,
            "nah_relationship_get": self.server.relationship_get,
            # Semantic memory (replaces mem0)
            "nah_remember_add": self.server.memory_add,
            "nah_remember_search": self.server.memory_search,
            "nah_remember_get_all": self.server.memory_get_all,
            "nah_remember_update": self.server.memory_update,
            "nah_remember_delete": self.server.memory_delete,
            "nah_remember_history": self.server.memory_get_history,
            # Unified search
            "nah_unified_search": self.server.unified_search,
            # Stats
            "nah_stats": self.server.get_stats,
            # Coherence check (Phase 3)
            "nah_coherence_check": self.server.coherence_check,
        }

        # Register Phase 3 tools if available
        if self.phase3:
            add_phase3_tools_to_mcp(self.tools, self.phase3)
            logger.info("Phase 3 tools registered with MCP handler")

    def get_tools_list(self) -> List[Dict]:
        """Return MCP tools list for initialization"""
        return [
            {
                "name": "nah_session_start",
                "description": "Start or resume a session for conversation tracking",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Unique session identifier",
                        },
                        "application": {
                            "type": "string",
                            "description": "Application name (claude-desktop, claude-code, etc.)",
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "default": "default_user",
                        },
                        "project_context": {
                            "type": "string",
                            "description": "Project context path",
                        },
                        "working_directory": {
                            "type": "string",
                            "description": "Current working directory",
                        },
                        "git_branch": {"type": "string", "description": "Current git branch"},
                        "metadata": {"type": "object", "description": "Additional metadata"},
                    },
                    "required": ["session_id", "application"],
                },
            },
            {
                "name": "nah_session_end",
                "description": "End a session with optional summary",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session to end"},
                        "summary": {"type": "string", "description": "Session summary"},
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "nah_session_context",
                "description": "Get full session context for restoration (messages, snapshots, entities)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"},
                        "message_limit": {
                            "type": "integer",
                            "description": "Max messages to return",
                            "default": 50,
                        },
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "nah_message_save",
                "description": "Save a message to the current session with optional embedding",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"},
                        "role": {
                            "type": "string",
                            "description": "Message role (user, assistant, system)",
                        },
                        "content": {"type": "string", "description": "Message content"},
                        "message_index": {"type": "integer", "description": "Message order index"},
                        "tool_calls": {"type": "object", "description": "Tool calls made"},
                        "tool_results": {"type": "object", "description": "Tool results"},
                        "importance_score": {
                            "type": "number",
                            "description": "Importance 0.0-1.0",
                            "default": 0.5,
                        },
                        "generate_embedding": {
                            "type": "boolean",
                            "description": "Generate embedding",
                            "default": True,
                        },
                    },
                    "required": ["session_id", "role", "content"],
                },
            },
            {
                "name": "nah_message_search",
                "description": "Search messages by semantic similarity",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string", "description": "Search query"},
                        "session_id": {"type": "string", "description": "Limit to session"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
                    },
                    "required": ["query_text"],
                },
            },
            {
                "name": "nah_context_save",
                "description": "Save a context snapshot (working state checkpoint)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"},
                        "snapshot_type": {
                            "type": "string",
                            "description": "Type: checkpoint, milestone, completion",
                        },
                        "context_data": {"type": "object", "description": "Full context data"},
                        "context_summary": {
                            "type": "string",
                            "description": "Human-readable summary",
                        },
                        "key_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key entities mentioned",
                        },
                        "action_items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Pending actions",
                        },
                        "decisions_made": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Decisions recorded",
                        },
                    },
                    "required": ["session_id", "snapshot_type", "context_data"],
                },
            },
            {
                "name": "nah_entity_add",
                "description": "Add or update a knowledge graph entity",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Entity name (unique with type)"},
                        "entity_type": {
                            "type": "string",
                            "description": "Entity type (person, project, concept, etc.)",
                        },
                        "description": {"type": "string", "description": "Entity description"},
                        "source_session_id": {
                            "type": "string",
                            "description": "Session where entity was discovered",
                        },
                        "metadata": {"type": "object", "description": "Additional metadata"},
                        "generate_embedding": {
                            "type": "boolean",
                            "description": "Generate embedding",
                            "default": True,
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "nah_entity_get",
                "description": "Get entity by name",
                "inputSchema": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Entity name"}},
                    "required": ["name"],
                },
            },
            {
                "name": "nah_entity_search",
                "description": "Search entities by semantic similarity",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string", "description": "Search query"},
                        "entity_type": {"type": "string", "description": "Filter by type"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
                    },
                    "required": ["query_text"],
                },
            },
            {
                "name": "nah_relationship_add",
                "description": "Add a relationship between two entities",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_name": {"type": "string", "description": "Source entity name"},
                        "target_name": {"type": "string", "description": "Target entity name"},
                        "relationship_type": {"type": "string", "description": "Relationship type"},
                        "description": {
                            "type": "string",
                            "description": "Relationship description",
                        },
                        "strength": {
                            "type": "number",
                            "description": "Relationship strength 0.0-1.0",
                            "default": 1.0,
                        },
                        "metadata": {"type": "object", "description": "Additional metadata"},
                    },
                    "required": ["source_name", "target_name", "relationship_type"],
                },
            },
            {
                "name": "nah_relationship_get",
                "description": "Get all relationships for an entity",
                "inputSchema": {
                    "type": "object",
                    "properties": {"entity_name": {"type": "string", "description": "Entity name"}},
                    "required": ["entity_name"],
                },
            },
            {
                "name": "nah_remember_add",
                "description": "Add a semantic memory (long-term knowledge)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "default": "default_user",
                        },
                        "agent_id": {"type": "string", "description": "Agent identifier"},
                        "memory_type": {"type": "string", "description": "Memory type category"},
                        "category": {"type": "string", "description": "Memory category"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization",
                        },
                        "metadata": {"type": "object", "description": "Additional metadata"},
                        "importance_score": {
                            "type": "number",
                            "description": "Importance 0.0-1.0",
                            "default": 0.5,
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "nah_remember_search",
                "description": "Search semantic memories by similarity",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "default": "default_user",
                        },
                        "agent_id": {"type": "string", "description": "Agent identifier"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "nah_remember_get_all",
                "description": "Get all memories for a user/agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "default": "default_user",
                        },
                        "agent_id": {"type": "string", "description": "Agent identifier"},
                    },
                },
            },
            {
                "name": "nah_remember_update",
                "description": "Update a semantic memory (creates version history)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory UUID"},
                        "content": {"type": "string", "description": "New content"},
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "default": "default_user",
                        },
                        "metadata": {"type": "object", "description": "Updated metadata"},
                    },
                    "required": ["memory_id", "content"],
                },
            },
            {
                "name": "nah_remember_delete",
                "description": "Delete a semantic memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory UUID"},
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "default": "default_user",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "nah_remember_history",
                "description": "Get version history for a memory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory UUID"},
                        "user_id": {
                            "type": "string",
                            "description": "User identifier",
                            "default": "default_user",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "nah_unified_search",
                "description": "Search across all memory types (messages, entities, semantic, procedures)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "memory_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["messages", "entities", "semantic", "procedures"],
                            },
                            "description": "Memory types to search",
                        },
                        "limit_per_type": {
                            "type": "integer",
                            "description": "Results per type",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "nah_stats",
                "description": "Get memory system statistics",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "nah_coherence_check",
                "description": (
                    "Phase 3 coherence check: cross-query nah-remember and nah-rag (GraphRAG) for the same concept. "
                    "Returns: nah-remember semantic results, GraphRAG entity matches, coverage score (overlap), "
                    "freshness gap (nah-remember vs GraphRAG index timestamp), and score delta. "
                    "Use to diagnose alignment between the two memory systems."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to check coherence for",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results per system (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    async def handle_request(self, request: Dict) -> Optional[Dict]:
        """Handle incoming JSON-RPC request or notification"""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        # Notifications don't have IDs and should not be responded to
        is_notification = request_id is None

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "nah-memory-unified", "version": "1.0.0"},
                    },
                }
            elif method == "notifications/initialized":
                # Notification - no response needed
                return None
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": self.get_tools_list()},
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if tool_name not in self.tools:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                    }

                try:
                    # Ensure database pool is available (lazy init if setup() failed)
                    await self.server.ensure_pool()
                    result = await self.tools[tool_name](**arguments)
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(result, default=str)}]
                        },
                    }
                except Exception as tool_error:
                    logger.error(f"Tool {tool_name} failed: {tool_error}")
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Tool {tool_name} failed: {str(tool_error)}",
                        },
                    }
            else:
                # Only return error if it's a request (has ID), not a notification
                if is_notification:
                    return None  # Ignore unknown notifications
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            safe_id = request_id if request_id is not None else "error"
            return {"jsonrpc": "2.0", "id": safe_id, "error": {"code": -32603, "message": str(e)}}


async def main():
    """Main entry point for MCP server"""
    server = NahMemoryMCPServer()
    phase3 = None
    try:
        await server.setup()
        # Initialize Phase 3 with the same pool
        phase3 = NahRememberPhase3(server.pool)
        logger.info("Phase 3 initialized successfully")
    except Exception as e:
        logger.error(f"Server setup failed: {e}")
        # Continue anyway - server can handle some operations without full setup
    handler = MCPHandler(server, phase3)

    logger.info("NAH Memory Unified MCP Server started (stdio mode)")

    # Read from stdin, write to stdout (synchronous I/O like rag-memory)
    request_id = 0  # Track request ID for error responses

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            try:
                request = json.loads(line.strip())
                request_id = request.get("id", request_id)  # Preserve last known ID
                response = await handler.handle_request(request)

                # Only send response if it's not None (notifications return None)
                if response is not None:
                    print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                # Return MCP-compliant error response
                print(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id if request_id else 0,
                            "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                        }
                    ),
                    flush=True,
                )
                continue
            except Exception as e:
                logger.error(f"Request handling error: {e}")
                # Return MCP-compliant error response
                print(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id if request_id else 0,
                            "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                        }
                    ),
                    flush=True,
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Server error: {e}")
            break

    # Cleanup on exit
    await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
