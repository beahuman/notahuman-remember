# nah-remember

**Unified Long-Term Memory, Knowledge Graph & Semantic Ranking MCP Server for AI Agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.5+-orange.svg)](https://github.com/pgvector/pgvector)

`nah-remember` is an open Model Context Protocol (MCP) server that provides AI agents with persistent, unified memory across conversations and sessions. It combines conversational message histories, episodic context snapshots, semantic knowledge graphs, and composite multi-factor ranked recall into a single PostgreSQL + pgvector substrate.

---

## Architecture Overview

```
                                      AI Agent Client
                      (Hermes, Claude, Codex, Cursor, Custom Agent)
                                             │
                                    MCP Protocol (stdio)
                                             ▼
                             ┌───────────────────────────────┐
                             │       nah-remember MCP        │
                             │   (Tools: 14 tools registered)│
                             └───────────────┬───────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
             PostgreSQL + pgvector                           Embedding API
           (nah_remember Schema)                      (Any OpenAI-compatible API)
         ├─ sessions                                  ├─ Ollama
         ├─ messages (vector)                         ├─ vLLM / LiteLLM
         ├─ context_snapshots                         ├─ OpenAI / Mistral
         ├─ entities (vector)                         └─ Cloudflare AI / Bedrock
         ├─ relationships
         ├─ semantic_memories (vector)
         ├─ memory_history (audit versions)
         ├─ procedures
         └─ procedure_executions
```

### Memory Layers

1. **Session & Message Layer**: Full conversational turn logging, token count tracking, importance scoring, and semantic message search.
2. **Episodic Context Snapshots**: High-level working state checkpoints capture agent goals, active hypotheses, and milestone progress.
3. **Semantic Knowledge Graph**: Bi-directional entities and relationships with graph path traversal and semantic similarity lookups.
4. **Procedures & Execution History**: Reusable multi-step workflows with recorded execution telemetry and outcomes.
5. **Phase 3 Unified Ranked Recall**: Composite scoring combining significance, frequency of access, recency decay, and user reinforcement into a unified ranking formula:
   $$\text{Score} = w_{\text{sig}} \cdot \text{Significance} + w_{\text{acc}} \cdot \text{Access} + w_{\text{rec}} \cdot \text{Recency} + w_{\text{reinf}} \cdot \text{Reinforcement}$$

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- PostgreSQL 15+ with the `pgvector` and `uuid-ossp` extensions enabled.
- An OpenAI-compatible embedding endpoint (e.g., local Ollama, vLLM, or hosted API).

### 2. Database Setup

Create the database and apply the schema:

```bash
# Connect to your PostgreSQL instance and run setup_schema.sql
createdb memory_db
psql -d memory_db -f setup_schema.sql
```

### 3. Installation

Install via pip or clone directly:

```bash
git clone https://github.com/beahuman/notahuman-remember.git
cd notahuman-remember

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or pip install -e .
```

### 4. Configuration

`nah-remember` is configured entirely via environment variables.

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `DATABASE_URL` or `NAH_REMEMBER_DATABASE_URL` | PostgreSQL connection string | `postgresql://localhost:5432/postgres` |
| `NAH_REMEMBER_EMBEDDING_URL` or `EMBEDDING_URL` | OpenAI-compatible embedding endpoint | `http://127.0.0.1:11434/v1/embeddings` |
| `NAH_REMEMBER_EMBEDDING_MODEL` or `EMBEDDING_MODEL` | Primary embedding model name | `bge-m3` |
| `NAH_REMEMBER_EMBEDDING_FALLBACK_MODEL` | Optional fallback embedding model | `""` (disabled) |
| `NAH_REMEMBER_EMBEDDING_API_KEY` or `OPENAI_API_KEY`| Bearer token for embedding API | `""` (no auth for local) |
| `NAH_REMEMBER_EMBEDDING_DIM` or `EMBEDDING_DIM` | Embedding vector dimension | `1024` |
| `NAH_REMEMBER_POOL_ACQUIRE_TIMEOUT` | Connection pool acquire timeout (sec) | `10.0` |
| `NAH_REMEMBER_COMMAND_TIMEOUT` | Database query execution timeout (sec) | `30.0` |
| `NAH_REMEMBER_GRAPHRAG_OUTPUT_PATH` | Optional directory containing GraphRAG parquets | `""` (disabled) |

---

## Agent Client Integration

`nah-remember` communicates over standard stdio JSON-RPC following the Model Context Protocol (MCP) specification. Any MCP client can integrate seamlessly.

### Hermes Agent

Add to your Hermes profile MCP configuration (`~/.hermes/config.yaml` or profile `config.yaml`):

```yaml
mcp_servers:
  nah-remember:
    command: "/path/to/notahuman-remember/venv/bin/python"
    args: ["/path/to/notahuman-remember/server.py"]
    env:
      DATABASE_URL: "postgresql://user:password@localhost:5432/memory_db"
      EMBEDDING_URL: "http://127.0.0.1:11434/v1/embeddings"
      EMBEDDING_MODEL: "bge-m3"
```

### Claude Desktop / Claude Code

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nah-remember": {
      "command": "/path/to/notahuman-remember/venv/bin/python",
      "args": ["/path/to/notahuman-remember/server.py"],
      "env": {
        "DATABASE_URL": "postgresql://user:password@localhost:5432/memory_db",
        "EMBEDDING_URL": "http://127.0.0.1:11434/v1/embeddings",
        "EMBEDDING_MODEL": "bge-m3"
      }
    }
  }
}
```

### OpenAI Codex / Cursor / Windsurf

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "nah-remember": {
      "command": "python3",
      "args": ["/path/to/notahuman-remember/server.py"],
      "env": {
        "DATABASE_URL": "postgresql://user:password@localhost:5432/memory_db",
        "EMBEDDING_URL": "https://api.openai.com/v1/embeddings",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIM": "1536",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

---

## Available MCP Tools

The server exposes 14 high-signal tools:

### Session & Epistemic Tracking
- `nah_session_start`: Register or resume a session with application metadata, project context, and git branch.
- `nah_session_end`: Terminate a session and persist summary findings.
- `nah_context_checkpoint`: Snapshot working state, hypotheses, and progress for cross-session resumption.
- `nah_context_restore`: Retrieve previous checkpoints by session or project context.

### Message & Conversational Memory
- `nah_message_save`: Store conversation turns with token counts, importance weighting, and semantic embeddings.
- `nah_message_search`: Semantic similarity search across historical conversational turns.

### Semantic Memory & Knowledge Graph
- `nah_memory_add`: Add a semantic memory with automatic embedding generation and version tracking.
- `nah_memory_search`: Vector similarity search across stored semantic memories with threshold filtering.
- `nah_memory_update`: Safely update existing memories with version auditing.
- `nah_entity_add`: Upsert knowledge graph entities with semantic embeddings.
- `nah_entity_search`: Vector and text search over knowledge graph entities.
- `nah_relationship_add`: Record directed graph relationships with confidence scores.
- `nah_graph_traverse`: Traverse graph neighborhoods and subgraphs.

### Unified Ranked Recall (Phase 3)
- `nah_remember_search_ranked`: Cross-layer unified recall returning composite multi-factor ranked results across semantic memories, entities, messages, and snapshots.

---

## Verification & Testing

Run the test suite to verify database schema, imports, and Phase 3 ranking logic:

```bash
# Verify imports
python3 test_imports.py

# Run Phase 3 composite scoring and ranking verification
python3 test_phase3.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
