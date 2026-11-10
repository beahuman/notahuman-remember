# nah-remember: Long-Term Memory MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.5.0+-orange.svg)](https://github.com/pgvector/pgvector)
[![MCP](https://img.shields.io/badge/MCP-Protocol-green.svg)](https://modelcontextprotocol.io/)

**nah-remember** is an enterprise-grade, multi-agent long-term memory system exposed via the Model Context Protocol (MCP). It unifies episodic memory, semantic knowledge graphs, procedural workflows, and autonomous memory consolidation over a PostgreSQL + `pgvector` substrate.

---

## 🧠 Cognitive Memory Architecture

nah-remember implements a decoupled, 5-layer cognitive memory model:

1. **Working & Ephemeral Memory (Layer 1):** Fast session state tracking, context snapshots, and compact rolling message windows.
2. **Episodic Memory (Layer 2):** First-class episodic engrams (`nah_remember_episode_open`, `nah_remember_episode_append_event`, `nah_remember_episode_close`) capturing sequential agent experience with telemetry and outcome scoring.
3. **Semantic Memory (Layer 3):** Dual-indexed vector (1024-dim BGE-M3) + relational knowledge graph storing entities, relationships, facts, and observations.
4. **Procedural Memory (Layer 4):** Workflow crystallization (`nah_remember_procedure_save`, `nah_remember_procedure_execute`) tracking executable playbooks with dynamic success rate evolution.
5. **System 1 Distillation & Offline Consolidation (Layer 5):** Background Dreamer daemon running offline Wald SPRT sequential analysis, double-entry conservation trial balances, and rotated wake gates to promote episodes into semantic rules and procedures without drift.

---

## 📦 Installation

nah-remember is a standard Python package installable with `pip` or `uv`:

```bash
# Clone the repository
git clone https://github.com/beahuman/notahuman-remember.git
cd notahuman-remember

# Install in editable mode or standard mode
pip install .
```

Or install directly via Git:
```bash
pip install git+https://github.com/beahuman/notahuman-remember.git
```

### Development Dependencies
```bash
pip install -e ".[dev]"
```

---

## ⚙️ Configuration & Prerequisites

### 1. PostgreSQL with pgvector
nah-remember requires a running PostgreSQL instance with `pgvector` enabled:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Run schema initialization:
```bash
psql -d <database_name> -f setup_schema.sql
```

### 2. Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://localhost:5432/notahuman` |
| `EMBEDDING_URL` | OpenAI-compatible embedding API endpoint | `http://127.0.0.1:52343/v1/embeddings` |
| `EMBEDDING_MODEL` | Embedding model name (1024-dim) | `local/bge-m3` |
| `LLM_API_URL` | Optional LLM endpoint for System 1 distillation | `http://127.0.0.1:20128/v1` |
| `NAH_REMEMBER_POOL_MIN` | Min asyncpg connections per MCP process | `1` |
| `NAH_REMEMBER_POOL_MAX` | Max asyncpg connections per MCP process | `3` |

---

## 🚀 Running the MCP Server

Start the MCP server directly over standard I/O (stdio):

```bash
python3 server.py
# or using the installed entry point:
nah-remember
```

### MCP Client Configuration (e.g. Claude Desktop, Hermes, Cursor)
```json
{
  "mcpServers": {
    "nah-remember": {
      "command": "python3",
      "args": ["-m", "server"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/notahuman"
      }
    }
  }
}
```

---

## 🧪 Testing

Run the full verification suite against your local test database:

```bash
pytest
```

---

## 📜 License
Apache-2.0 / MIT
