# Changelog

All notable changes to `notahuman-remember` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-07

### Added
- Standard Model Context Protocol (MCP) server for persistent agent memory backed by PostgreSQL and `pgvector`.
- Support for official `mcp` SDK (`mcp>=1.2.0`) with standard `stdio` transport.
- Multi-tier memory model:
  - **Episodic memory:** Per-session and cross-session conversational message logs.
  - **Semantic memory:** Key-value and structured fact store with embedding-based similarity lookup.
  - **Entity graph:** Entities and relational links for agent world modeling.
- Hybrid ranking engine:
  - 4-factor composite scoring combining significance, access frequency, recency decay, and reinforcement score.
  - Cross-layer retrieval and unified query interface.
- Complete environment-variable configuration interface (`NAH_REMEMBER_*` / standard fallbacks).
- Generic agent configuration examples for Hermes, Claude Desktop, Cursor, Codex, and custom MCP clients.
- Automated GitHub Actions CI/CD workflows for linting, testing, and packaging across Python 3.10–3.13.
- `pip install .` packaging support with console script `nah-remember`.
