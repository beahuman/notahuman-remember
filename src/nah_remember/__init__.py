"""notahuman-remember: Long-term memory MCP server backed by PostgreSQL + pgvector."""

__version__ = "0.1.0"
__author__ = "notahuman"
__license__ = "MIT"

from nah_remember.ranking import (
    MemoryLayer,
    NahRememberPhase3,
    RankedResult,
)
from nah_remember.server import (
    MCPHandler,
    NahMemoryMCPServer,
    NahRememberMCPServer,
    cli,
    main,
)

__all__ = [
    "__version__",
    "NahRememberMCPServer",
    "NahMemoryMCPServer",
    "MCPHandler",
    "NahRememberPhase3",
    "RankedResult",
    "MemoryLayer",
    "main",
    "cli",
]
