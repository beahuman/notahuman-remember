"""Context compaction configuration for notahuman Anthropic SDK callers.

Cookbook ref: §6.1–6.2 (automatic-context-compaction.ipynb + memory_cookbook.ipynb)
Ticket: #191 / NAH-388

Usage
-----
Any notahuman service calling the Anthropic HTTP API directly should import
from this module to get consistent compaction settings:

    from context_compaction import CONTEXT_EDITING_BETAS, get_context_editing_config, MEMORY_TOOL

    response = client.beta.messages.create(
        betas=CONTEXT_EDITING_BETAS,
        model="claude-opus-4-7",
        max_tokens=8192,
        context_editing=get_context_editing_config(),
        messages=messages,
    )

Current notahuman SDK callers: none active (matrix daemons offline 2026-04-30).
Wire these into daemon invocations when Matrix resumes.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Context editing betas
# ---------------------------------------------------------------------------

#: Beta flags that enable context editing on the Anthropic Messages API.
#: - clear_tool_uses_20250919: clears old tool call/result pairs when context grows
#: - clear_thinking_20251015: clears extended thinking blocks from context
CONTEXT_EDITING_BETAS: list[str] = [
    "clear_tool_uses_20250919",
    "clear_thinking_20251015",
]


def get_context_editing_config(
    threshold: int = 50_000,
    tool_use_retention: str = "last_3",
    thinking_retention: str = "none",
) -> dict[str, Any]:
    """Return the context_editing parameter dict for client.beta.messages.create().

    Args:
        threshold: Token count at which compaction triggers. Default 50,000.
        tool_use_retention: How many recent tool results to preserve. Default "last_3".
        thinking_retention: How to handle thinking blocks. Default "none" (clear all).

    Returns:
        Dict ready to pass as ``context_editing=`` to the Anthropic beta messages API.
    """
    return {
        "trigger": {"type": "token_count", "threshold": threshold},
        "strategy": {
            "tool_use_retention": tool_use_retention,
            "thinking_retention": thinking_retention,
        },
    }


# ---------------------------------------------------------------------------
# Memory tool definition (cookbook-canonical)
# ---------------------------------------------------------------------------

#: MCP/tool-use tool definition for save_memory.
#: Include in your agent's tools list to enable cross-session learning.
#: The tool is backed by nah-remember's ``save_memory`` MCP tool.
MEMORY_TOOL: dict[str, Any] = {
    "name": "save_memory",
    "description": (
        "Save an important finding or learned pattern to persistent memory for future sessions. "
        "Use this whenever you discover a pattern, fix, error, or best practice worth preserving."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Short identifier for this memory (e.g. 'uv-sync-canonical-command')",
            },
            "value": {
                "type": "string",
                "description": "The learning or finding to preserve in full",
            },
            "category": {
                "type": "string",
                "enum": ["pattern", "finding", "error", "best_practice"],
                "description": "Category: pattern=reusable approach, finding=discovered fact, error=failure mode, best_practice=recommended technique",
            },
        },
        "required": ["key", "value", "category"],
    },
}
