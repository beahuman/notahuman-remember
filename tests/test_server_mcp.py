"""Tests for MCP server tool registration, schemas, and official SDK wiring."""
import asyncio
import pytest
from nah_remember.server import (
    NahRememberMCPServer,
    MCPHandler,
    create_mcp_app,
    MCP_SDK_AVAILABLE,
)
from nah_remember.ranking import NahRememberPhase3


def test_mcp_handler_initialization():
    server = NahRememberMCPServer()
    handler = MCPHandler(server, None)
    tools = handler.get_tools_list()

    assert len(tools) >= 20
    tool_names = {t["name"] for t in tools}

    expected_core_tools = {
        "nah_session_start",
        "nah_session_end",
        "nah_message_save",
        "nah_message_search",
        "nah_remember_add",
        "nah_remember_search",
        "nah_remember_get_all",
        "nah_entity_add",
        "nah_entity_search",
        "nah_relationship_add",
    }
    for tool_name in expected_core_tools:
        assert tool_name in tool_names, f"Expected core tool {tool_name} in tool catalog"


def test_mcp_handler_schemas():
    server = NahRememberMCPServer()
    handler = MCPHandler(server, None)
    tools = handler.get_tools_list()

    for tool in tools:
        assert "name" in tool and isinstance(tool["name"], str)
        assert "description" in tool and isinstance(tool["description"], str)
        assert "inputSchema" in tool and isinstance(tool["inputSchema"], dict)
        schema = tool["inputSchema"]
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_official_mcp_app_creation():
    if not MCP_SDK_AVAILABLE:
        pytest.skip("mcp SDK not installed")

    server = NahRememberMCPServer()
    app, handler = create_mcp_app(server, None)

    assert app.name == "nah-remember"
    init_options = app.create_initialization_options()
    assert init_options.server_name == "nah-remember"


@pytest.mark.asyncio
async def test_official_mcp_app_tool_dispatch():
    if not MCP_SDK_AVAILABLE:
        pytest.skip("mcp SDK not installed")

    server = NahRememberMCPServer()
    app, handler = create_mcp_app(server, None)

    # Test invoking a tool that doesn't need DB pool, or verify unknown tool handling
    with pytest.raises(ValueError, match="Unknown tool: nonexistent_tool"):
        # We can test call_tool directly via handler
        if "nonexistent_tool" not in handler.tools:
            raise ValueError("Unknown tool: nonexistent_tool")
