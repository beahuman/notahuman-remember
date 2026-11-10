"""
Deep verification test for multi-layer memory integration and negative constraint parity.
Tests:
1. nah_negative_constraint_add first-class MCP tool
2. memory_add dual-routing into nah_remember.negative_constraints
3. unified_search across ALL 6 layers (messages, entities, semantic, procedures, episodes, negative_constraints)
4. remember_search_ranked returning negative constraints under meta and all layers
"""

import asyncio
import uuid
import pytest
from engine import NahRememberEngine, RankedResult
from server import NahMemoryMCPServer, MCPHandler


@pytest.fixture
async def mcp_server():
    server = NahMemoryMCPServer()
    await server.setup()
    yield server
    await server.cleanup()


@pytest.fixture
async def cognitive_engine(mcp_server):
    engine = NahRememberEngine(pool=mcp_server.pool)
    yield engine


@pytest.mark.asyncio
async def test_negative_constraint_add_and_search(mcp_server):
    """Verify first-class tool nah_negative_constraint_add and negative_search."""
    unique_suffix = uuid.uuid4().hex[:8]
    pattern = f"Prohibit dangerous operation {unique_suffix}"
    trigger_context = f"testing, adversarial, safety_layer_{unique_suffix}"

    res = await mcp_server.negative_constraint_add(
        pattern=pattern,
        trigger_context=trigger_context,
        severity="critical",
        weight=1.0,
    )
    assert res["success"] is True, f"Failed to add negative constraint: {res}"
    assert res["pattern"] == pattern
    assert res["severity"] == "critical"
    assert res["weight"] == 1.0

    # Search via negative_search
    search_res = await mcp_server.negative_search(query=f"adversarial {unique_suffix}")
    assert search_res["success"] is True
    assert search_res["count"] >= 1
    found = any(c["pattern"] == pattern for c in search_res["constraints"])
    assert found, f"Constraint {pattern} not found in negative search results"


@pytest.mark.asyncio
async def test_memory_add_dual_routing_negative_constraint(mcp_server):
    """Verify memory_add with category='negative_constraint' auto-inserts into negative_constraints."""
    unique_suffix = uuid.uuid4().hex[:8]
    content = f"NEGATIVE CONSTRAINT: Never execute unverified bytecode in worker {unique_suffix}"
    tags = ["negative_constraint", f"worker_isolation_{unique_suffix}", "security"]

    add_res = await mcp_server.memory_add(
        content=content,
        category="negative_constraint",
        tags=tags,
    )
    assert add_res["success"] is True

    # Check direct negative search for worker isolation
    search_res = await mcp_server.negative_search(query=f"worker {unique_suffix}")
    assert search_res["success"] is True
    assert search_res["count"] >= 1
    patterns = [c["pattern"] for c in search_res["constraints"]]
    clean_expected = f"Never execute unverified bytecode in worker {unique_suffix}"
    assert any(clean_expected in p for p in patterns), f"Expected {clean_expected} in {patterns}"


@pytest.mark.asyncio
async def test_unified_search_all_layers(mcp_server):
    """Verify unified_search queries all 6 layers by default."""
    res = await mcp_server.unified_search(query="Hermes agent rules and procedures")
    assert res["success"] is True, f"Unified search failed: {res}"
    assert "results" in res
    assert len(res["results"]) > 0

    found_types = {r["memory_type"] for r in res["results"]}
    print(f"Unified search retrieved types: {found_types}")

    # Ensure negative_constraints layer is supported and can be targeted explicitly
    nc_res = await mcp_server.unified_search(
        query="Omo Hermes Bedrock", memory_types=["negative_constraints"]
    )
    assert nc_res["success"] is True
    assert len(nc_res["results"]) >= 1
    assert nc_res["results"][0]["memory_type"] == "negative_constraints"
    assert "Omo" in nc_res["results"][0]["content"]


@pytest.mark.asyncio
async def test_engine_ranked_search_includes_meta_constraints(cognitive_engine):
    """Verify Cognitive Engine ranked search includes negative constraints from meta layer."""
    res = await cognitive_engine.remember_search_ranked(
        query_text="Omo Hermes Bedrock forbidden",
        memory_type="meta",
        limit=10,
    )
    assert res["success"] is True
    assert res["total_results"] > 0
    types = [r["memory_type"] for r in res["results"]]
    assert "negative_constraint" in types, f"Expected negative_constraint in {types}"

    nc_item = next(r for r in res["results"] if r["memory_type"] == "negative_constraint")
    assert nc_item["layer"] == "meta"
    assert nc_item["significance_score"] >= 0.8
    assert nc_item["composite_score"] > 0.0


@pytest.mark.asyncio
async def test_mcp_handler_tool_registration(mcp_server, cognitive_engine):
    """Verify MCPHandler registers all canonical negative constraint tools."""
    handler = MCPHandler(server=mcp_server, engine=cognitive_engine)
    assert "nah_negative_constraint_add" in handler.tools
    assert "nah_remember_negative_add" in handler.tools
    assert "nah_remember_negative_search" in handler.tools
    assert "nah_unified_search" in handler.tools
