#!/usr/bin/env python3
"""
Unit and Integration Tests for Dedicated Procedural Tools (Phase 4A)
Ecosystem: nah-neural / Subsystem: nah-remember
"""

import asyncio
import os
import sys
import uuid
import pytest
import asyncpg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (
    NahRememberEngine,
    add_engine_tools_to_mcp,
    NahRememberPhase3,
    add_phase3_tools_to_mcp,
)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")


@pytest.mark.asyncio
async def test_procedural_lifecycle():
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    phase3 = NahRememberPhase3(pool)

    try:
        # 1. Save Procedure
        proc_name = f"Test Deploy Workflow {uuid.uuid4().hex[:6]}"
        steps = [
            {"step": 1, "action": "checkout_worktree", "tool": "terminal"},
            {"step": 2, "action": "run_migrations", "tool": "terminal"},
            {"step": 3, "action": "verify_health", "tool": "curl"},
        ]
        triggers = ["deploy", "launch", "fastapi"]
        unique_embedding = [0.88] + [0.01] * 1023

        proc_id = await phase3.save_procedure(
            name=proc_name,
            description="Automated playbook for deploying FastAPI LaunchAgent services",
            steps=steps,
            trigger_conditions=triggers,
            procedure_type="workflow",
            embedding=unique_embedding,
            initial_success_rate=1.0,
            agent_id="astro",
            metadata={"environment": "production"},
        )
        assert proc_id is not None
        print(f"✅ Procedure saved: {proc_id}")

        # 2. Record First Execution (Success)
        exec_1 = await phase3.record_procedure_execution(
            procedure_id=proc_id,
            success=True,
            duration_ms=450.0,
            session_id="session-test-01",
            input_data={"target": "nah-brain-reader"},
            output_data={"status": "online"},
        )
        assert exec_1["success"] is True
        assert exec_1["execution_count"] == 1
        assert exec_1["success_rate"] == 1.0
        assert exec_1["avg_duration_ms"] == 450.0
        print("✅ Execution 1 logged (Success -> 1.0)")

        # 3. Record Second Execution (Failure)
        exec_2 = await phase3.record_procedure_execution(
            procedure_id=proc_id,
            success=False,
            duration_ms=150.0,
            session_id="session-test-02",
            error_message="Connection timed out",
        )
        assert exec_2["success"] is True
        assert exec_2["execution_count"] == 2
        # (1.0 * 1 + 0.0) / 2 = 0.5
        assert exec_2["success_rate"] == 0.5
        assert exec_2["avg_duration_ms"] == 300.0
        print("✅ Execution 2 logged (Failure -> 0.5)")

        # 4. Record Third Execution (Success)
        exec_3 = await phase3.record_procedure_execution(
            procedure_id=proc_id,
            success=True,
            duration_ms=300.0,
            session_id="session-test-03",
        )
        assert exec_3["execution_count"] == 3
        # (0.5 * 2 + 1.0) / 3 = 0.6667
        assert abs(exec_3["success_rate"] - 0.6667) < 0.001
        print("✅ Execution 3 logged (Success -> 0.6667)")

        # 5. Query Procedural Layer directly
        results = await phase3.query_procedural(unique_embedding, limit=50)
        assert len(results) > 0
        found = any(r.memory_id == proc_id for r in results)
        assert found, f"Procedure {proc_id} not found in query results"
        print("✅ Procedural layer query retrieved the saved playbook")

        # 6. Test MCP Handler Wrappers
        mcp_tools = {}

        async def mock_embed(t):
            return [0.1] * 1024

        add_phase3_tools_to_mcp(mcp_tools, phase3, embedding_fn=mock_embed)

        assert "nah_remember_procedure_save" in mcp_tools
        assert "nah_remember_procedure_execute" in mcp_tools

        unique_test_name = f"MCP Procedural Test {uuid.uuid4().hex[:6]}"
        mcp_save_res = await mcp_tools["nah_remember_procedure_save"](
            name=unique_test_name,
            description="Testing MCP procedural wrapper",
            steps=[{"action": "ping"}],
            trigger_conditions=["ping"],
        )
        assert mcp_save_res["success"] is True
        mcp_pid = mcp_save_res["procedure_id"]

        mcp_exec_res = await mcp_tools["nah_remember_procedure_execute"](
            procedure_id=mcp_pid,
            success=True,
            duration_ms=50.0,
        )
        assert mcp_exec_res["success"] is True
        assert mcp_exec_res["execution_count"] == 1
        assert mcp_exec_res["success_rate"] == 1.0
        print("✅ MCP procedural tool wrappers verified")

    finally:
        await pool.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Running Procedural Tools Test Suite")
    print("=" * 60)
    asyncio.run(test_procedural_lifecycle())
    print("=" * 60)
    print("ALL PROCEDURAL TOOL TESTS PASSED (100%)")
    print("=" * 60)
