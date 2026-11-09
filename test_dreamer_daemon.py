#!/usr/bin/env python3
"""
Unit and Integration Tests for Dreamer Consolidation Daemon (Phase 4A)
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation
"""

import asyncio
from decimal import Decimal
import os
import sys
import uuid
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "consolidation"))

from consolidation.dreamer_daemon import DreamerDaemon
from consolidation.evidence_sufficiency import sufficient
from consolidation.wake_gate import evaluate_wake_gate, EXPECTED_PRODUCER
from consolidation.trial_balance import trial_balance

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")


def test_candidate_synthesis():
    daemon = DreamerDaemon()
    sample_episodes = [
        {
            "id": uuid.uuid4(),
            "task_type": "deploy_service",
            "hunt_context": "launchd",
            "outcome": "success",
            "narrative": "Successfully deployed launchd service",
            "agent_id": "astro",
            "events": [
                {"tool_name": "write_file", "description": "Write plist"},
                {"tool_name": "run_command", "description": "launchctl load"},
            ],
        },
        {
            "id": uuid.uuid4(),
            "task_type": "deploy_service",
            "hunt_context": "launchd",
            "outcome": "success",
            "narrative": "Another successful deploy",
            "agent_id": "astro",
            "events": [
                {"tool_name": "write_file", "description": "Write plist"},
                {"tool_name": "run_command", "description": "launchctl load"},
            ],
        },
        {
            "id": uuid.uuid4(),
            "task_type": "git_push",
            "hunt_context": "branch_guard",
            "outcome": "failure",
            "events": [
                {
                    "tool_name": "run_command",
                    "event_type": "error",
                    "description": "Direct push to main blocked by git-guard",
                },
            ],
        },
    ]

    synth = daemon.synthesize_candidates(sample_episodes)
    candidate = synth["candidate"]
    ledger = synth["ledger"]
    obs = synth["observations"]

    assert len(ledger) > 0
    assert candidate["producer"] == EXPECTED_PRODUCER
    assert len(candidate["procedures"]) == 1
    assert candidate["procedures"][0]["memory_id"] == "proc-deploy_service"
    assert len(candidate["constraints"]) == 1
    assert "Direct push" in candidate["constraints"][0]["pattern"]

    # Verify Wald SPRT observations: 2 successful episodes -> stream of [5, 5]
    assert obs["proc-deploy_service"] == [5, 5]
    sprt_res = sufficient(obs["proc-deploy_service"])
    assert sprt_res["decision"] == "act"
    assert sprt_res["sufficient"] is True
    print("✅ test_candidate_synthesis passed")


def test_consolidation_gating_chain():
    # Valid candidate with balanced deltas
    candidate = {
        "producer": EXPECTED_PRODUCER,
        "deltas": {"proc-1": Decimal("0.5"), "proc-2": Decimal("-0.5")},
    }
    ledger = [
        {"memory_id": "proc-1", "delta": Decimal("0.5"), "source": "dreamer_engine"},
        {"memory_id": "proc-2", "delta": Decimal("-0.5"), "source": "dreamer_engine"},
    ]

    # 1. Trial balance check
    tb = trial_balance(candidate, ledger, universe_size=100)
    assert tb["verdict"] == "pass"

    # 2. Wake gate with independent models
    gate = evaluate_wake_gate(
        candidate=candidate,
        ledger=ledger,
        baseline_scores=[0.8, 0.8],
        candidate_scores=[0.9, 0.95],
        synthesizer_model="qwen2.5-1.5b-instruct",
        judge_model="bge-reranker-v2-m3",
    )
    assert gate["verdict"] == "promote"
    print("✅ test_consolidation_gating_chain passed")


@pytest.mark.asyncio
async def test_dreamer_dry_run_cycle():
    daemon = DreamerDaemon(db_url=DATABASE_URL)
    await daemon.setup()
    try:
        res = await daemon.run_consolidation_cycle(dry_run=True, limit=10)
        assert res["success"] is True
        assert res["dry_run"] is True
        print(f"✅ Dreamer dry-run cycle completed successfully: {res}")
    finally:
        await daemon.cleanup()


if __name__ == "__main__":
    print("=" * 60)
    print("Running Dreamer Daemon Consolidation Test Suite")
    print("=" * 60)
    test_candidate_synthesis()
    test_consolidation_gating_chain()
    asyncio.run(test_dreamer_dry_run_cycle())
    print("=" * 60)
    print("ALL DREAMER DAEMON TESTS PASSED (100%)")
    print("=" * 60)
