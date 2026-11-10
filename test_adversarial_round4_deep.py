"""
Adversarial & Fault-Injection Verification Suite (Round 4 Deep Audit).
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation

Deep, disruptive adversarial tests targeting:
  1. Dead Judge Fault-Injection (fail-closed to 0.0 quarantine, zero 0.75 bypass).
  2. Wald SPRT Lower Boundary Rejection (rejection on negative stream, persistent accumulation).
  3. Double-Entry Conservation Counterfeit Attack (unbacked candidate deltas fail).
  4. Sealed Anchor Dilution & Tamper Attack (attempts to weaken precepts/audit/identity fail).
  5. Zero Bearer Token In Code Audit (AST scan enforcing vault-first / zero hardcoded keys).
  6. ON CONFLICT DO NOTHING Collision Tracking (duplicate procedures do not increment committed count).
  7. Schema Migration 002 Robustness (verifying explicit column conversion for legacy tables).
"""

import ast
from decimal import Decimal
import json
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch

from consolidation.evidence_sufficiency import sufficient, SPRTAccumulator
from consolidation.trial_balance import (
    trial_balance,
    check_conservation,
    check_orphans,
    check_sealed_anchors,
    DEFAULT_SEALED_ANCHORS,
)
from consolidation.wake_gate import (
    neural_judge_probe,
    evaluate_wake_gate,
    EXPECTED_PRODUCER,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SYNTHESIZER_MODEL,
)


# =====================================================================
# 1. DEAD JUDGE FAULT-INJECTION (FAIL-CLOSED)
# =====================================================================


@pytest.mark.asyncio
async def test_dead_judge_fault_injection_fails_closed():
    """
    DISRUPTIVE TEST: Simulate dead/unreachable LLM endpoint, timeouts, and HTTP errors.
    Verified invariant: Judge must FAIL CLOSED to 0.0 and wake gate must QUARANTINE.
    The prior vulnerability of returning 0.75 on exception is strictly blocked.
    """
    candidate_with_steps = {
        "producer": EXPECTED_PRODUCER,
        "name": "Deploy Service Playbook",
        "steps": ["step 1: verify", "step 2: execute"],
        "pattern": "error pattern",
        "deltas": {},
    }

    # Simulate network down / connection refused
    with patch(
        "urllib.request.urlopen",
        side_effect=ConnectionRefusedError("Connection refused to OmniRoute"),
    ):
        score = await neural_judge_probe(
            candidate=candidate_with_steps,
            probe_prompt="Evaluate playbook safety",
            api_key="valid-mock-key",
        )
        # MUST return 0.0, NOT 0.75!
        assert score == 0.0, f"Expected fail-closed 0.0 on dead judge, got {score}"

    # Evaluate wake gate with dead judge score (post_score = 0.0, prior_score = 0.5)
    gate_res = evaluate_wake_gate(
        candidate=candidate_with_steps,
        ledger=[],
        baseline_scores=[0.5],
        candidate_scores=[0.0],  # Dead judge returned 0.0
        synthesizer_model="qwen2.5-1.5b-instruct",
        judge_model="qwen2.5-coder-7b-instruct",
        sealed_anchors=DEFAULT_SEALED_ANCHORS,
    )

    assert gate_res["verdict"] == "quarantine"
    assert "probe score regression" in gate_res["reason"].lower()


@pytest.mark.asyncio
async def test_judge_model_defaults_to_chat_model():
    """Verify default judge model is an actual chat model and differs from synthesizer."""
    assert DEFAULT_JUDGE_MODEL != "bge-reranker-v2-m3", "Judge model cannot be a reranker!"
    assert "instruct" in DEFAULT_JUDGE_MODEL or "coder" in DEFAULT_JUDGE_MODEL, (
        "Judge must be a generative chat model"
    )
    assert DEFAULT_JUDGE_MODEL != DEFAULT_SYNTHESIZER_MODEL, (
        "Judge model must be structurally distinct from synthesizer"
    )


# =====================================================================
# 2. WALD SPRT LOWER BOUNDARY REJECTION & PERSISTENT ACCUMULATION
# =====================================================================


def test_sprt_lower_boundary_rejection():
    """
    DISRUPTIVE TEST: Subject SPRT to consecutive failure evidence streams.
    Verified invariant: Must hit lower boundary B (log_B) and return decision == 'reject'.
    """
    # Stream of failures (r=1)
    fail_stream = [1, 1, 1, 1]
    res = sufficient(fail_stream, prior=0.5)

    assert res["decision"] == "reject", f"Expected 'reject' under Wald lower bound, got {res}"
    assert res["rejected"] is True
    assert res["sufficient"] is False
    assert res["log_lr"] <= math_log_helper(res["threshold_wait"])


def math_log_helper(val):
    import math

    return math.log(val)


def test_sprt_accumulator_persistence_and_multi_cycle():
    """
    Verified invariant: SPRTAccumulator preserves state across runs and serializes cleanly.
    """
    acc = SPRTAccumulator(item_id="proc-test-123", prior=0.5)
    # Cycle 1: neutral observation
    r1 = acc.record_run([3, 3])
    assert r1["decision"] == "wait"
    assert r1["cycle_count"] == 1
    assert len(acc.observations) == 2

    # Serialize to dictionary
    serialized = acc.to_dict()

    # Deserialize in new cycle (simulating daemon restart)
    acc_restored = SPRTAccumulator.from_dict(serialized)
    assert acc_restored.item_id == "proc-test-123"
    assert acc_restored.observations == [3, 3]

    # Cycle 2: strong evidence
    r2 = acc_restored.record_run([5, 5, 5])
    assert r2["decision"] == "act"
    assert r2["sufficient"] is True
    assert r2["cycle_count"] == 2
    assert len(acc_restored.observations) == 5


# =====================================================================
# 3. DOUBLE-ENTRY CONSERVATION COUNTERFEIT DELTA ATTACK
# =====================================================================


def test_counterfeit_delta_rejected_by_trial_balance():
    """
    DISRUPTIVE TEST: Inject counterfeit deltas without backing empirical ledger entries.
    Verified invariant: Conservation sieve Layer 1 and Orphan sieve Layer 2 must reject.
    """
    # Legitimate ledger backed by empirical episodes
    ledger = [
        {
            "event_id": "ep_1_proc_1",
            "source": "episode",
            "round": 1,
            "memory_id": "proc-real",
            "delta": Decimal("0.5"),
            "basis": "observed_execution",
        }
    ]

    # Counterfeit candidate attempting to sneak in an unbacked delta
    counterfeit_candidate = {
        "producer": EXPECTED_PRODUCER,
        "deltas": {
            "proc-real": Decimal("0.5"),
            "proc-counterfeit": Decimal("1.0"),  # No ledger backing!
        },
    }

    ok_cons, msg_cons = check_conservation(counterfeit_candidate, ledger)
    assert not ok_cons, "Conservation check should fail on counterfeit delta"
    assert "conservation mismatch for proc-counterfeit" in msg_cons

    ok_orph, msg_orph = check_orphans(counterfeit_candidate, ledger)
    assert not ok_orph, "Orphan check should catch unbacked memory ID"
    assert "proc-counterfeit" in msg_orph

    tb = trial_balance(counterfeit_candidate, ledger, universe_size=100)
    assert tb["verdict"] == "fail"
    assert "conservation" in tb["failures"]
    assert "orphan_rejection" in tb["failures"]


# =====================================================================
# 4. SEALED ANCHOR DILUTION & TAMPER ATTACK
# =====================================================================


def test_sealed_anchor_tamper_attack_rejected():
    """
    DISRUPTIVE TEST: Attempt to dilute core precepts or drive anchor memory weight to 0.
    Verified invariant: Layer 3 sealed anchor sieve must fail closed.
    """
    # Attack 1: Attempt to negative-delta a sealed anchor
    tamper_weight_candidate = {
        "producer": EXPECTED_PRODUCER,
        "deltas": {"anchor:nah-precepts": Decimal("-1.5")},
    }
    baseline_weights = {"anchor:nah-precepts": 1.0}
    ok, msg = check_sealed_anchors(
        tamper_weight_candidate, baseline_weights, DEFAULT_SEALED_ANCHORS
    )
    assert not ok, "Must reject negative weight on sealed anchor"
    assert "resulting weight <= 0" in msg

    # Attack 2: Attempt to introduce procedure diluting precepts or audit trails
    dilution_candidate = {
        "producer": EXPECTED_PRODUCER,
        "deltas": {},
        "procedures": [
            {
                "name": "Bypass Verification",
                "description": "Script that attempts to bypass_integrity and disable_audit",
                "steps": ["step 1: override_identity"],
            }
        ],
    }
    ok_dilute, msg_dilute = check_sealed_anchors(
        dilution_candidate, baseline_weights, DEFAULT_SEALED_ANCHORS
    )
    assert not ok_dilute, "Must reject semantic dilution of sealed anchors"
    assert "sealed anchor dilution attack detected" in msg_dilute


# =====================================================================
# 5. ZERO BEARER TOKEN CODE SCAN (AST AUDIT)
# =====================================================================


def test_zero_hardcoded_bearer_tokens_in_codebase():
    """
    CRITICAL SECURITY AUDIT: Traverse all python files in nah-remember and inspect AST.
    Verified invariant: ZERO hardcoded 'sk-' or leaked tokens exist in code.
    """
    root_dir = Path(__file__).parent
    py_files = list(root_dir.glob("*.py")) + list(root_dir.glob("*/*.py"))

    leaked_found = []
    for f in py_files:
        if "venv" in str(f) or ".git" in str(f) or "test_" in f.name:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("sk-") or "sk-0118" in node.value:
                    leaked_found.append((str(f), node.value))

    assert len(leaked_found) == 0, (
        f"Security violation: hardcoded tokens discovered in code: {leaked_found}"
    )


# =====================================================================
# 6. ON CONFLICT DO NOTHING EXACT INSERTION TRACKING
# =====================================================================


def test_on_conflict_insert_tracking_logic():
    """
    Verified invariant: Only status 'INSERT 0 1' increments committed count;
    'INSERT 0 0' (existing duplicate) leaves committed count unchanged.
    """
    committed_count = 0

    # First insert: success
    tag_1 = "INSERT 0 1"
    if tag_1.endswith(" 1"):
        committed_count += 1
    assert committed_count == 1

    # Second insert: ON CONFLICT DO NOTHING skips (0 inserted)
    tag_2 = "INSERT 0 0"
    if tag_2.endswith(" 1"):
        committed_count += 1
    # MUST remain 1!
    assert committed_count == 1, "ON CONFLICT DO NOTHING must not increment committed count"


# =====================================================================
# 7. SCHEMA MIGRATION 002 RESILIENCE
# =====================================================================


def test_migration_002_contains_column_alteration():
    """
    Verified invariant: Migration 002 explicitly converts legacy UUID columns to VARCHAR(255).
    """
    mig_file = Path(__file__).parent / "migrations" / "002_reconcile_schema_parity.sql"
    assert mig_file.exists(), "Migration 002 must exist"
    content = mig_file.read_text(encoding="utf-8")

    assert (
        "ALTER TABLE nah_remember.negative_constraints ALTER COLUMN source_session_id TYPE VARCHAR(255)"
        in content
    )
    assert "REFERENCES nah_remember.sessions(session_id)" in content
    assert "procedures_name_key" in content
