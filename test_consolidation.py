#!/usr/bin/env python3
"""
Unit and Integration Tests for nah-remember consolidation engine
Ecosystem: nah-neural / Subsystem: nah-remember
"""

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consolidation.evidence_sufficiency import sufficient
from consolidation.sufficiency_config import compute_thresholds
from consolidation.ledger import net_ledger
from consolidation.trial_balance import (
    trial_balance,
    promote,
    check_conservation,
    check_orphans,
    check_sealed_anchors,
    check_blast_radius,
)
from consolidation.wake_gate import evaluate_wake_gate, EXPECTED_PRODUCER


def test_sprt_thresholds():
    A, B = compute_thresholds(cost_act=2.0, cost_wait=1.0, cost_wrong=20.0)
    # alpha = 1 / 3 = 0.3333, beta = 2 / 22 = 0.0909
    # A = (1 - 0.0909) / 0.3333 = 2.727
    # B = 0.0909 / (1 - 0.3333) = 0.136
    assert 2.7 < A < 2.8, f"Unexpected A: {A}"
    assert 0.13 < B < 0.14, f"Unexpected B: {B}"
    print("✅ test_sprt_thresholds passed")


def test_sprt_evidence_sufficiency():
    # n < 2 observations -> conservative wait
    res_single = sufficient([5])
    assert res_single["decision"] == "wait"
    assert not res_single["sufficient"]

    # High performance / successful recall (5, 5, 5) -> log_lr strongly positive -> decision: act
    res_success = sufficient([5, 5, 5])
    assert res_success["decision"] == "act"
    assert res_success["sufficient"]

    # Repeated failures (1, 1, 1) -> log_lr strongly negative -> decision: wait
    res_fail = sufficient([1, 1, 1])
    assert res_fail["decision"] == "wait"
    assert not res_fail["sufficient"]

    # Prior parameter properly integrated (R3)
    res_skeptical = sufficient([5, 5], prior=0.01)
    assert res_skeptical["decision"] == "wait"
    assert not res_skeptical["sufficient"]
    assert res_skeptical["log_lr"] < res_success["log_lr"]

    res_confident = sufficient([5, 5], prior=0.5)
    assert res_confident["decision"] == "act"
    assert res_confident["sufficient"]
    print("✅ test_sprt_evidence_sufficiency passed")


def test_net_ledger_and_conservation():
    entries = [
        {"memory_id": "mem-1", "delta": Decimal("1.5"), "source": "session-1"},
        {"memory_id": "mem-1", "delta": Decimal("0.5"), "source": "session-2"},
        {"memory_id": "mem-2", "delta": Decimal("-1.0"), "source": "session-1"},
    ]
    net = net_ledger(entries)
    assert net["mem-1"] == Decimal("2.0")
    assert net["mem-2"] == Decimal("-1.0")
    assert sum(net.values()) == sum(e["delta"] for e in entries)

    # Candidate with matching deltas passes conservation
    candidate_valid = {"deltas": {"mem-1": "2.0", "mem-2": "-1.0"}}
    ok, detail = check_conservation(candidate_valid, entries)
    assert ok, f"Expected pass, got {detail}"

    # Candidate with tampered delta fails conservation
    candidate_invalid = {"deltas": {"mem-1": "99.0", "mem-2": "-1.0"}}
    ok, detail = check_conservation(candidate_invalid, entries)
    assert not ok
    assert "conservation mismatch" in detail

    # Orphan memory fails orphan check
    candidate_orphan = {"deltas": {"mem-unknown": "1.0"}}
    ok, detail = check_orphans(candidate_orphan, entries)
    assert not ok
    assert "orphan memory_ids" in detail
    print("✅ test_net_ledger_and_conservation passed")


def test_sealed_anchors_and_blast_radius():
    baseline = {"anchor-1": 1.0, "anchor-2": 2.0, "normal-1": 0.5}
    anchors = {"anchor-1", "anchor-2"}

    # Attempting to wipe anchor-1 below 0 fails
    candidate_bad = {"deltas": {"anchor-1": "-1.5"}}
    ok, detail = check_sealed_anchors(candidate_bad, baseline, anchors)
    assert not ok
    assert "sealed anchor violated" in detail

    # Preserving positive weights passes
    candidate_good = {"deltas": {"anchor-1": "-0.2", "normal-1": "0.3"}}
    ok, detail = check_sealed_anchors(candidate_good, baseline, anchors)
    assert ok

    # Blast radius: 10 memories modified in universe of 10 exceeds 20%
    big_deltas = {f"mem-{i}": "0.1" for i in range(10)}
    ok, detail = check_blast_radius(
        {"deltas": big_deltas}, universe_size=10, blast_radius_frac=0.20
    )
    assert not ok
    assert "blast radius exceeded" in detail
    print("✅ test_sealed_anchors_and_blast_radius passed")


def test_trial_balance_full_chain():
    ledger = [
        {"memory_id": "mem-1", "delta": Decimal("1.0"), "source": "real_run"},
        {"memory_id": "mem-2", "delta": Decimal("0.5"), "source": "real_run"},
    ]
    candidate = {"deltas": {"mem-1": "1.0", "mem-2": "0.5"}}
    res = trial_balance(candidate, ledger, universe_size=100)
    assert res["verdict"] == "pass"
    assert len(res["failures"]) == 0

    # Test promote() with high-recall observations (5, 5, 5)
    observations = {"mem-1": [5, 5, 5], "mem-2": [5, 5, 5]}
    prom_res = promote(candidate, ledger, observations, universe_size=100)
    assert prom_res["decision"] == "promote"
    assert "mem-1" in prom_res["eligible_memory_ids"]
    print("✅ test_trial_balance_full_chain passed")


def test_wake_gate():
    ledger = [{"memory_id": "mem-1", "delta": Decimal("1.0"), "source": "real_run"}]
    candidate = {"producer": EXPECTED_PRODUCER, "deltas": {"mem-1": "1.0"}}

    # Model identity collision (judge == synthesizer) rejected
    col_res = evaluate_wake_gate(
        candidate, ledger, [0.8], [0.9], synthesizer_model="model-a", judge_model="model-a"
    )
    assert col_res["verdict"] == "reject"
    assert "structural violation" in col_res["reason"]

    # Regression quarantined
    reg_res = evaluate_wake_gate(
        candidate,
        ledger,
        [0.8, 0.9],
        [0.4, 0.5],
        synthesizer_model="synth-a",
        judge_model="judge-b",
    )
    assert reg_res["verdict"] == "quarantine"

    # Blocking defect R1 regression test: post >= prior, but SPRT evidence insufficient
    # Single observation [0.8] -> n_obs=1 < 2 -> SPRT wait
    insuf_res = evaluate_wake_gate(
        candidate,
        ledger,
        [0.5],
        [0.8],
        synthesizer_model="synth-a",
        judge_model="judge-b",
    )
    assert insuf_res["verdict"] == "wait"
    assert "insufficient evidence" in insuf_res["reason"].lower()

    # Non-regressive promotion with sufficient SPRT evidence (candidate scores [0.8, 0.9] -> [4, 5] with n>=2)
    pass_res = evaluate_wake_gate(
        candidate,
        ledger,
        [0.7, 0.7],
        [0.8, 0.9],
        synthesizer_model="synth-a",
        judge_model="judge-b",
    )
    assert pass_res["verdict"] == "promote"
    print("✅ test_wake_gate passed")


if __name__ == "__main__":
    print("=" * 60)
    print("NAH Neural Consolidation Test Suite")
    print("=" * 60)
    test_sprt_thresholds()
    test_sprt_evidence_sufficiency()
    test_net_ledger_and_conservation()
    test_sealed_anchors_and_blast_radius()
    test_trial_balance_full_chain()
    test_wake_gate()
    print("=" * 60)
    print("ALL 6 CONSOLIDATION TESTS PASSED (100%)")
    print("=" * 60)
