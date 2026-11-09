#!/usr/bin/env python3
"""
Test Suite for Wave 5: Reversible Quarantine, Reflex Corrections, and System 1 Distillation.
Ecosystem: nah-neural / Subsystem: nah-remember
"""

import asyncio
from decimal import Decimal
import os
from pathlib import Path
import sys
import tempfile
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "consolidation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "distillation"))

from consolidation.reflex import reflex, undo_reflex
from consolidation.wake_gate import evaluate_wake_gate, restore_quarantine, write_quarantine
from distillation.system_one_distiller import SystemOneDistiller


def test_reflex_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mid = f"proc-test-{uuid.uuid4().hex[:6]}"

        # 1. Zero delta is rejected
        res_zero = reflex(mid, 0.0, 1.0, reflex_dir=tmp_path)
        assert res_zero["applied"] is False
        assert "zero" in res_zero["reason"]

        # 2. Exceeding max delta is rejected
        res_huge = reflex(mid, 15.0, 1.0, reflex_dir=tmp_path)
        assert res_huge["applied"] is False
        assert "exceeds safety cap" in res_huge["reason"]

        # 3. Valid delta applied
        res = reflex(mid, Decimal("-0.3"), 1.0, reflex_dir=tmp_path)
        assert res["applied"] is True
        assert res["w_before"] == 1.0
        assert round(res["w_after"], 2) == 0.70
        assert res["undo_token"] is not None

        # 4. Undo token file created
        undo_file = tmp_path / f"reflex_{res['undo_token']}.json"
        assert undo_file.exists()

        # 5. Undo reflex restores previous weight
        undo_res = undo_reflex(res["undo_token"], reflex_dir=tmp_path)
        assert undo_res["reversed"] is True
        assert undo_res["w_restored"] == 1.0

        # 6. Re-undo fails (token spent)
        spent_res = undo_reflex(res["undo_token"], reflex_dir=tmp_path)
        assert spent_res["reversed"] is False
        assert "already reversed" in spent_res["reason"]

    print("✅ test_reflex_lifecycle passed")


def test_quarantine_and_restoration():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        candidate = {
            "producer": "nah-neural.dreamer.propose/deterministic",
            "procedures": [{"memory_id": "proc-quarantine-test", "steps": ["step1"]}],
            "deltas": {"proc-1": Decimal("0.5"), "proc-2": Decimal("-0.5")},
        }

        # 1. Write quarantine
        q_path = write_quarantine(
            candidate,
            reasons=["test failure regression"],
            score_before=0.85,
            score_after=0.60,
            quarantine_dir=tmp_path,
        )
        assert q_path.exists()

        # 2. Restore quarantine
        restored = restore_quarantine(str(q_path))
        assert restored["producer"] == candidate["producer"]
        assert len(restored["procedures"]) == 1
        assert restored["procedures"][0]["memory_id"] == "proc-quarantine-test"

    print("✅ test_quarantine_and_restoration passed")


def test_wake_gate_regression_quarantine():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        candidate = {
            "producer": "nah-neural.dreamer.propose/deterministic",
            "deltas": {"proc-1": Decimal("0.5"), "proc-2": Decimal("-0.5")},
        }
        ledger = [
            {"memory_id": "proc-1", "delta": Decimal("0.5"), "source": "dreamer_engine"},
            {"memory_id": "proc-2", "delta": Decimal("-0.5"), "source": "dreamer_engine"},
        ]

        # Post-score lower than baseline -> triggers quarantine
        gate = evaluate_wake_gate(
            candidate=candidate,
            ledger=ledger,
            baseline_scores=[0.9, 0.95],
            candidate_scores=[0.5, 0.55],
            quarantine_dir=tmp_path,
        )
        assert gate["verdict"] == "quarantine"
        assert gate["quarantine_path"] is not None
        assert Path(gate["quarantine_path"]).exists()

        # Confirm restorable
        restored = restore_quarantine(gate["quarantine_path"])
        assert restored["producer"] == candidate["producer"]

    print("✅ test_wake_gate_regression_quarantine passed")


@pytest.mark.asyncio
async def test_system_one_distiller_live_call():
    distiller = SystemOneDistiller()
    prompt = (
        "Summarize in 10 words or less: Memory consolidation passed all unit and integration tests."
    )
    resp = await distiller.generate_completion(prompt)
    assert len(resp) > 0
    print(f"✅ Local System 1 live completion verified: {resp}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Wave 5 Reflex, Quarantine & Distillation Test Suite")
    print("=" * 60)
    test_reflex_lifecycle()
    test_quarantine_and_restoration()
    test_wake_gate_regression_quarantine()
    asyncio.run(test_system_one_distiller_live_call())
    print("=" * 60)
    print("ALL WAVE 5 TESTS PASSED (100%)")
    print("=" * 60)
