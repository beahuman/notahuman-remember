"""
Rotated-Judge Wake Gate for nah-remember offline consolidation.
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation

Algorithm:
  1. Verify producer identity (reject uncertified sources).
  2. Enforce judge model != synthesis model (structural independence).
  3. Verify trial_balance(candidate, ledger) == "pass".
  4. Compare prior probe score vs candidate post probe score.
  5. Check Wald SPRT evidence sufficiency; if ongoing failure, revert to quarantine.
  6. Promote iff post_score >= prior_score and evidence is sufficient; else quarantine.
"""

import logging
from typing import Dict, List, Any, Optional

from consolidation.evidence_sufficiency import sufficient
from consolidation.trial_balance import trial_balance

logger = logging.getLogger(__name__)

EXPECTED_PRODUCER = "nah-neural.dreamer.propose/deterministic"


def evaluate_wake_gate(
    candidate: Dict[str, Any],
    ledger: List[Dict[str, Any]],
    baseline_scores: List[float],
    candidate_scores: List[float],
    synthesizer_model: str = "qwen2.5-1.5b-instruct",
    judge_model: str = "bge-reranker-v2-m3",
    baseline_weights: Optional[Dict[str, float]] = None,
    sealed_anchors: Optional[set] = None,
    universe_size: int = 100,
) -> Dict[str, Any]:
    """Evaluate candidate consolidation proposal through independent wake probe gate."""

    # 1. Structural independence check
    if judge_model == synthesizer_model:
        return {
            "verdict": "reject",
            "reason": f"structural violation: judge model ({judge_model}) identical to synthesizer",
        }

    # 2. Producer check
    producer = candidate.get("producer", EXPECTED_PRODUCER)
    if producer != EXPECTED_PRODUCER:
        return {
            "verdict": "reject",
            "reason": f"unrecognized producer: {producer} != {EXPECTED_PRODUCER}",
        }

    # 3. Double-entry trial balance check
    tb = trial_balance(
        candidate,
        ledger,
        baseline_weights=baseline_weights,
        sealed_anchors=sealed_anchors,
        universe_size=universe_size,
    )
    if tb["verdict"] != "pass":
        return {
            "verdict": "reject",
            "reason": f"trial balance rejected: {tb['failures']}",
            "trial_balance": tb,
        }

    # 4. Compare scores
    prior_avg = sum(baseline_scores) / max(1, len(baseline_scores))
    post_avg = sum(candidate_scores) / max(1, len(candidate_scores))

    # Convert candidate probe scores (normalized 0.0-1.0) into discrete 1-5 recall scale for SPRT
    discretized = [
        max(1, min(5, int(round(min(1.0, max(0.0, s)) * 4)) + 1)) for s in candidate_scores
    ]
    sprt = sufficient(discretized)

    # 5. Check performance regression
    if post_avg < prior_avg:
        return {
            "verdict": "quarantine",
            "reason": f"performance regression: post ({post_avg:.3f}) < prior ({prior_avg:.3f})",
            "prior_score": prior_avg,
            "post_score": post_avg,
            "sprt": sprt,
            "trial_balance": tb,
        }

    # 6. Check SPRT evidence sufficiency (Fix for Blocking Defect R1)
    if not sprt.get("sufficient", False):
        return {
            "verdict": "wait",
            "reason": f"insufficient evidence under Wald SPRT: decision={sprt['decision']}, log_lr={sprt['log_lr']:.3f} < threshold={sprt['threshold_act']:.3f}",
            "prior_score": prior_avg,
            "post_score": post_avg,
            "sprt": sprt,
            "trial_balance": tb,
        }

    return {
        "verdict": "promote",
        "reason": f"performance verified: post ({post_avg:.3f}) >= prior ({prior_avg:.3f}) with valid trial balance and sufficient SPRT evidence",
        "prior_score": prior_avg,
        "post_score": post_avg,
        "sprt": sprt,
        "trial_balance": tb,
    }
