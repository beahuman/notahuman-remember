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

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

from consolidation.evidence_sufficiency import sufficient
from consolidation.trial_balance import trial_balance

logger = logging.getLogger(__name__)

EXPECTED_PRODUCER = "nah-neural.dreamer.propose/deterministic"
DEFAULT_QUARANTINE_DIR = Path(__file__).parent / "_quarantine"


def write_quarantine(
    candidate: Dict[str, Any],
    reasons: List[str],
    score_before: float,
    score_after: float,
    *,
    quarantine_dir: Optional[Path] = None,
) -> Path:
    """Write a reversible quarantine record. Never deletes the original candidate."""
    q_dir = Path(quarantine_dir) if quarantine_dir else DEFAULT_QUARANTINE_DIR
    q_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    q_id = str(uuid.uuid4())[:8]
    q_path = q_dir / f"quarantine_{ts}_{q_id}.json"

    record = {
        "candidate": candidate,
        "score_before": score_before,
        "score_after": score_after,
        "reasons": reasons,
        "quarantined_at": ts,
        "restorable": True,
    }
    q_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote reversible quarantine record to {q_path}")
    return q_path


def restore_quarantine(quarantine_path: str) -> Dict[str, Any]:
    """Restore a quarantined candidate dict from its reversible quarantine record."""
    p = Path(quarantine_path)
    if not p.exists():
        raise FileNotFoundError(f"Quarantine file does not exist: {quarantine_path}")
    record = json.loads(p.read_text(encoding="utf-8"))
    return record.get("candidate", {})


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
    quarantine_dir: Optional[Path] = None,
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
        reasons = [f"performance regression: post ({post_avg:.3f}) < prior ({prior_avg:.3f})"]
        q_path = write_quarantine(
            candidate, reasons, prior_avg, post_avg, quarantine_dir=quarantine_dir
        )
        return {
            "verdict": "quarantine",
            "reason": reasons[0],
            "quarantine_path": str(q_path),
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
