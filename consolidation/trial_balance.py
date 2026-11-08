"""
Conservation-identity trial-balance for nah-remember offline consolidation.
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation

Five-layer sieve chain:
  Layer 1 — Conservation identity: deltas must equal net_ledger(real_sources)
  Layer 2 — Orphan rejection: all modified memory IDs must have backing ledger entries
  Layer 3 — Sealed anchors: critical foundation memories must remain positive
  Layer 4 — Bounded blast-radius: modifications bounded (default <= 20% of universe)
  Layer 5 — Promotion rule: evidence sufficiency (Wald SPRT) gated promotion
"""

from decimal import Decimal
from typing import Dict, List, Set, Tuple, Optional, Any

from consolidation.evidence_sufficiency import sufficient
from consolidation.ledger import net_ledger

DEFAULT_BLAST_RADIUS_FRAC = 0.20


def _get_deltas(candidate: Dict[str, Any]) -> Dict[str, Decimal]:
    """Return Decimal-valued deltas from candidate."""
    if "_deltas_decimal" in candidate:
        return candidate["_deltas_decimal"]
    return {mid: Decimal(str(d)) for mid, d in candidate.get("deltas", {}).items()}


def check_conservation(candidate: Dict[str, Any], ledger: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Layer 1: each delta must equal net_ledger(real_entries)[mid] exactly."""
    deltas = _get_deltas(candidate)
    real_entries = [e for e in ledger if e.get("source") != "synthetic"]
    net_real = net_ledger(real_entries)

    failures = []
    for mid, d in deltas.items():
        expected = net_real.get(mid, Decimal(0))
        if d != expected:
            failures.append(f"conservation mismatch for {mid}: candidate={d} net_ledger={expected}")

    if failures:
        return False, "; ".join(failures)
    return True, "ok"


def check_orphans(candidate: Dict[str, Any], ledger: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Layer 2: every mid in deltas must appear in net_ledger(real_entries)."""
    deltas = _get_deltas(candidate)
    real_entries = [e for e in ledger if e.get("source") != "synthetic"]
    net_real = net_ledger(real_entries)

    orphans = [mid for mid in deltas if mid not in net_real]
    if orphans:
        return False, f"orphan memory_ids: {orphans}"
    return True, "ok"


def check_sealed_anchors(
    candidate: Dict[str, Any],
    baseline_weights: Dict[str, float],
    anchors: Set[str],
) -> Tuple[bool, str]:
    """Layer 3: anchor memories must maintain w0 + delta > 0 after applying candidate."""
    deltas = _get_deltas(candidate)
    w0: Dict[str, Decimal] = {mid: Decimal(str(w)) for mid, w in baseline_weights.items()}

    failures = []
    for anchor_id in anchors:
        current_w = w0.get(anchor_id, Decimal(1))
        delta = deltas.get(anchor_id, Decimal(0))
        if current_w + delta <= Decimal(0):
            failures.append(f"sealed anchor violated: {anchor_id} resulting weight <= 0")

    if failures:
        return False, "; ".join(failures)
    return True, "ok"


def check_blast_radius(
    candidate: Dict[str, Any],
    universe_size: int,
    blast_radius_frac: float = DEFAULT_BLAST_RADIUS_FRAC,
) -> Tuple[bool, str]:
    """Layer 4: modified memory fraction must not exceed blast_radius_frac."""
    deltas = _get_deltas(candidate)
    if universe_size <= 0:
        return True, "ok"

    frac = len(deltas) / universe_size
    if frac > blast_radius_frac:
        return (
            False,
            f"blast radius exceeded: {len(deltas)}/{universe_size} ({frac:.1%}) > {blast_radius_frac:.1%}",
        )
    return True, "ok"


def trial_balance(
    candidate: Dict[str, Any],
    ledger: List[Dict[str, Any]],
    baseline_weights: Optional[Dict[str, float]] = None,
    sealed_anchors: Optional[Set[str]] = None,
    universe_size: int = 100,
    blast_radius_frac: float = DEFAULT_BLAST_RADIUS_FRAC,
) -> Dict[str, Any]:
    """Run the conservation sieve chain on a candidate consolidation proposal."""
    base_w = baseline_weights or {}
    anchors = sealed_anchors or set()

    checks: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []

    layers = [
        ("conservation", check_conservation(candidate, ledger)),
        ("orphan_rejection", check_orphans(candidate, ledger)),
        ("sealed_anchor", check_sealed_anchors(candidate, base_w, anchors)),
        ("blast_radius", check_blast_radius(candidate, universe_size, blast_radius_frac)),
    ]

    for name, (ok, detail) in layers:
        checks[name] = {"ok": ok, "detail": detail}
        if not ok:
            failures.append(name)

    return {
        "verdict": "fail" if failures else "pass",
        "checks": checks,
        "failures": failures,
    }


def promote(
    candidate: Dict[str, Any],
    ledger: List[Dict[str, Any]],
    observations: Dict[str, List[int]],
    baseline_weights: Optional[Dict[str, float]] = None,
    sealed_anchors: Optional[Set[str]] = None,
    universe_size: int = 100,
    blast_radius_frac: float = DEFAULT_BLAST_RADIUS_FRAC,
) -> Dict[str, Any]:
    """Sieve candidate through trial_balance then apply Layer 5 Wald SPRT evidence sufficiency."""
    tb_result = trial_balance(
        candidate,
        ledger,
        baseline_weights=baseline_weights,
        sealed_anchors=sealed_anchors,
        universe_size=universe_size,
        blast_radius_frac=blast_radius_frac,
    )

    if tb_result["verdict"] != "pass":
        return {
            "decision": "reject",
            "reason": f"trial_balance failed: {tb_result['failures']}",
            "trial_balance": tb_result,
            "sprt_checks": {},
        }

    deltas = _get_deltas(candidate)
    sprt_checks = {}
    eligible_promotions = []

    for mid in deltas.keys():
        stream = observations.get(mid, [3, 3])
        sprt = sufficient(stream)
        sprt_checks[mid] = sprt
        if sprt["sufficient"]:
            eligible_promotions.append(mid)

    if not eligible_promotions:
        return {
            "decision": "wait",
            "reason": "insufficient evidence for promotion under Wald SPRT",
            "trial_balance": tb_result,
            "sprt_checks": sprt_checks,
        }

    return {
        "decision": "promote",
        "eligible_memory_ids": eligible_promotions,
        "trial_balance": tb_result,
        "sprt_checks": sprt_checks,
    }
