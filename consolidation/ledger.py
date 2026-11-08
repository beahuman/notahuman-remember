"""
Attributed event-ledger for nah-remember consolidation.
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation

net_ledger(entries: list[dict]) -> dict[str, Decimal]
Each entry has schema: event_id, source, round, memory_id, delta, basis.
"""

from collections import defaultdict
from decimal import Decimal
import hashlib
from typing import Dict, List, Any


def make_event_id(source: str, rnd: int, memory_id: str) -> str:
    raw = f"{source}:{rnd}:{memory_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def net_ledger(entries: List[Dict[str, Any]]) -> Dict[str, Decimal]:
    """Net deltas per memory_id across all entries.

    Invariant: sum(net_ledger(entries).values()) == sum(e["delta"] for e in entries).
    """
    totals = defaultdict(Decimal)
    for e in entries:
        mid = e.get("memory_id") or e.get("passage_id", "")
        delta = Decimal(str(e["delta"])) if not isinstance(e["delta"], Decimal) else e["delta"]
        totals[mid] += delta
    return dict(totals)
