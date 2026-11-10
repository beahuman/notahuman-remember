"""
nah-remember consolidation module.
Ecosystem: nah-neural
"""

from consolidation.evidence_sufficiency import sufficient, SPRTAccumulator
from consolidation.sufficiency_config import (
    compute_thresholds,
    PRIOR,
    COST_ACT,
    COST_WAIT,
    COST_WRONG,
)
from consolidation.ledger import net_ledger, make_event_id
from consolidation.trial_balance import (
    trial_balance,
    promote,
    check_conservation,
    check_orphans,
    DEFAULT_SEALED_ANCHORS,
)
from consolidation.wake_gate import evaluate_wake_gate, EXPECTED_PRODUCER, neural_judge_probe

__all__ = [
    "sufficient",
    "SPRTAccumulator",
    "compute_thresholds",
    "PRIOR",
    "COST_ACT",
    "COST_WAIT",
    "COST_WRONG",
    "net_ledger",
    "make_event_id",
    "trial_balance",
    "promote",
    "check_conservation",
    "check_orphans",
    "DEFAULT_SEALED_ANCHORS",
    "evaluate_wake_gate",
    "EXPECTED_PRODUCER",
    "neural_judge_probe",
]
