"""
Frozen constants and thresholds for evidence sufficiency.
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation
"""

PRIOR = 0.5
COST_WAIT = 1.0
COST_ACT = 5.0
COST_WRONG = 50.0


def compute_thresholds(cost_act=COST_ACT, cost_wait=COST_WAIT, cost_wrong=COST_WRONG):
    """Derive Wald SPRT decision thresholds from cost parameters.

    alpha (acceptable false-positive rate) = cost_wait / (cost_act + cost_wait)
    beta  (acceptable false-negative rate) = cost_act  / (cost_wrong + cost_act)

    Upper boundary A = (1 - beta) / alpha  -> ACT  if cumulative LR > A
    Lower boundary B = beta / (1 - alpha)  -> WAIT if cumulative LR < B
    Inconclusive if B <= LR <= A           -> default WAIT (conservative)
    """
    alpha = cost_wait / (cost_act + cost_wait)
    beta = cost_act / (cost_wrong + cost_act)
    A = (1.0 - beta) / alpha
    B = beta / (1.0 - alpha)
    return A, B
