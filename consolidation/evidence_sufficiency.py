"""
Bayesian evidence sufficiency decision via Wald Sequential Probability Ratio Test (SPRT).
Ecosystem: nah-neural / Subsystem: nah-remember / Module: consolidation

sufficient(stream, prior, cost_act, cost_wait, cost_wrong) -> dict

Thresholds derived from cost ratios:
  alpha = cost_wait / (cost_act + cost_wait)
  beta  = cost_act  / (cost_wrong + cost_act)
  A = (1 - beta) / alpha  (upper boundary -> ACT)
  B = beta / (1 - alpha)  (lower boundary -> WAIT)

Likelihood ratio per observation (recall / performance r in {1..5}):
  LR(r) = P(r | recovered/success) / P(r | unrecovered/failure) = r / (6 - r)

Accumulated in log-space with prior log-odds initialization for numerical stability.
"""

import math
from typing import Dict, List, Any


def sufficient(
    stream: List[int],
    prior: float = 0.5,
    cost_act: float = 5.0,
    cost_wait: float = 1.0,
    cost_wrong: float = 50.0,
) -> Dict[str, Any]:
    """Return evidence-sufficiency decision for an observed recall / performance stream.

    Args:
        stream:     list of int values in {1..5}, one per probe/test round
        prior:      float prior probability in (0, 1)
        cost_act:   float cost of acting (false alarm penalty)
        cost_wait:  float cost of waiting when action was needed
        cost_wrong: float cost of acting when recovered (catastrophic wrong act)

    Returns:
        dict with keys:
            decision       — "act" or "wait"
            sufficient     — bool, True iff decision == "act"
            n_obs          — int, number of observations
            log_lr         — float, cumulative log likelihood ratio including prior
            threshold_act  — float, A (upper SPRT boundary)
            threshold_wait — float, B (lower SPRT boundary)
    """
    alpha = cost_wait / (cost_act + cost_wait)
    beta = cost_act / (cost_wrong + cost_act)
    A = (1.0 - beta) / alpha
    B = beta / (1.0 - alpha)

    # Prior log-odds initialization (Fix for R3)
    p = min(0.999, max(0.001, float(prior)))
    prior_log_odds = math.log(p / (1.0 - p))

    n_obs = len(stream)

    if n_obs < 2:
        single_term = 0.0
        if n_obs == 1:
            r0 = max(1, min(5, stream[0]))
            single_term = math.log(r0 / (6.0 - r0))
        return {
            "decision": "wait",
            "sufficient": False,
            "n_obs": n_obs,
            "log_lr": prior_log_odds + single_term,
            "threshold_act": A,
            "threshold_wait": B,
        }

    # Likelihood ratio per observation: r / (6 - r) (Fix for R2)
    obs_log_lr = sum(math.log(max(1, min(5, r)) / (6.0 - max(1, min(5, r)))) for r in stream)
    log_lr = prior_log_odds + obs_log_lr

    if log_lr >= math.log(A):
        decision = "act"
    else:
        decision = "wait"

    return {
        "decision": decision,
        "sufficient": decision == "act",
        "n_obs": n_obs,
        "log_lr": log_lr,
        "threshold_act": A,
        "threshold_wait": B,
    }


class SPRTAccumulator:
    """Accumulates sequential observations across multiple consolidation cycles."""

    def __init__(
        self,
        item_id: str,
        prior: float = 0.5,
        cost_act: float = 5.0,
        cost_wait: float = 1.0,
        cost_wrong: float = 50.0,
    ):
        self.item_id = item_id
        self.prior = prior
        self.cost_act = cost_act
        self.cost_wait = cost_wait
        self.cost_wrong = cost_wrong
        self.observations: List[int] = []
        self.cycle_count: int = 0

    def record_run(self, new_obs: List[int]) -> Dict[str, Any]:
        """Append observations from a run and return updated SPRT decision."""
        self.observations.extend(new_obs)
        self.cycle_count += 1
        res = sufficient(
            self.observations,
            prior=self.prior,
            cost_act=self.cost_act,
            cost_wait=self.cost_wait,
            cost_wrong=self.cost_wrong,
        )
        res["item_id"] = self.item_id
        res["cycle_count"] = self.cycle_count
        res["total_observations"] = len(self.observations)
        return res
