# Purpose: Policy module for bet sizing and gating
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from __future__ import annotations

from typing import Optional


class BettingPolicy:
    def __init__(self, kelly_fraction: float = 0.5, cap: float = 0.02, uncertainty_threshold: float = 0.2):
        self.kelly_fraction = float(max(0.0, min(1.0, kelly_fraction)))
        self.cap = float(max(0.0, cap))
        self.uncertainty_threshold = float(max(0.0, min(1.0, uncertainty_threshold)))

    def fractional_kelly(self, prob: float, odds: float) -> float:
        prob = float(max(0.0, min(1.0, prob)))
        odds = float(max(1e-6, odds))
        edge = prob * (odds - 1) - (1 - prob)
        if edge <= 0:
            return 0.0
        kelly = edge / (odds - 1)
        stake_fraction = max(0.0, min(self.cap, self.kelly_fraction * kelly))
        return stake_fraction

    def should_bet(self, prob: float, odds: float, uncertainty: Optional[float] = None) -> bool:
        if uncertainty is not None and uncertainty > self.uncertainty_threshold:
            return False
        return self.fractional_kelly(prob, odds) > 0
