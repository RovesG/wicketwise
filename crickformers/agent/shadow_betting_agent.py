# Purpose: Implements the logic for a shadow betting agent.
# Author: Shamus Rae, Last Modified: 2024-07-30

from typing import Any, Dict, Optional
import math


class ShadowBettingAgent:
    """
    An agent that makes betting decisions based on model predictions
    and market odds.
    """

    def __init__(
        self,
        value_threshold: float = 0.05,
        risk_confidence_threshold: float = 0.75,
    ):
        """
        Initializes the agent with configurable thresholds.

        Args:
            value_threshold: The minimum positive delta between model
                             probability and market-implied probability
                             to flag a "value_bet".
            risk_confidence_threshold: The model's confidence level above
                                       which a "risk_alert" may be triggered
                                       if odds are unexpectedly long.
        """
        if not 0 < value_threshold < 1:
            raise ValueError("value_threshold must be between 0 and 1.")
        if not 0 < risk_confidence_threshold < 1:
            raise ValueError("risk_confidence_threshold must be between 0 and 1.")

        self.value_threshold = value_threshold
        self.risk_confidence_threshold = risk_confidence_threshold

    def _odds_to_probability(self, decimal_odds: Optional[float]) -> Optional[float]:
        """Converts decimal odds to an implied probability."""
        if decimal_odds is None or decimal_odds <= 1.0 or math.isnan(decimal_odds):
            return None
        return 1.0 / decimal_odds

    def make_decision(
        self,
        win_probability: float,
        market_odds: Optional[float],
        odds_mispricing_prob: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Applies logic to decide on a bet.

        Args:
            win_probability: The model's predicted win probability.
            market_odds: The current decimal odds from the market.
            odds_mispricing_prob: The model's predicted probability of a
                                  mispricing event.

        Returns:
            A dictionary containing the decision, the reason, and an
            audit trail of the inputs.
        """
        market_implied_prob = self._odds_to_probability(market_odds)

        if market_implied_prob is None:
            decision = "no_bet"
            reason = "Invalid or missing market odds."
            delta = None
        else:
            delta = win_probability - market_implied_prob
            # Risk Alert: High model confidence but long odds
            if (
                win_probability > self.risk_confidence_threshold
                and market_odds > (1 / (self.risk_confidence_threshold * 0.5))
            ):
                decision = "risk_alert"
                reason = "High model confidence with unexpectedly long odds."
            # Value Bet: Model edge exceeds the threshold
            elif delta > self.value_threshold:
                decision = "value_bet"
                reason = "Model probability exceeds market implied probability by value threshold."
            # No Bet: Not enough value
            else:
                decision = "no_bet"
                reason = "No significant value detected."

        log = {
            "decision": decision,
            "reason": reason,
            "inputs": {
                "model_win_probability": win_probability,
                "market_odds": market_odds,
                "market_implied_probability": market_implied_prob,
                "odds_mispricing_prob": odds_mispricing_prob,
            },
            "derived_values": {
                "value_delta": delta,
                "value_threshold": self.value_threshold,
                "risk_confidence_threshold": self.risk_confidence_threshold,
            },
        }

        return log 