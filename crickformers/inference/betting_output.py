# Purpose: Formats model outputs into a shadow betting decision.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the logic for interpreting the model's predictions
in the context of betting market odds to produce a clear, actionable
"shadow bet" decision.
"""

from typing import Dict, Any, Optional
import math

def _odds_to_probability(decimal_odds: Optional[float]) -> Optional[float]:
    """Converts decimal odds to an implied probability."""
    if decimal_odds is None or decimal_odds <= 1.0 or math.isnan(decimal_odds):
        return None
    return 1.0 / decimal_odds


def format_shadow_bet_decision(
    model_outputs: Dict[str, Any],
    betfair_odds: Optional[float],
    value_threshold: float = 0.05,
    confidence_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Compares model win probability against market odds to flag a bet decision.

    Args:
        model_outputs: The dictionary of predictions from the inference wrapper.
                       Expected to contain a 'win_probability' key.
        betfair_odds: The current decimal odds from the Betfair market.
        value_threshold: The minimum positive delta required to flag a value bet.
        confidence_threshold: A model win_probability threshold to flag high-risk alerts.

    Returns:
        A dictionary containing the decision and supporting metrics.
    """
    model_prob = model_outputs.get("win_probability")
    implied_prob = _odds_to_probability(betfair_odds)

    if model_prob is None or implied_prob is None:
        return {
            "decision": "no_bet",
            "reason": "Invalid model or market probability",
            "model_probability": model_prob,
            "market_implied_probability": implied_prob,
            "delta": None,
            "confidence": model_prob
        }
        
    delta = model_prob - implied_prob

    # Risk Alert: Model is highly confident, but odds are very long.
    # This might indicate the model is missing crucial information (e.g., injury).
    if model_prob > confidence_threshold and betfair_odds > (1 / (confidence_threshold * 0.5)):
        decision = "risk_alert"
        reason = "Model is highly confident but odds are unexpectedly long."
    # Value Bet: Model probability exceeds market probability by the threshold.
    elif delta > value_threshold:
        decision = "value_bet"
        reason = "Model probability exceeds market implied probability by value threshold."
    # No Bet: Not enough value to warrant a bet.
    else:
        decision = "no_bet"
        reason = "No significant value detected."

    return {
        "decision": decision,
        "reason": reason,
        "model_probability": round(model_prob, 4),
        "market_implied_probability": round(implied_prob, 4),
        "delta": round(delta, 4),
        "confidence": round(model_prob, 4),
    } 