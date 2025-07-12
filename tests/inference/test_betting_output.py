# Purpose: Tests for the betting output decision logic.
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
import math
from crickformers.inference.betting_output import format_shadow_bet_decision

def test_value_bet_decision():
    """Tests a clear scenario where a value bet should be flagged."""
    model_outputs = {"win_probability": 0.60} # 60%
    betfair_odds = 2.0  # Implies 50% probability
    
    decision = format_shadow_bet_decision(model_outputs, betfair_odds, value_threshold=0.05)
    
    assert decision["decision"] == "value_bet"
    assert decision["delta"] == 0.10
    assert decision["reason"] == "Model probability exceeds market implied probability by value threshold."

def test_no_bet_decision_insufficient_value():
    """Tests a scenario where the delta is positive but below the threshold."""
    model_outputs = {"win_probability": 0.52} # 52%
    betfair_odds = 2.0  # Implies 50% probability
    
    decision = format_shadow_bet_decision(model_outputs, betfair_odds, value_threshold=0.05)
    
    assert decision["decision"] == "no_bet"
    assert decision["delta"] == 0.02
    assert decision["reason"] == "No significant value detected."

def test_no_bet_decision_negative_value():
    """Tests a scenario where the model has a lower probability than the market."""
    model_outputs = {"win_probability": 0.45} # 45%
    betfair_odds = 2.0  # Implies 50% probability
    
    decision = format_shadow_bet_decision(model_outputs, betfair_odds, value_threshold=0.05)
    
    assert decision["decision"] == "no_bet"
    assert decision["delta"] == -0.05

def test_risk_alert_decision():
    """
    Tests a scenario where the model is highly confident but odds are unexpectedly long.
    """
    model_outputs = {"win_probability": 0.80} # 80% confident
    betfair_odds = 3.0 # Implies 33.3% probability, very long for 80% confidence
    
    decision = format_shadow_bet_decision(
        model_outputs, betfair_odds, confidence_threshold=0.75
    )
    
    assert decision["decision"] == "risk_alert"
    assert decision["reason"] == "Model is highly confident but odds are unexpectedly long."

def test_stable_handling_of_null_odds():
    """Ensures the function handles None for betfair_odds without error."""
    model_outputs = {"win_probability": 0.60}
    
    decision = format_shadow_bet_decision(model_outputs, None)
    
    assert decision["decision"] == "no_bet"
    assert decision["reason"] == "Invalid model or market probability"
    assert decision["delta"] is None

def test_stable_handling_of_invalid_odds():
    """Ensures the function handles odds <= 1.0 or NaN without error."""
    model_outputs = {"win_probability": 0.60}
    
    # Test with odds of 1.0
    decision_one = format_shadow_bet_decision(model_outputs, 1.0)
    assert decision_one["decision"] == "no_bet"
    
    # Test with NaN odds
    decision_nan = format_shadow_bet_decision(model_outputs, math.nan)
    assert decision_nan["decision"] == "no_bet"

def test_stable_handling_of_missing_model_prob():
    """Ensures the function handles missing win_probability key."""
    model_outputs = {"other_output": 0.5} # Missing 'win_probability'
    betfair_odds = 2.0
    
    decision = format_shadow_bet_decision(model_outputs, betfair_odds)
    
    assert decision["decision"] == "no_bet"
    assert decision["reason"] == "Invalid model or market probability" 