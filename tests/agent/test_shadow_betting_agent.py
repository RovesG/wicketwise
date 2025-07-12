# Purpose: Tests for the ShadowBettingAgent class.
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
import math
from crickformers.agent.shadow_betting_agent import ShadowBettingAgent


@pytest.fixture
def agent():
    """Returns a ShadowBettingAgent with default thresholds."""
    return ShadowBettingAgent(value_threshold=0.1, risk_confidence_threshold=0.8)


def test_initialization_with_valid_thresholds():
    """Tests that the agent can be initialized with valid thresholds."""
    agent = ShadowBettingAgent(value_threshold=0.05, risk_confidence_threshold=0.75)
    assert agent.value_threshold == 0.05
    assert agent.risk_confidence_threshold == 0.75


def test_initialization_with_invalid_thresholds():
    """Tests that initialization fails with out-of-range thresholds."""
    with pytest.raises(ValueError):
        ShadowBettingAgent(value_threshold=1.5)
    with pytest.raises(ValueError):
        ShadowBettingAgent(risk_confidence_threshold=-0.5)


def test_decision_is_value_bet(agent):
    """Tests the 'value_bet' decision logic."""
    # Model prob (0.7) > Market prob (0.5) + Threshold (0.1)
    decision_log = agent.make_decision(win_probability=0.7, market_odds=2.0)
    assert decision_log["decision"] == "value_bet"
    assert "value_delta" in decision_log["derived_values"]
    assert decision_log["derived_values"]["value_delta"] == pytest.approx(0.2)


def test_decision_is_no_bet_insufficient_value(agent):
    """Tests 'no_bet' when the value edge is below the threshold."""
    # Model prob (0.55) < Market prob (0.5) + Threshold (0.1)
    decision_log = agent.make_decision(win_probability=0.55, market_odds=2.0)
    assert decision_log["decision"] == "no_bet"


def test_decision_is_no_bet_model_under_market(agent):
    """Tests 'no_bet' when the model is less confident than the market."""
    decision_log = agent.make_decision(win_probability=0.4, market_odds=2.0)
    assert decision_log["decision"] == "no_bet"


def test_decision_is_risk_alert(agent):
    """
    Tests the 'risk_alert' logic for high model confidence but long odds.
    """
    # High model confidence (0.9 > 0.8) and long odds (3.0 > 1/(0.8*0.5)=2.5)
    decision_log = agent.make_decision(win_probability=0.9, market_odds=3.0)
    assert decision_log["decision"] == "risk_alert"


def test_handling_of_missing_market_odds(agent):
    """Tests that 'no_bet' is returned when market_odds is None."""
    decision_log = agent.make_decision(win_probability=0.8, market_odds=None)
    assert decision_log["decision"] == "no_bet"
    assert "Invalid or missing market odds" in decision_log["reason"]


def test_handling_of_invalid_market_odds(agent):
    """Tests that 'no_bet' is returned for invalid odds (e.g., <= 1.0)."""
    decision_log = agent.make_decision(win_probability=0.8, market_odds=1.0)
    assert decision_log["decision"] == "no_bet"
    decision_log_nan = agent.make_decision(win_probability=0.8, market_odds=float('nan'))
    assert decision_log_nan["decision"] == "no_bet"


def test_audit_log_structure(agent):
    """Verifies that the audit log contains all expected keys."""
    decision_log = agent.make_decision(win_probability=0.6, market_odds=1.8)
    assert "decision" in decision_log
    assert "reason" in decision_log
    assert "inputs" in decision_log
    assert "derived_values" in decision_log
    assert "model_win_probability" in decision_log["inputs"]
    assert "value_delta" in decision_log["derived_values"] 