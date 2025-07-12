# Purpose: Tests for the TacticalFeedbackAgent.
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
from crickformers.agent.tactical_agent import TacticalFeedbackAgent, TacticalFeedback


@pytest.fixture
def agent():
    """Returns a TacticalFeedbackAgent."""
    return TacticalFeedbackAgent()


@pytest.fixture
def mock_prediction_outputs():
    """Provides a sample dictionary of model prediction outputs."""
    return {
        "win_probability": 0.65,
        "next_ball_outcome": [0.1, 0.2, 0.05, 0.05, 0.4, 0.15, 0.05],
        "odds_mispricing": 0.12,
    }


@pytest.fixture
def mock_context_data():
    """Provides a sample dictionary of match context data."""
    return {
        "phase": "Middle Overs",
        "batter_id": "player_123",
        "bowler_type": "leg-spin",
        "recent_shot_types": ["drive", "sweep", "defence"],
    }


def test_prompt_formatting(agent, mock_prediction_outputs, mock_context_data):
    """Tests that the prompt is formatted correctly based on inputs."""
    prompt = agent._format_prompt(mock_prediction_outputs, mock_context_data)
    assert "**Phase:** Middle Overs" in prompt
    assert "**Batter ID:** player_123" in prompt
    assert "**Bowler Type:** leg-spin" in prompt
    assert "drive, sweep, defence" in prompt
    assert "**Win Probability:** 0.65" in prompt
    assert "**Odds Mispricing Signal:** 0.12" in prompt


def test_fallback_mock_response(agent, mock_prediction_outputs, mock_context_data):
    """
    Tests that the agent returns the expected mock feedback structure regardless
    of the specific inputs, as the LLM call is mocked.
    """
    feedback = agent.get_tactical_feedback(mock_prediction_outputs, mock_context_data)
    assert isinstance(feedback, TacticalFeedback)
    assert "deep square leg" in feedback.summary_text  # Check for content from mock response
    assert feedback.confidence_level == 0.88
    assert feedback.recommended_action == "Adjust deep field placement for pull shot."


def test_different_tactical_scenarios(agent):
    """
    Simulates different tactical scenarios to ensure the agent's logic
    (currently just prompt formatting) handles them correctly.
    """
    # Scenario 1: Powerplay with an aggressive batter
    powerplay_context = {
        "phase": "Powerplay",
        "batter_id": "player_aggressive",
        "bowler_type": "fast",
        "recent_shot_types": ["loft", "drive"],
    }
    powerplay_preds = {"win_probability": 0.8, "next_ball_outcome": [0.05, 0.1, 0.05, 0.0, 0.5, 0.3, 0.0]}
    
    prompt1 = agent._format_prompt(powerplay_preds, powerplay_context)
    assert "**Phase:** Powerplay" in prompt1
    assert "**Batter ID:** player_aggressive" in prompt1

    # Scenario 2: Death overs with a new batter
    death_overs_context = {
        "phase": "Death Overs",
        "batter_id": "player_new",
        "bowler_type": "yorker-specialist",
        "recent_shot_types": [],
    }
    death_overs_preds = {"win_probability": 0.4, "next_ball_outcome": [0.4, 0.3, 0.1, 0.0, 0.1, 0.0, 0.1]}

    prompt2 = agent._format_prompt(death_overs_preds, death_overs_context)
    assert "**Phase:** Death Overs" in prompt2
    assert "**Recent Shot Types:**" in prompt2 and "[]" not in prompt2
    
    # Check that feedback is still returned
    feedback = agent.get_tactical_feedback(death_overs_preds, death_overs_context)
    assert isinstance(feedback, TacticalFeedback) 