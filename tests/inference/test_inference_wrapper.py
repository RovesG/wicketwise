# Purpose: Tests for the inference wrapper.
# Author: Shamus Rae, Last Modified: 2024-07-30

import torch
import pytest
from torch import nn
from typing import Dict

from crickformers.data.data_schema import (
    CurrentBallFeatures,
    RecentBallHistoryEntry,
    GNNEmbeddings,
)
from crickformers.inference.inference_wrapper import run_inference

# --- Mock Model and Fixtures ---

class MockCrickformersModel(nn.Module):
    """
    A mock model that simulates the output of the actual Crickformers model.
    It returns a dictionary of raw logits for the three prediction heads.
    """
    def __init__(self, outcome_classes: int = 6):
        super().__init__()
        self.outcome_classes = outcome_classes
        # These layers are just for show, the forward method returns fixed logits.
        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of predefined raw logits.
        Args:
            inputs: A dictionary of input tensors (not used in this mock).
        Returns:
            A dictionary of raw output logits.
        """
        batch_size = inputs["numeric_ball_features"].shape[0]
        return {
            "next_ball_outcome": torch.randn(batch_size, self.outcome_classes),
            "win_probability": torch.randn(batch_size, 1),
            "odds_mispricing": torch.randn(batch_size, 1),
        }

@pytest.fixture
def mock_model():
    """Provides a mock Crickformers model for testing."""
    return MockCrickformersModel()

@pytest.fixture
def sample_data_point():
    """Provides a single, complete data point for inference testing."""
    current_ball = CurrentBallFeatures(
        match_id="1", match_date="2024-01-01", match_year=2024,
        competition_name="T20 Blast", venue_id="1", venue_name="Lord's",
        home_team_id="H", home_team_name="Home", away_team_id="A", away_team_name="Away",
        batting_team_name="Home", innings=1, winner="None", toss_winner="Home",
        over=1, delivery=1, ball="1.1", batter_id="B1", bowler_id="P1", nonstriker_id="B2",
        bowler_style="Right-arm fast", batting_style="Right-hand bat", runs=1, extras=0, noball=0,
        wide=0, byes=0, legbyes=0, dot=0, four=0, six=0, single=1, balls_remaining=119,
        batsman_runs_ball=1, target_score=180.0, batsman_runs=1, bowler_runs_ball=1,
        batsman_balls=1, batter_orden=1
    )
    history = [RecentBallHistoryEntry.padding_entry()] * 5
    embeddings = GNNEmbeddings(
        batter_embedding=[0.1] * 128,
        bowler_type_embedding=[0.2] * 128,
        venue_embedding=[0.3] * 64
    )
    return current_ball, history, embeddings

# --- Tests ---

def test_run_inference_output_structure(mock_model, sample_data_point):
    """
    Tests that the inference wrapper returns a dictionary with the correct keys.
    """
    current_ball, history, embeddings = sample_data_point
    predictions = run_inference(mock_model, current_ball, history, embeddings)

    assert "next_ball_outcome" in predictions
    assert "win_probability" in predictions
    assert "odds_mispricing" in predictions
    assert "probability" in predictions["odds_mispricing"]
    assert "binary_prediction" in predictions["odds_mispricing"]

def test_run_inference_output_values(mock_model, sample_data_point):
    """
    Tests that the output values are correctly formatted.
    """
    current_ball, history, embeddings = sample_data_point
    predictions = run_inference(mock_model, current_ball, history, embeddings)

    # Test next_ball_outcome (softmax)
    outcome_probs = predictions["next_ball_outcome"]
    assert isinstance(outcome_probs, list)
    assert len(outcome_probs) == mock_model.outcome_classes
    assert abs(sum(outcome_probs) - 1.0) < 1e-6 # Probabilities should sum to 1

    # Test win_probability (sigmoid)
    win_prob = predictions["win_probability"]
    assert isinstance(win_prob, float)
    assert 0.0 <= win_prob <= 1.0

    # Test odds_mispricing (sigmoid + binary)
    mispricing = predictions["odds_mispricing"]
    assert isinstance(mispricing["probability"], float)
    assert 0.0 <= mispricing["probability"] <= 1.0
    assert mispricing["binary_prediction"] in [0, 1] 