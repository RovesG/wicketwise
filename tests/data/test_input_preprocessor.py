# Purpose: Tests for the input preprocessor function.
# Author: Shamus Rae, Last Modified: 2024-07-30

import numpy as np
import pytest

from crickformers.data.data_schema import (
    CurrentBallFeatures,
    RecentBallHistoryEntry,
    GNNEmbeddings,
    VideoSignals,
)
from crickformers.data.input_preprocessor import prepare_model_inputs

@pytest.fixture
def sample_current_ball_features():
    """Provides a sample CurrentBallFeatures object for tests."""
    return CurrentBallFeatures(
        match_id="12345", match_date="2024-07-30", match_year=2024,
        competition_name="T20 Blast", venue_id="1", venue_name="Lord's",
        home_team_id="H1", home_team_name="Home", away_team_id="A1", away_team_name="Away",
        batting_team_name="Home", innings=1, winner="Home", toss_winner="Home",
        over=5, delivery=3, ball="5.3", batter_id="B1", bowler_id="Bo1", nonstriker_id="B2",
        bowler_style="Right-arm fast", batting_style="Right-hand bat",
        runs=4, extras=0, noball=0, wide=0, byes=0, legbyes=0, dot=0, four=1, six=0, single=0,
        balls_remaining=87, batsman_runs_ball=4, target_score=190.0,
        batsman_runs=25, bowler_runs_ball=10, batsman_balls=15, batter_orden=3
    )

@pytest.fixture
def sample_ball_history():
    """Provides a sample list of 5 recent ball history entries."""
    return [
        RecentBallHistoryEntry(runs_scored=1, bowler_type="fast", shot_type="drive", dismissal=False),
        RecentBallHistoryEntry(runs_scored=0, bowler_type="fast", shot_type="block", dismissal=False),
        RecentBallHistoryEntry(runs_scored=4, bowler_type="spin", shot_type="sweep", dismissal=False),
        RecentBallHistoryEntry(runs_scored=6, bowler_type="fast", shot_type="pull", dismissal=False, head_stability=0.9),
        RecentBallHistoryEntry(runs_scored=0, bowler_type="spin", shot_type="defence", dismissal=True),
    ]

@pytest.fixture
def sample_gnn_embeddings():
    """Provides sample GNN embeddings."""
    return GNNEmbeddings(
        batter_embedding=list(np.random.rand(128)),
        bowler_type_embedding=list(np.random.rand(128)),
        venue_embedding=list(np.random.rand(64)),
        batter_bowler_edge_embedding=list(np.random.rand(64)),
    )

@pytest.fixture
def sample_video_signals():
    """Provides sample video signals."""
    return VideoSignals(
        headStability=0.95, backlift=0.8, footworkDirection="forward", shotCommitment=0.9,
        shotType="pull", runningSpeed=5.5, nonStrikerReaction=0.5,
        releasePointConsistency=0.98, bowlingArmPath=0.85, paceProxyFrames=2.1,
        creasePosition=0.5, frontArmStability=0.9, followThroughMomentum=0.7,
        reactionTime=0.2, closingSpeed=8.0, pathEfficiency=0.9,
        interceptionType="clean", handsTechnique="good", getToThrowTime=1.2, throwAccuracy=0.9
    )

def test_prepare_model_inputs_with_video(
    sample_current_ball_features, sample_ball_history, sample_gnn_embeddings, sample_video_signals
):
    """Tests the preprocessor with all inputs present."""
    inputs = prepare_model_inputs(
        current_ball_features=sample_current_ball_features,
        recent_ball_history=sample_ball_history,
        gnn_embeddings=sample_gnn_embeddings,
        video_signals=sample_video_signals
    )

    assert isinstance(inputs, dict)
    assert "numeric_ball_features" in inputs
    assert "categorical_ball_features" in inputs
    assert "ball_history" in inputs
    assert "video_features" in inputs
    assert "video_mask" in inputs
    assert "gnn_embeddings" in inputs

    assert inputs["numeric_ball_features"].shape == (7,)
    assert inputs["categorical_ball_features"].shape == (4,)
    assert inputs["ball_history"].shape == (5, 6)
    assert inputs["video_features"].shape == (11,)
    assert inputs["video_mask"].item() == 1.0
    assert inputs["gnn_embeddings"].shape == (128 + 128 + 64 + 64,)

def test_prepare_model_inputs_without_video(
    sample_current_ball_features, sample_ball_history, sample_gnn_embeddings
):
    """Tests the preprocessor when video signals are missing."""
    inputs = prepare_model_inputs(
        current_ball_features=sample_current_ball_features,
        recent_ball_history=sample_ball_history,
        gnn_embeddings=sample_gnn_embeddings,
        video_signals=None
    )

    assert isinstance(inputs, dict)
    assert inputs["video_features"].shape == (11,)
    assert np.all(inputs["video_features"] == 0)
    assert inputs["video_mask"].item() == 0.0

def test_history_padding_and_truncation(sample_current_ball_features, sample_gnn_embeddings):
    """Tests that ball history is correctly padded or truncated to 5 entries."""
    # Test with fewer than 5 entries (padding)
    short_history = [RecentBallHistoryEntry.padding_entry()] * 3
    inputs = prepare_model_inputs(sample_current_ball_features, short_history, sample_gnn_embeddings)
    assert inputs["ball_history"].shape == (5, 6)

    # Test with more than 5 entries (truncation)
    long_history = [RecentBallHistoryEntry.padding_entry()] * 7
    inputs = prepare_model_inputs(sample_current_ball_features, long_history, sample_gnn_embeddings)
    assert inputs["ball_history"].shape == (5, 6) 