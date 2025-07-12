# Purpose: Tests for the live inference pipeline.
# Author: Shamus Rae, Last Modified: 2024-07-30

from unittest.mock import MagicMock, patch

import pytest
import torch
from crickformers.data.data_schema import (
    BettingDecision,
    CurrentBallFeatures,
    GNNEmbeddings,
    MarketOdds,
    ModelOutput,
    RecentBallHistoryEntry,
    VideoSignals,
)
from crickformers.inference.live_pipeline import run_live_pipeline


@pytest.fixture
def mock_model():
    """Mocks the Crickformer model."""
    model = MagicMock(spec=torch.nn.Module)
    return model


@pytest.fixture
def mock_current_ball_features():
    """Provides sample current ball features."""
    return CurrentBallFeatures(
        match_id="1",
        match_date="2024-01-01",
        match_year=2024,
        competition_name="Test",
        venue_id="1",
        venue_name="Test Venue",
        home_team_id="1",
        home_team_name="A",
        away_team_id="2",
        away_team_name="B",
        batting_team_name="A",
        innings=1,
        winner="A",
        toss_winner="A",
        over=1,
        delivery=1,
        ball="1.1",
        batter_id="1",
        bowler_id="1",
        nonstriker_id="2",
        bowler_style="fast",
        batting_style="rhb",
        runs=1,
        extras=0,
        noball=0,
        wide=0,
        byes=0,
        legbyes=0,
        dot=0,
        four=0,
        six=0,
        single=1,
        balls_remaining=119,
        batsman_runs_ball=1,
        target_score=200,
        batsman_runs=1,
        bowler_runs_ball=1,
        batsman_balls=1,
        batter_orden=1,
    )


@pytest.fixture
def mock_recent_ball_history():
    """Provides a list of sample historical ball events."""
    return [RecentBallHistoryEntry.padding_entry()] * 5


@pytest.fixture
def mock_gnn_embeddings():
    """Mocks GNN embeddings."""
    return GNNEmbeddings(
        batter_embedding=[0.1] * 128,
        bowler_type_embedding=[0.2] * 128,
        venue_embedding=[0.3] * 64,
    )


@pytest.fixture
def mock_market_odds():
    """Provides sample market odds, keyed by team ID."""
    return MarketOdds(
        win_odds={"1": 1.8, "2": 2.2},
        next_ball_odds={"dot": 2.0, "one": 1.5, "four": 5.0},
    )


@pytest.fixture
def mock_video_signals():
    """Provides mock video signals."""
    return VideoSignals(
        headStability=0.9,
        backlift=0.8,
        footworkDirection="forward",
        shotCommitment=0.95,
        shotType="drive",
        runningSpeed=5.0,
        nonStrikerReaction=0.5,
        releasePointConsistency=0.9,
        bowlingArmPath=0.85,
        paceProxyFrames=10.0,
        creasePosition=0.5,
        frontArmStability=0.9,
        followThroughMomentum=0.8,
        reactionTime=0.2,
        closingSpeed=8.0,
        pathEfficiency=0.9,
        interceptionType="clean",
        handsTechnique="good",
        getToThrowTime=1.5,
        throwAccuracy=0.95,
    )


@patch("crickformers.inference.live_pipeline.format_shadow_bet_decision")
@patch("crickformers.inference.live_pipeline.run_inference")
def test_run_live_pipeline_end_to_end(
    mock_run_inference,
    mock_format_shadow_bet_decision,
    mock_model,
    mock_current_ball_features,
    mock_recent_ball_history,
    mock_gnn_embeddings,
    mock_market_odds,
    mock_video_signals,
):
    """Tests the full pipeline from input to betting decision."""
    mock_run_inference.return_value = {
        "win_probability": 0.6,
        "next_ball_outcome": [0.1, 0.5, 0.4],
    }
    mock_format_shadow_bet_decision.return_value = {
        "decision": "value_bet",
        "reason": "Model probability exceeds market implied probability.",
    }

    def mock_video_fetcher(ball_str):
        return mock_video_signals

    result = run_live_pipeline(
        model=mock_model,
        current_ball_features=mock_current_ball_features,
        recent_ball_history=mock_recent_ball_history,
        gnn_embeddings=mock_gnn_embeddings,
        market_odds=mock_market_odds,
        video_fetcher=mock_video_fetcher,
    )

    assert "predictions" in result
    assert "betting_decision" in result
    assert result["betting_decision"]["decision"] == "value_bet"
    mock_run_inference.assert_called_once()
    mock_format_shadow_bet_decision.assert_called_once()


@patch("crickformers.inference.live_pipeline.run_inference")
def test_pipeline_handles_missing_video(
    mock_run_inference,
    mock_model,
    mock_current_ball_features,
    mock_recent_ball_history,
    mock_gnn_embeddings,
    mock_market_odds,
):
    """Ensures the pipeline runs correctly when video signals are unavailable."""
    mock_run_inference.return_value = {
        "win_probability": 0.5,
        "next_ball_outcome": [0.2, 0.6, 0.2],
    }

    def no_video_fetcher(ball_str):
        return None

    run_live_pipeline(
        model=mock_model,
        current_ball_features=mock_current_ball_features,
        recent_ball_history=mock_recent_ball_history,
        gnn_embeddings=mock_gnn_embeddings,
        market_odds=mock_market_odds,
        video_fetcher=no_video_fetcher,
    )

    _, kwargs = mock_run_inference.call_args
    assert kwargs["video_signals"] is None 