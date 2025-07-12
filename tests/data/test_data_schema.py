# Purpose: Tests for the data schema Pydantic models.
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
from crickformers.data.data_schema import (
    CurrentBallFeatures,
    RecentBallHistoryEntry,
    GNNEmbeddings,
    VideoSignals,
)

def test_current_ball_features_instantiation():
    """Tests that the CurrentBallFeatures model can be instantiated."""
    features = CurrentBallFeatures(
        match_id="12345",
        match_date="2024-07-30",
        match_year=2024,
        competition_name="T20 Blast",
        venue_id="1",
        venue_name="Lord's",
        home_team_id="H1",
        home_team_name="Home Team",
        away_team_id="A1",
        away_team_name="Away Team",
        batting_team_name="Home Team",
        innings=1,
        winner="Home Team",
        toss_winner="Home Team",
        over=1,
        delivery=1,
        ball="1.1",
        batter_id="B1",
        bowler_id="Bo1",
        nonstriker_id="B2",
        bowler_style="Right-arm fast",
        batting_style="Right-hand bat",
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
        target_score=180.0,
        batsman_runs=1,
        bowler_runs_ball=1,
        batsman_balls=1,
        batter_orden=1,
    )
    assert features.match_id == "12345"
    assert features.over == 1

def test_recent_ball_history_entry_instantiation():
    """Tests that RecentBallHistoryEntry can be instantiated."""
    entry = RecentBallHistoryEntry(
        runs_scored=4,
        bowler_type="Left-arm spin",
        shot_type="Cover drive",
        dismissal=False,
        head_stability=0.9,
    )
    assert entry.runs_scored == 4
    assert entry.dismissal is False

def test_recent_ball_history_padding():
    """Tests the padding_entry classmethod."""
    padding = RecentBallHistoryEntry.padding_entry()
    assert padding.runs_scored == 0
    assert padding.bowler_type == "none"
    assert padding.dismissal is False
    assert padding.head_stability is None

def test_gnn_embeddings_instantiation():
    """Tests that GNNEmbeddings can be instantiated with and without data."""
    # Test with data
    embeddings = GNNEmbeddings(
        batter_embedding=[0.1] * 128,
        bowler_type_embedding=[0.2] * 128,
        venue_embedding=[0.3] * 64,
    )
    assert len(embeddings.batter_embedding) == 128
    assert len(embeddings.bowler_type_embedding) == 128
    assert len(embeddings.venue_embedding) == 64
    assert embeddings.batter_bowler_edge_embedding == []

    # Test default factory
    default_embeddings = GNNEmbeddings()
    assert default_embeddings.batter_embedding == []
    assert default_embeddings.bowler_type_embedding == []
    assert default_embeddings.venue_embedding == []

def test_video_signals_instantiation():
    """Tests that VideoSignals can be instantiated."""
    signals = VideoSignals(
        headStability=0.95,
        backlift=0.8,
        footworkDirection="forward",
        shotCommitment=0.9,
        shotType="pull",
        runningSpeed=5.5,
        nonStrikerReaction=0.5,
        releasePointConsistency=0.98,
        bowlingArmPath=0.85,
        paceProxyFrames=2.1,
        creasePosition=0.5,
        frontArmStability=0.9,
        followThroughMomentum=0.7,
        reactionTime=0.2,
        closingSpeed=8.0,
        pathEfficiency=0.9,
        interceptionType="clean",
        handsTechnique="good",
        getToThrowTime=1.2,
        throwAccuracy=0.9,
    )
    assert signals.headStability == 0.95
    assert signals.throwAccuracy == 0.9 