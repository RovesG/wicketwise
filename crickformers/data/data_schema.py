# Purpose: Defines the data structures for Crickformers using Pydantic.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the Pydantic models that define the schema for all data
structures used in the Crickformers project.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class CurrentBallFeatures(BaseModel):
    """
    Represents the full set of structured features available for the current ball.
    This combines match-level context with ball-level specifics.
    """
    # Match-Level Features
    match_id: str
    match_date: str
    match_year: int
    competition_name: str
    venue_id: str
    venue_name: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    batting_team_name: str
    innings: int
    winner: str
    toss_winner: str

    # Ball-Level Features
    over: int
    delivery: int
    ball: str
    batter_id: str
    bowler_id: str
    nonstriker_id: str
    bowler_style: str
    batting_style: str
    runs: int
    extras: int
    noball: int
    wide: int
    byes: int
    legbyes: int
    dot: int
    four: int
    six: int
    single: int
    balls_remaining: int
    batsman_runs_ball: int
    target_score: float
    batsman_runs: int
    bowler_runs_ball: int
    batsman_balls: int
    batter_orden: int

class RecentBallHistoryEntry(BaseModel):
    """
    Represents the features of a single ball from the recent history (e.g., last 5 balls).
    Includes a default factory for creating padding entries.
    """
    runs_scored: int
    bowler_type: str
    shot_type: str
    dismissal: bool
    head_stability: Optional[float] = None
    shot_commitment: Optional[float] = None
    pace_proxy: Optional[float] = None

    @classmethod
    def padding_entry(cls):
        """Returns a default entry used for padding shorter sequences."""
        return cls(
            runs_scored=0,
            bowler_type="none",
            shot_type="none",
            dismissal=False,
        )

class GNNEmbeddings(BaseModel):
    """
    Container for the pretrained GNN embeddings.
    """
    # Expected length: 128
    batter_embedding: List[float] = Field(default_factory=list)
    # Expected length: 128
    bowler_type_embedding: List[float] = Field(default_factory=list)
    # Expected length: 64
    venue_embedding: List[float] = Field(default_factory=list)
    # Expected length: 64
    batter_bowler_edge_embedding: Optional[List[float]] = Field(default_factory=list)


class VideoSignals(BaseModel):
    """
    Represents all video-derived signals for the current ball,
    covering batsman, bowler, and fielder analysis.
    """
    # Batsman Analysis
    headStability: float
    backlift: float
    footworkDirection: str
    shotCommitment: float
    shotType: str
    runningSpeed: float
    nonStrikerReaction: float

    # Bowler Analysis
    releasePointConsistency: float
    bowlingArmPath: float
    paceProxyFrames: float
    creasePosition: float
    frontArmStability: float
    followThroughMomentum: float
    bounceBackAbility: Optional[float] = None

    # Fielder Analysis
    reactionTime: float
    closingSpeed: float
    pathEfficiency: float
    interceptionType: str
    handsTechnique: str
    getToThrowTime: float
    throwAccuracy: float 