# Purpose: Data schemas for CSV cricket data format
# Author: Assistant, Last Modified: 2024

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import numpy as np

class CurrentBallFeatures(BaseModel):
    """
    Current ball features extracted from CSV data format.
    Matches the structure used in our CSV adapter.
    """
    # Match identification
    match_id: str
    competition_name: str
    venue: str
    venue_city: str
    venue_country: str
    
    # Ball identification
    innings: int
    over: float
    ball_in_over: int
    innings_ball: int
    
    # Players
    batter_name: str
    batter_id: str
    bowler_name: str
    bowler_id: str
    
    # Ball outcome
    runs_scored: int
    extras: int
    is_wicket: bool
    
    # Team state
    team_score: int
    team_wickets: int
    
    # Team names
    batting_team: str
    bowling_team: str
    
    # Player attributes
    batter_hand: str
    bowler_type: str
    
    # Ball tracking (from nvplay data)
    field_x: float = 0.0
    field_y: float = 0.0
    pitch_x: float = 0.0
    pitch_y: float = 0.0
    
    # Match situation
    powerplay: float = 0.0
    run_rate: float = 0.0
    req_run_rate: float = 0.0

class RecentBallHistoryEntry(BaseModel):
    """
    Single entry in recent ball history.
    """
    runs_scored: int
    extras: int
    is_wicket: bool
    batter_name: str
    bowler_name: str
    batter_hand: str
    bowler_type: str

class VideoSignals(BaseModel):
    """
    Video-derived signals for the current ball.
    For CSV data, these are mostly mock/placeholder values.
    """
    ball_tracking_confidence: float
    player_detection_confidence: float
    scene_classification: str
    motion_vectors: np.ndarray
    optical_flow: np.ndarray
    
    class Config:
        arbitrary_types_allowed = True

class GNNEmbeddings(BaseModel):
    """
    GNN embeddings for players and venue.
    """
    batter_embedding: np.ndarray
    bowler_embedding: np.ndarray
    venue_embedding: np.ndarray
    edge_embeddings: np.ndarray
    
    class Config:
        arbitrary_types_allowed = True

class MarketOdds(BaseModel):
    """
    Market odds from decimal data.
    """
    win_probability: float
    total_runs_over: float = 0.0
    total_runs_under: float = 0.0
    next_wicket_over: float = 0.0
    next_wicket_under: float = 0.0
    match_odds_home: float = 0.0
    match_odds_away: float = 0.0 