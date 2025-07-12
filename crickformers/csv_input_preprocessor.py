# Purpose: Input preprocessor for CSV cricket data format
# Author: Assistant, Last Modified: 2024

import logging
from typing import List, Optional, Dict
import numpy as np

from .csv_data_schema import (
    CurrentBallFeatures,
    RecentBallHistoryEntry,
    GNNEmbeddings,
    VideoSignals,
    MarketOdds
)

logger = logging.getLogger(__name__)

# Categorical mappings for encoding
CATEGORICAL_MAPPING = {
    "competition": {
        "Big Bash League": 0, "Pakistan Super League": 1, "IPL": 2, "T20 Blast": 3,
        "unknown": 99
    },
    "batter_hand": {
        "RHB": 0, "LHB": 1, "right-hand bat": 0, "left-hand bat": 1, "PADDING": 99
    },
    "bowler_type": {
        "RM": 0, "LM": 1, "RS": 2, "LS": 3, "Right-arm fast-medium": 0,
        "Left-arm fast-medium": 1, "Right-arm spin": 2, "Left-arm spin": 3,
        "PADDING": 99, "unknown": 99
    },
    "scene_classification": {
        "cricket_match": 0, "unknown": 1
    }
}

def _encode_categorical(feature_name: str, value: str) -> int:
    """Encode categorical value to integer"""
    mapping = CATEGORICAL_MAPPING.get(feature_name, {})
    encoded = mapping.get(value, mapping.get("unknown", 0))
    
    # Ensure the encoded value is within reasonable bounds
    max_values = {
        "competition": 99,
        "batter_hand": 99,
        "bowler_type": 99,
        "scene_classification": 99
    }
    
    max_val = max_values.get(feature_name, 99)
    return min(encoded, max_val)

def _normalize_numeric(value: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Simple normalization (would use real stats in production)"""
    return (value - mean) / (std + 1e-7)

def preprocess_ball_input(
    current_features: CurrentBallFeatures,
    recent_history: List[RecentBallHistoryEntry],
    video_signals: Optional[VideoSignals] = None,
    gnn_embeddings: Optional[GNNEmbeddings] = None,
    market_odds: Optional[MarketOdds] = None
) -> Dict[str, np.ndarray]:
    """
    Preprocess CSV cricket data into model input format.
    
    Args:
        current_features: Current ball features
        recent_history: List of recent ball history entries
        video_signals: Optional video signals
        gnn_embeddings: Optional GNN embeddings
        market_odds: Optional market odds
        
    Returns:
        Dictionary of preprocessed arrays ready for model input
    """
    
    # 1. Process current ball numeric features
    numeric_features = np.array([
        _normalize_numeric(current_features.over, 10.0, 5.0),
        _normalize_numeric(current_features.ball_in_over, 3.0, 1.5),
        _normalize_numeric(current_features.innings_ball, 60.0, 35.0),
        _normalize_numeric(current_features.runs_scored, 1.0, 1.5),
        _normalize_numeric(current_features.extras, 0.2, 0.5),
        _normalize_numeric(current_features.team_score, 100.0, 50.0),
        _normalize_numeric(current_features.team_wickets, 3.0, 2.5),
        _normalize_numeric(current_features.field_x, 150.0, 100.0),
        _normalize_numeric(current_features.field_y, 150.0, 100.0),
        _normalize_numeric(current_features.pitch_x, 0.0, 10.0),
        _normalize_numeric(current_features.pitch_y, 0.0, 10.0),
        _normalize_numeric(current_features.powerplay, 0.5, 0.5),
        _normalize_numeric(current_features.run_rate, 7.0, 2.0),
        _normalize_numeric(current_features.req_run_rate, 8.0, 3.0),
        1.0 if current_features.is_wicket else 0.0
    ], dtype=np.float32)
    
    # 2. Process current ball categorical features
    categorical_features = np.array([
        _encode_categorical("competition", current_features.competition_name),
        _encode_categorical("batter_hand", current_features.batter_hand),
        _encode_categorical("bowler_type", current_features.bowler_type),
        current_features.innings
    ], dtype=np.int32)
    
    # 3. Process ball history
    history_length = 5
    if len(recent_history) > history_length:
        recent_history = recent_history[:history_length]
    
    # Pad history if needed
    while len(recent_history) < history_length:
        recent_history.append(RecentBallHistoryEntry(
            runs_scored=0, extras=0, is_wicket=False,
            batter_name="PADDING", bowler_name="PADDING",
            batter_hand="PADDING", bowler_type="PADDING"
        ))
    
    history_vectors = []
    for entry in recent_history:
        hist_vec = np.array([
            _normalize_numeric(entry.runs_scored, 1.0, 1.5),
            _normalize_numeric(entry.extras, 0.2, 0.5),
            1.0 if entry.is_wicket else 0.0,
            _encode_categorical("batter_hand", entry.batter_hand),
            _encode_categorical("bowler_type", entry.bowler_type),
            0.0  # Padding feature to make it 6-dimensional
        ], dtype=np.float32)
        history_vectors.append(hist_vec)
    
    ball_history = np.stack(history_vectors)  # Shape: (5, 6)
    
    # 4. Process video signals
    if video_signals is not None:
        video_numeric = np.array([
            video_signals.ball_tracking_confidence,
            video_signals.player_detection_confidence
        ], dtype=np.float32)
        
        video_categorical = np.array([
            _encode_categorical("scene_classification", video_signals.scene_classification)
        ], dtype=np.int32)
        
        # Flatten motion vectors and optical flow
        motion_flat = video_signals.motion_vectors.flatten()
        optical_flat = video_signals.optical_flow.flatten()
        
        # Combine all video features
        video_features = np.concatenate([
            video_numeric,
            video_categorical.astype(np.float32),
            motion_flat,
            optical_flat
        ])
        video_mask = np.array([1.0], dtype=np.float32)
    else:
        # Default video feature size: 2 numeric + 1 categorical + 32 motion + 64 optical = 99
        video_features = np.zeros(99, dtype=np.float32)
        video_mask = np.array([0.0], dtype=np.float32)
    
    # 5. Process GNN embeddings
    if gnn_embeddings is not None:
        gnn_vector = np.concatenate([
            gnn_embeddings.batter_embedding,
            gnn_embeddings.bowler_embedding,
            gnn_embeddings.venue_embedding
        ])
    else:
        # Default embedding sizes: 128 + 128 + 64 = 320
        gnn_vector = np.zeros(320, dtype=np.float32)
    
    # 6. Process market odds
    if market_odds is not None:
        odds_features = np.array([
            market_odds.win_probability,
            market_odds.total_runs_over,
            market_odds.total_runs_under,
            market_odds.next_wicket_over,
            market_odds.next_wicket_under,
            market_odds.match_odds_home,
            market_odds.match_odds_away
        ], dtype=np.float32)
        odds_mask = np.array([1.0], dtype=np.float32)
    else:
        odds_features = np.zeros(7, dtype=np.float32)
        odds_mask = np.array([0.0], dtype=np.float32)
    
    # 7. Assemble final input dictionary
    return {
        "numeric_ball_features": numeric_features,
        "categorical_ball_features": categorical_features,
        "ball_history": ball_history,
        "video_features": video_features,
        "video_mask": video_mask,
        "gnn_embeddings": gnn_vector,
        "market_odds": odds_features,
        "market_odds_mask": odds_mask
    } 