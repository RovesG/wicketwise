# Purpose: Prepares raw data objects for model consumption.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains functions to transform the raw Pydantic data schemas
into numerical tensors suitable for input into a deep learning model.
This includes normalization, encoding, and assembly into a single batch.
"""

import logging
from typing import List, Optional, Dict

import numpy as np

from .data_schema import (
    CurrentBallFeatures,
    RecentBallHistoryEntry,
    GNNEmbeddings,
    VideoSignals,
)

logger = logging.getLogger(__name__)

# In a real system, these would be loaded from a fitted preprocessor
# or a vocabulary file.
CATEGORICAL_MAPPING = {
    "competition_name": {"T20 Blast": 0, "IPL": 1},
    "venue_name": {"Lord's": 0, "The Oval": 1},
    "bowler_style": {"Right-arm fast": 0, "Left-arm spin": 1},
    "batting_style": {"Right-hand bat": 0, "Left-hand bat": 1},
    "shot_type": {"Cover drive": 0, "pull": 1, "none": 2},
    "bowler_type": {"fast": 0, "spin": 1, "none": 2},
    "footworkDirection": {"forward": 0, "backward": 1, "none": 2},
    "interceptionType": {"clean": 0, "fumble": 1, "none": 2},
    "handsTechnique": {"good": 0, "poor": 1, "none": 2},
}

# Placeholder for normalization constants (mean, std).
# These should be computed from the training dataset.
NORMALIZATION_CONSTANTS = {
    "over": (10.0, 5.0),
    "delivery": (3.5, 1.5),
    "runs": (1.5, 2.0),
    "balls_remaining": (60, 35),
    "target_score": (180, 30),
    # ... add other numeric features
}


def _normalize(feature_name: str, value: float) -> float:
    if feature_name in NORMALIZATION_CONSTANTS:
        mean, std = NORMALIZATION_CONSTANTS[feature_name]
        return (value - mean) / (std + 1e-7)
    return value


def _encode(feature_name: str, value: str) -> int:
    return CATEGORICAL_MAPPING.get(feature_name, {}).get(value, 0)


def prepare_model_inputs(
    current_ball_features: CurrentBallFeatures,
    recent_ball_history: List[RecentBallHistoryEntry],
    gnn_embeddings: GNNEmbeddings,
    video_signals: Optional[VideoSignals] = None,
) -> Dict[str, np.ndarray]:
    """
    Prepares a single data point for model inference or training.

    Args:
        current_ball_features: The structured features for the current ball.
        recent_ball_history: A list of the last 5 ball history entries.
        gnn_embeddings: The pretrained GNN embeddings for batter, bowler, etc.
        video_signals: Optional video-derived signals for the current ball.

    Returns:
        A dictionary containing numpy arrays ready for the model.
    """
    # 1. Process Current Ball Features
    numeric_ball_features = np.array([
        _normalize("over", current_ball_features.over),
        _normalize("delivery", current_ball_features.delivery),
        _normalize("runs", current_ball_features.runs),
        _normalize("balls_remaining", current_ball_features.balls_remaining),
        _normalize("target_score", current_ball_features.target_score),
        current_ball_features.batsman_runs,
        current_ball_features.batsman_balls,
    ], dtype=np.float32)

    categorical_ball_features = np.array([
        _encode("competition_name", current_ball_features.competition_name),
        _encode("venue_name", current_ball_features.venue_name),
        _encode("bowler_style", current_ball_features.bowler_style),
        _encode("batting_style", current_ball_features.batting_style),
    ], dtype=np.int32)

    # 2. Process Recent Ball History
    if len(recent_ball_history) != 5:
        logger.warning(
            f"Expected 5 recent ball entries, but got {len(recent_ball_history)}. "
            "Padding/truncating."
        )
        history = recent_ball_history[:5]
        while len(history) < 5:
            history.append(RecentBallHistoryEntry.padding_entry())
    else:
        history = recent_ball_history

    history_vectors = [
        np.array([
            entry.runs_scored,
            1.0 if entry.dismissal else 0.0,
            _encode("bowler_type", entry.bowler_type),
            _encode("shot_type", entry.shot_type),
            entry.head_stability or 0.0,
            entry.shot_commitment or 0.0,
        ], dtype=np.float32)
        for entry in history
    ]
    history_tensor = np.stack(history_vectors)

    # 3. Process Video Signals
    if video_signals:
        video_numeric = np.array([
            video_signals.headStability,
            video_signals.backlift,
            video_signals.shotCommitment,
            video_signals.runningSpeed,
            video_signals.releasePointConsistency,
            video_signals.paceProxyFrames,
            video_signals.reactionTime,
            video_signals.closingSpeed,
        ], dtype=np.float32)
        video_categorical = np.array([
            _encode("footworkDirection", video_signals.footworkDirection),
            _encode("interceptionType", video_signals.interceptionType),
            _encode("handsTechnique", video_signals.handsTechnique),
        ], dtype=np.int32)
        video_features = np.concatenate([video_numeric, video_categorical.astype(np.float32)])
        video_mask = np.array([1.0], dtype=np.float32)
    else:
        logger.info("No video signals provided. Using zeros and mask.")
        video_features = np.zeros(11, dtype=np.float32) # Match the size above
        video_mask = np.array([0.0], dtype=np.float32)

    # 4. Process GNN Embeddings
    gnn_vector = np.concatenate([
        np.array(gnn_embeddings.batter_embedding, dtype=np.float32),
        np.array(gnn_embeddings.bowler_type_embedding, dtype=np.float32),
        np.array(gnn_embeddings.venue_embedding, dtype=np.float32),
        np.array(gnn_embeddings.batter_bowler_edge_embedding or [], dtype=np.float32),
    ])

    # 5. Assemble final input dictionary
    return {
        "numeric_ball_features": numeric_ball_features,
        "categorical_ball_features": categorical_ball_features,
        "ball_history": history_tensor,
        "video_features": video_features,
        "video_mask": video_mask,
        "gnn_embeddings": gnn_vector,
    } 