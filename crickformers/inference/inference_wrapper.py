# Purpose: Provides a high-level wrapper for running live inference.
# Author: Shamus Rae, Last Modified: 2024-07-30

"""
This module contains the `run_inference` function, which serves as the main
entry point for making live predictions. It encapsulates the preprocessing,
model prediction, and output formatting steps.
"""

from typing import List, Optional, Dict, Any

import torch
import numpy as np
from torch import nn

from crickformers.data.data_schema import (
    CurrentBallFeatures,
    RecentBallHistoryEntry,
    GNNEmbeddings,
    VideoSignals,
)
from crickformers.data.input_preprocessor import prepare_model_inputs


def run_inference(
    model: nn.Module,
    current_ball_features: CurrentBallFeatures,
    recent_ball_history: List[RecentBallHistoryEntry],
    gnn_embeddings: GNNEmbeddings,
    video_signals: Optional[VideoSignals] = None,
) -> Dict[str, Any]:
    """
    Runs a forward pass to get predictions for a single data point.

    Args:
        model: A pretrained PyTorch model. The model is expected to return a
               dictionary of raw logits for each prediction head.
        current_ball_features: The structured features for the current ball.
        recent_ball_history: A list of the last 5 ball history entries.
        gnn_embeddings: The pretrained GNN embeddings.
        video_signals: Optional video-derived signals.

    Returns:
        A dictionary containing the formatted model predictions.
    """
    model.eval()

    # 1. Preprocess the raw inputs into a dictionary of NumPy arrays
    processed_inputs = prepare_model_inputs(
        current_ball_features=current_ball_features,
        recent_ball_history=recent_ball_history,
        gnn_embeddings=gnn_embeddings,
        video_signals=video_signals,
    )

    # 2. Convert NumPy arrays to PyTorch tensors and add a batch dimension
    tensor_inputs = {
        key: torch.from_numpy(value).unsqueeze(0)
        for key, value in processed_inputs.items()
    }

    # 3. Run the model prediction
    with torch.no_grad():
        # The model is expected to accept the dictionary of tensors
        # and return a dictionary of raw output logits.
        raw_outputs = model(tensor_inputs)

    # 4. Format the output from the three prediction heads
    # next_ball_outcome: Apply softmax to get a probability distribution
    outcome_probs = torch.softmax(raw_outputs["next_ball_outcome"], dim=-1).squeeze().tolist()

    # win_probability: Apply sigmoid to scale the logit to a 0-1 probability
    win_prob = torch.sigmoid(raw_outputs["win_probability"]).squeeze().item()

    # odds_mispricing: Apply sigmoid for probability or use a threshold for binary
    mispricing_prob = torch.sigmoid(raw_outputs["odds_mispricing"]).squeeze().item()
    mispricing_binary = 1 if mispricing_prob > 0.5 else 0

    return {
        "next_ball_outcome": outcome_probs,
        "win_probability": win_prob,
        "odds_mispricing": {
            "probability": mispricing_prob,
            "binary_prediction": mispricing_binary,
        },
    } 