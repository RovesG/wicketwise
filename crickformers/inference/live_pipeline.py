# Purpose: Orchestrates the real-time inference pipeline for a single ball event.
# Author: Shamus Rae, Last Modified: 2024-07-30

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from crickformers.data.data_schema import (
    CurrentBallFeatures,
    GNNEmbeddings,
    MarketOdds,
    RecentBallHistoryEntry,
    VideoSignals,
)
from crickformers.inference.betting_output import format_shadow_bet_decision
from crickformers.inference.inference_wrapper import run_inference

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_live_pipeline(
    model: torch.nn.Module,
    current_ball_features: CurrentBallFeatures,
    recent_ball_history: List[RecentBallHistoryEntry],
    gnn_embeddings: GNNEmbeddings,
    market_odds: MarketOdds,
    video_fetcher: Callable[[str], Optional[VideoSignals]],
) -> Dict[str, Any]:
    """
    Executes the full inference pipeline for a single live ball event.
    """
    video_signals = video_fetcher(current_ball_features.ball)

    predictions = run_inference(
        model=model,
        current_ball_features=current_ball_features,
        recent_ball_history=recent_ball_history,
        gnn_embeddings=gnn_embeddings,
        video_signals=video_signals,
    )

    # Determine which team is batting to fetch the correct odds
    batting_team_id = (
        current_ball_features.home_team_id
        if current_ball_features.batting_team_name == current_ball_features.home_team_name
        else current_ball_features.away_team_id
    )
    market_odds_for_team = market_odds.win_odds.get(batting_team_id)

    betting_decision = format_shadow_bet_decision(
        model_outputs=predictions,
        betfair_odds=market_odds_for_team,
    )

    result = {
        "predictions": predictions,
        "betting_decision": betting_decision,
    }

    logger.info(f"Pipeline result for ball {current_ball_features.ball}: {result}")
    return result 