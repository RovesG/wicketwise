# Purpose: A Streamlit UI for live inference with the Crickformer model.
# Author: Shamus Rae, Last Modified: 2024-07-30

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import torch
from crickformers.data.data_schema import (
    CurrentBallFeatures,
    GNNEmbeddings,
    MarketOdds,
    RecentBallHistoryEntry,
    VideoSignals,
)
from crickformers.inference.inference_wrapper import run_inference
from crickformers.model.crickformer_model import CrickformerModel
from crickformers.agent.shadow_betting_agent import ShadowBettingAgent


# Mock Model and Agent
# In a real application, these would be loaded from saved artifacts.
@st.cache_resource
def load_model_and_agent():
    # Placeholder configurations matching the test setup
    config = {
        "model": {
            "sequence_encoder": {"input_dim": 4, "hidden_dim": 32, "num_layers": 2},
            "static_context_encoder": {"input_dim": 8, "hidden_dim": 16},
            "fusion_layer": {
                "sequence_dim": 32,
                "context_dim": 16,
                "kg_dim": 128,
                "hidden_dims": [64],
                "latent_dim": 64,
            },
            "prediction_heads": {
                "next_ball_outcome": {"input_dim": 64, "output_dim": 7},
                "win_probability": {"input_dim": 64, "output_dim": 1},
                "odds_mispricing": {"input_dim": 64, "output_dim": 1},
            },
        },
    }
    model = CrickformerModel(
        sequence_config=config["model"]["sequence_encoder"],
        static_config=config["model"]["static_context_encoder"],
        fusion_config=config["model"]["fusion_layer"],
        prediction_heads_config=config["model"]["prediction_heads"],
    )
    agent = ShadowBettingAgent()
    return model, agent

MODEL, AGENT = load_model_and_agent()


def main():
    st.set_page_config(layout="wide", page_title="Crickformers Live Inference")
    st.title("🏏 Crickformers: Live T20 Prediction")

    # --- Data Loading ---
    st.sidebar.header("Load Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Ball Sequence JSON", type=["json"]
    )
    use_sample = st.sidebar.button("Load Sample Data")

    if use_sample:
        with open("samples/mock_ball_sequence.json", "r") as f:
            data = json.load(f)
    elif uploaded_file is not None:
        data = json.load(uploaded_file)
    else:
        data = {}

    # --- User Inputs ---
    st.sidebar.header("Manual Input")
    # Simplified inputs for demonstration
    current_over = st.sidebar.number_input(
        "Current Over", min_value=0, max_value=19, value=data.get("current_ball_features", {}).get("over", 15)
    )
    batter_runs = st.sidebar.number_input(
        "Batter Runs", value=data.get("current_ball_features", {}).get("batsman_runs", 45)
    )
    market_odds_input = st.sidebar.number_input(
        "Market Win Odds", value=data.get("market_odds", {}).get("win_odds", {}).get("team_a", 1.5), format="%.2f"
    )

    st.header("Current Match State")
    col1, col2, col3 = st.columns(3)
    col1.metric("Over", current_over)
    col2.metric("Batter Runs", batter_runs)
    col3.metric("Market Odds", f"{market_odds_input:.2f}")

    if st.button("Run Prediction"):
        if not data:
            st.error("Please load data or fill in manual inputs.")
            return

        # --- Prepare Inputs for Model ---
        # This is a simplified mapping from the full data structure
        current_ball = CurrentBallFeatures(**data["current_ball_features"])
        recent_history = [
            RecentBallHistoryEntry(**entry) for entry in data["recent_ball_history"]
        ]
        gnn_embeddings = GNNEmbeddings(**data["gnn_embeddings"])
        video_signals = (
            VideoSignals(**data["video_signals"]) if "video_signals" in data else None
        )
        market_odds = MarketOdds(**data["market_odds"])
        
        # --- Run Inference ---
        predictions = run_inference(
            model=MODEL,
            current_ball_features=current_ball,
            recent_ball_history=recent_history,
            gnn_embeddings=gnn_embeddings,
            video_signals=video_signals,
        )

        batting_team_id = (
            current_ball.home_team_id
            if current_ball.batting_team_name == current_ball.home_team_name
            else current_ball.away_team_id
        )
        betting_decision = AGENT.make_decision(
            win_probability=predictions["win_probability"],
            market_odds=market_odds.win_odds.get(batting_team_id),
        )

        # --- Display Outputs ---
        st.header("Model Predictions")
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            st.subheader("Win Probability")
            win_prob = predictions["win_probability"]
            st.progress(win_prob)
            st.metric("Model Win Probability", f"{win_prob:.2%}")

        with pred_col2:
            st.subheader("Next Ball Outcome")
            outcome_labels = ["Dot", "1", "2", "3", "4", "6", "Wicket"]
            outcome_probs = predictions["next_ball_outcome"]
            chart_data = pd.DataFrame({"Probability": outcome_probs}, index=outcome_labels)
            st.bar_chart(chart_data)

        st.header("Shadow Betting Decision")
        decision = betting_decision["decision"]
        reason = betting_decision["reason"]
        
        if decision == "value_bet":
            st.success(f"**Decision:** {decision.upper()} - {reason}")
        elif decision == "risk_alert":
            st.warning(f"**Decision:** {decision.upper()} - {reason}")
        else:
            st.info(f"**Decision:** {decision.upper()} - {reason}")

        with st.expander("Show Decision Log"):
            st.json(betting_decision)


if __name__ == "__main__":
    main() 