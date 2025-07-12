# Purpose: Centralized configuration management for the Crickformers project.
# Author: Shamus Rae, Last Modified: 2024-07-30

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ModelArchConfig(BaseModel):
    """Configuration for the model architecture."""
    sequence_encoder: Dict[str, Any] = Field(
        ..., description="Config for BallHistoryEncoder."
    )
    static_context_encoder: Dict[str, Any] = Field(
        ..., description="Config for StaticContextEncoder."
    )
    fusion_layer: Dict[str, Any] = Field(..., description="Config for FusionLayer.")
    prediction_heads: Dict[str, Any] = Field(
        ..., description="Configs for all prediction heads."
    )


class GNNConfig(BaseModel):
    """Configuration for GNN embeddings."""
    embedding_dim: int = Field(128, description="Dimensionality of GNN embeddings.")
    embeddings_path: str = Field(
        "data/gnn_embeddings.pt", description="Path to pre-trained GNN embeddings."
    )


class AgentConfig(BaseModel):
    """Configuration for the ShadowBettingAgent."""
    value_threshold: float = Field(
        0.05, description="Minimum edge to consider a value bet."
    )
    risk_confidence_threshold: float = Field(
        0.75, description="Confidence level to trigger risk alerts."
    )


class DataPathConfig(BaseModel):
    """Configuration for data paths."""
    video_features_dir: Optional[str] = Field(
        None, description="Directory for video features JSON files."
    )
    market_odds_dir: Optional[str] = Field(
        None, description="Directory for market odds data."
    )


class CrickformerConfig(BaseModel):
    """
    Root configuration class for the Crickformer project.
    """
    model: ModelArchConfig
    gnn: GNNConfig
    agent: AgentConfig
    data_paths: DataPathConfig

    @classmethod
    def from_file(cls, path: str | Path) -> CrickformerConfig:
        """Loads configuration from a YAML or JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found at: {p}")

        with p.open("r") as f:
            if p.suffix in (".yaml", ".yml"):
                config_dict = yaml.safe_load(f)
            elif p.suffix == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError("Unsupported config file format. Use .yaml or .json.")
        
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration to a dictionary."""
        return self.dict() 