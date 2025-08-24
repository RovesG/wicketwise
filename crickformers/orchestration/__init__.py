# Purpose: Orchestration layer for WicketWise AI system
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Orchestration layer that coordinates between different AI components:
- Mixture of Experts routing
- Knowledge Graph queries
- Real-time prediction pipeline
- Betting intelligence
"""

from .moe_orchestrator import MoEOrchestrator
from .prediction_pipeline import PredictionPipeline

__all__ = ["MoEOrchestrator", "PredictionPipeline"]
