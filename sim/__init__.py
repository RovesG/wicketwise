# Purpose: WicketWise Simulator & Market Replay (SIM) System
# Author: WicketWise AI, Last Modified: 2024

"""
WicketWise Simulator & Market Replay (SIM) System

Provides offline and semi-online environment to replay, stress-test, and optimize
WicketWise strategies and the Deterministic Governance Layer (DGL) before real money exposure.

Key Features:
- Market Replay: Historical ball-by-ball events + price/volume snapshots
- Model-in-the-Loop: Real match events with contemporaneous model predictions
- Synthetic Monte Carlo: Generated futures using innings/win models
- Walk-Forward Backtesting: Rolling windows with strict cutoff enforcement
- Paper Trading: Same execution path as live, routed to SIM matching engine

Architecture:
- Deterministic by default with seed control
- Latency & liquidity first-class citizens
- Reproducible runs with hash verification
- Single source of truth via MatchState and MarketState
"""

from .config import SimulationConfig, SimulationResult
from .state import MatchState, MarketState, MatchEvent, MarketSnapshot
from .strategy import Strategy, StrategyAction, FillEvent, AccountState
from .adapters import ReplayAdapter, SyntheticAdapter, WalkForwardAdapter
from .matching import MatchingEngine
from .dgl_adapter import SimDGLAdapter
from .orchestrator import SimOrchestrator
from .metrics import SimMetrics, KPICalculator

__all__ = [
    "SimulationConfig",
    "SimulationResult", 
    "MatchState",
    "MarketState",
    "MatchEvent",
    "MarketSnapshot",
    "Strategy",
    "StrategyAction",
    "FillEvent", 
    "AccountState",
    "ReplayAdapter",
    "SyntheticAdapter",
    "WalkForwardAdapter",
    "MatchingEngine",
    "SimDGLAdapter",
    "SimOrchestrator",
    "SimMetrics",
    "KPICalculator"
]
