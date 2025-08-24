# Purpose: LLM Agent Orchestration Layer - Module Initialization
# Author: WicketWise Team, Last Modified: 2025-08-24

"""
LLM Agent Orchestration Layer
=============================

Multi-agent system for coordinating specialized cricket analysis agents:

Core Components:
- BaseAgent: Abstract agent protocol
- PerformanceAgent: Player performance analysis
- TacticalAgent: Match strategy and tactics
- PredictionAgent: Match outcome predictions
- BettingAgent: Value opportunity analysis
- OrchestrationEngine: Agent coordination and routing

Agent Types:
- Specialist agents for focused analysis
- Coordinator agents for complex workflows
- Monitor agents for performance tracking
"""

from .base_agent import BaseAgent, AgentCapability, AgentResponse
from .orchestration_engine import OrchestrationEngine, AgentCoordinator
from .performance_agent import PerformanceAgent
from .tactical_agent import TacticalAgent
from .prediction_agent import PredictionAgent
from .betting_agent import BettingAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentCapability", 
    "AgentResponse",
    
    # Orchestration
    "OrchestrationEngine",
    "AgentCoordinator",
    
    # Specialized agents
    "PerformanceAgent",
    "TacticalAgent", 
    "PredictionAgent",
    "BettingAgent"
]

__version__ = "1.0.0"
