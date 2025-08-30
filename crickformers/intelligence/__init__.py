# Purpose: Intelligence Module for WicketWise
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
WicketWise Intelligence Module

This module provides intelligent agents for gathering and processing
real-time cricket information from various sources:

- WebIntelligenceAgent: Gathers web/news data
- PlayerInsightAgent: LLM-powered player analysis  
- MatchIntelligenceAgent: Match-specific insights
- BettingIntelligenceAgent: Market and odds analysis

All agents are designed to be reusable across different betting
and analysis systems.
"""

from .web_intelligence_agent import (
    WebIntelligenceAgent,
    IntelligenceType,
    IntelligenceResult,
    create_web_intelligence_agent
)

try:
    from .web_cricket_intelligence_agent import (
        WebCricketIntelligenceAgent,
        WebIntelRequest,
        WebIntelResponse,
        WebIntelIntent,
        WebIntelStatus,
        create_web_cricket_intelligence_agent
    )
    WEB_CRICKET_INTEL_AVAILABLE = True
except ImportError:
    WEB_CRICKET_INTEL_AVAILABLE = False

try:
    from .player_insight_agent import (
        PlayerInsightAgent,
        create_player_insight_agent
    )
    PLAYER_INSIGHT_AVAILABLE = True
except ImportError:
    PLAYER_INSIGHT_AVAILABLE = False

__all__ = [
    'WebIntelligenceAgent',
    'IntelligenceType', 
    'IntelligenceResult',
    'create_web_intelligence_agent'
]

if WEB_CRICKET_INTEL_AVAILABLE:
    __all__.extend([
        'WebCricketIntelligenceAgent',
        'WebIntelRequest',
        'WebIntelResponse', 
        'WebIntelIntent',
        'WebIntelStatus',
        'create_web_cricket_intelligence_agent'
    ])

if PLAYER_INSIGHT_AVAILABLE:
    __all__.extend([
        'PlayerInsightAgent',
        'create_player_insight_agent'
    ])
