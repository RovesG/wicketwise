# Purpose: Betting Intelligence module for WicketWise
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Betting Intelligence module that provides:
- Mispricing detection across multiple bookmakers
- Value betting calculations with Kelly criterion
- Market analysis and sentiment tracking
- Risk management and portfolio optimization
- Performance analytics and backtesting

This module enables sophisticated betting strategies based on AI predictions
and market inefficiencies.
"""

from .mispricing_engine import MispricingEngine, OddsData, ValueOpportunity

__all__ = [
    "MispricingEngine",
    "OddsData", 
    "ValueOpportunity"
]
