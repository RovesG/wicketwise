# Purpose: DGL Rules module - individual rule implementations
# Author: WicketWise AI, Last Modified: 2024

"""
DGL Rules Module

Contains individual rule implementations for the Deterministic Governance Layer.
Each rule is implemented as a separate class for modularity and testability.

Rule Categories:
- Bankroll Rules: Exposure limits based on bankroll percentages
- P&L Rules: Loss protection guards
- Liquidity Rules: Market liquidity and execution constraints
- Rate Limit Rules: Request throttling and DDoS protection
- Concentration Rules: Market and correlation exposure limits
- Compliance Rules: Jurisdiction and regulatory compliance
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"
