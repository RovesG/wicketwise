# Purpose: Deterministic Governance Layer (DGL) - AI-independent safety engine
# Author: WicketWise AI, Last Modified: 2024

"""
Deterministic Governance Layer (DGL)

The DGL is an AI-independent safety engine that sits between WicketWise's 
orchestrator and any execution target (Betfair, Decimal, or simulator).
It enforces bankroll, exposure, liquidity, and compliance rules deterministically.

Key Components:
- Rule Engine: Validates bet proposals against hard constraints
- Audit System: Immutable log with hash chaining for compliance
- State Machine: READY → SHADOW → LIVE → KILLED governance states
- API Layer: FastAPI endpoints for governance decisions and monitoring
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"
