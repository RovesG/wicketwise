# Purpose: API module for DGL governance endpoints
# Author: WicketWise AI, Last Modified: 2024

"""
DGL API Module

Provides REST API endpoints for:
- Governance decision evaluation
- Exposure monitoring and reporting
- Rules configuration and management
- Audit trail access
- System health and metrics
"""

from .governance import governance_router
from .exposure import exposure_router
from .rules import rules_router
from .audit import audit_router

__all__ = [
    "governance_router",
    "exposure_router", 
    "rules_router",
    "audit_router"
]
