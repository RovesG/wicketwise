# Purpose: DGL client module for orchestrator integration
# Author: WicketWise AI, Last Modified: 2024

"""
DGL Client Module

Provides client interfaces for:
- DGL service integration
- Orchestrator communication
- Request/response handling
- Connection management
"""

from .dgl_client import DGLClient
from .orchestrator_mock import MockOrchestrator
from .integration import DGLIntegration

__all__ = [
    "DGLClient",
    "MockOrchestrator", 
    "DGLIntegration"
]
