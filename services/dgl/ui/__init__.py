# Purpose: DGL UI module for Streamlit interfaces
# Author: WicketWise AI, Last Modified: 2024

"""
DGL UI Module

Provides Streamlit-based user interfaces for:
- Limits and governance configuration
- Real-time monitoring dashboards
- Rule management interfaces
- Audit trail visualization
"""

from .governance_dashboard import GovernanceDashboard
from .limits_manager import LimitsManager
from .audit_viewer import AuditViewer
from .monitoring_panel import MonitoringPanel

__all__ = [
    "GovernanceDashboard",
    "LimitsManager",
    "AuditViewer", 
    "MonitoringPanel"
]
