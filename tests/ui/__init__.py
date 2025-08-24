# Purpose: UI testing module initialization
# Author: WicketWise AI, Last Modified: 2024

"""
UI Testing Module

This module contains comprehensive tests for the WicketWise UI components,
including the Agent Dashboard, explainability features, and backend integration.

Test Categories:
- Backend API endpoint testing
- JavaScript functionality validation  
- UI integration scenarios
- Multi-agent visualization testing
- Explainable AI component testing
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Test utilities and fixtures
from .test_agent_dashboard import (
    TestAgentDashboardBackend,
    TestAgentDashboardJavaScript, 
    TestUIIntegration,
    run_ui_tests
)

__all__ = [
    'TestAgentDashboardBackend',
    'TestAgentDashboardJavaScript',
    'TestUIIntegration', 
    'run_ui_tests'
]
