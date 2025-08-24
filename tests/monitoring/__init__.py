# Purpose: Performance monitoring tests module initialization
# Author: WicketWise AI, Last Modified: 2024

"""
Performance Monitoring Tests Module

This module contains comprehensive tests for the WicketWise performance monitoring
and optimization system, including system resource monitoring, agent performance
tracking, and compliance monitoring.

Test Categories:
- Performance monitor core functionality
- System resource monitoring and alerting
- Agent performance tracking and optimization
- Compliance monitoring and audit trails
- Integration testing with the agent system
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Test utilities and fixtures
from .test_performance_monitor import (
    TestPerformanceMonitor,
    TestSystemResourceMonitor,
    TestMetricsCollector
)

# Agent performance tracker tests (to be implemented)
# from .test_agent_performance_tracker import (
#     TestAgentPerformanceTracker,
#     TestAgentMetrics,
#     TestOptimizationRecommendations
# )

__all__ = [
    'TestPerformanceMonitor',
    'TestSystemResourceMonitor', 
    'TestMetricsCollector'
    # 'TestAgentPerformanceTracker',
    # 'TestAgentMetrics',
    # 'TestOptimizationRecommendations'
]
