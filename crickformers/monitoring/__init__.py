# Purpose: Performance monitoring and metrics collection module
# Author: WicketWise AI, Last Modified: 2024

"""
Performance Monitoring Module

This module provides comprehensive performance monitoring, metrics collection,
and system health tracking for the WicketWise cricket intelligence platform.

Key Components:
- Performance metrics collection and aggregation
- System resource monitoring (CPU, memory, disk)
- Agent performance tracking and optimization
- Real-time alerting and notification system
- Performance analytics and reporting
- Compliance monitoring and audit trails
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Core monitoring components
from .performance_monitor import (
    PerformanceMonitor,
    MetricsCollector,
    SystemResourceMonitor,
    PerformanceAlert,
    AlertSeverity
)

from .agent_performance_tracker import (
    AgentPerformanceTracker,
    AgentMetrics,
    PerformanceThreshold,
    OptimizationRecommendation
)

# Compliance monitoring (to be implemented in Phase 6.3)
# from .compliance_monitor import (
#     ComplianceMonitor,
#     AuditLogger,
#     DataPrivacyTracker,
#     ComplianceReport
# )

__all__ = [
    # Performance monitoring
    'PerformanceMonitor',
    'MetricsCollector', 
    'SystemResourceMonitor',
    'PerformanceAlert',
    'AlertSeverity',
    
    # Agent performance tracking
    'AgentPerformanceTracker',
    'AgentMetrics',
    'PerformanceThreshold',
    'OptimizationRecommendation',
    
    # Compliance monitoring (to be implemented in Phase 6.3)
    # 'ComplianceMonitor',
    # 'AuditLogger',
    # 'DataPrivacyTracker',
    # 'ComplianceReport'
]
