# Purpose: DGL observability module for metrics, monitoring, and audit verification
# Author: WicketWise AI, Last Modified: 2024

"""
DGL Observability Module

Provides comprehensive observability for DGL system:
- Metrics collection and aggregation
- Performance monitoring and alerting
- Audit trail verification and integrity checks
- System health monitoring
- Real-time dashboards and reporting
"""

from .metrics_collector import MetricsCollector, MetricType, Metric
from .performance_monitor import PerformanceMonitor, PerformanceAlert, AlertSeverity
from .audit_verifier import AuditVerifier, IntegrityCheck, VerificationResult
from .health_monitor import HealthMonitor, HealthStatus, ComponentHealth
from .dashboard_exporter import DashboardExporter, DashboardConfig

__all__ = [
    "MetricsCollector",
    "MetricType",
    "Metric",
    "PerformanceMonitor",
    "PerformanceAlert",
    "AlertSeverity",
    "AuditVerifier",
    "IntegrityCheck",
    "VerificationResult",
    "HealthMonitor",
    "HealthStatus",
    "ComponentHealth",
    "DashboardExporter",
    "DashboardConfig"
]
