# Purpose: Dashboard data exporter for DGL observability
# Author: WicketWise AI, Last Modified: 2024

"""
Dashboard Exporter

Exports observability data for dashboards and reporting:
- Metrics data export
- Health status export
- Performance reports
- Compliance reports
- Real-time data feeds
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor
from .audit_verifier import AuditVerifier
from .health_monitor import HealthMonitor


logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Export format options"""
    JSON = "json"
    PROMETHEUS = "prometheus"
    CSV = "csv"
    HTML = "html"


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    name: str
    description: str
    refresh_interval_seconds: int = 30
    data_retention_hours: int = 24
    export_format: ExportFormat = ExportFormat.JSON
    include_metrics: bool = True
    include_health: bool = True
    include_performance: bool = True
    include_audit: bool = True
    custom_queries: Dict[str, str] = field(default_factory=dict)


class DashboardExporter:
    """
    Dashboard data exporter for DGL observability
    
    Provides unified data export for dashboards, monitoring tools,
    and reporting systems with multiple format support.
    """
    
    def __init__(self, 
                 metrics_collector: Optional[MetricsCollector] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 audit_verifier: Optional[AuditVerifier] = None,
                 health_monitor: Optional[HealthMonitor] = None):
        """
        Initialize dashboard exporter
        
        Args:
            metrics_collector: Metrics collector instance
            performance_monitor: Performance monitor instance
            audit_verifier: Audit verifier instance
            health_monitor: Health monitor instance
        """
        self.metrics_collector = metrics_collector
        self.performance_monitor = performance_monitor
        self.audit_verifier = audit_verifier
        self.health_monitor = health_monitor
        
        # Dashboard configurations
        self.dashboards: Dict[str, DashboardConfig] = {}
        
        # Setup default dashboards
        self._setup_default_dashboards()
        
        logger.info("Dashboard exporter initialized")
    
    def register_dashboard(self, config: DashboardConfig):
        """Register a dashboard configuration"""
        self.dashboards[config.name] = config
        logger.info(f"Registered dashboard: {config.name}")
    
    def unregister_dashboard(self, name: str):
        """Unregister a dashboard"""
        if name in self.dashboards:
            del self.dashboards[name]
            logger.info(f"Unregistered dashboard: {name}")
    
    def export_dashboard_data(self, dashboard_name: str, 
                            hours: int = None) -> Dict[str, Any]:
        """
        Export data for a specific dashboard
        
        Args:
            dashboard_name: Name of dashboard to export
            hours: Hours of historical data to include
            
        Returns:
            Dashboard data dictionary
        """
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_name} not found")
        
        config = self.dashboards[dashboard_name]
        hours = hours or config.data_retention_hours
        
        dashboard_data = {
            "dashboard": {
                "name": config.name,
                "description": config.description,
                "generated_at": datetime.now().isoformat(),
                "data_period_hours": hours
            }
        }
        
        # Export metrics data
        if config.include_metrics and self.metrics_collector:
            dashboard_data["metrics"] = self._export_metrics_data(hours)
        
        # Export health data
        if config.include_health and self.health_monitor:
            dashboard_data["health"] = self._export_health_data(hours)
        
        # Export performance data
        if config.include_performance and self.performance_monitor:
            dashboard_data["performance"] = self._export_performance_data(hours)
        
        # Export audit data
        if config.include_audit and self.audit_verifier:
            dashboard_data["audit"] = self._export_audit_data(hours)
        
        # Execute custom queries
        if config.custom_queries:
            dashboard_data["custom"] = self._execute_custom_queries(config.custom_queries, hours)
        
        return dashboard_data
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.metrics_collector:
            return "# No metrics collector available\n"
        
        return self.metrics_collector.export_prometheus_format()
    
    def export_health_status(self) -> Dict[str, Any]:
        """Export current health status"""
        if not self.health_monitor:
            return {"error": "No health monitor available"}
        
        if not self.health_monitor.health_history:
            return {"error": "No health data available"}
        
        latest_health = self.health_monitor.health_history[-1]
        return latest_health.to_dict()
    
    def export_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Export performance report"""
        if not self.performance_monitor:
            return {"error": "No performance monitor available"}
        
        try:
            report = self.performance_monitor.generate_performance_report(hours)
            
            return {
                "period_start": report.period_start.isoformat(),
                "period_end": report.period_end.isoformat(),
                "sla_compliance": report.sla_compliance,
                "performance_summary": report.performance_summary,
                "alerts_summary": report.alerts_summary,
                "recommendations": report.recommendations,
                "trends": report.trends
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {"error": str(e)}
    
    def export_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Export audit verification summary"""
        if not self.audit_verifier:
            return {"error": "No audit verifier available"}
        
        try:
            # Get recent verification results
            recent_verifications = self.audit_verifier.get_verification_history(limit=5)
            
            if not recent_verifications:
                return {"error": "No audit verification data available"}
            
            latest_verification = recent_verifications[0]
            
            return {
                "latest_verification": latest_verification.to_dict(),
                "recent_verifications": [v.to_dict() for v in recent_verifications],
                "verification_trend": self._analyze_verification_trend(recent_verifications)
            }
            
        except Exception as e:
            logger.error(f"Error generating audit summary: {str(e)}")
            return {"error": str(e)}
    
    def export_system_overview(self) -> Dict[str, Any]:
        """Export comprehensive system overview"""
        overview = {
            "generated_at": datetime.now().isoformat(),
            "system_status": "unknown"
        }
        
        # Health overview
        if self.health_monitor and self.health_monitor.health_history:
            latest_health = self.health_monitor.health_history[-1]
            overview["system_status"] = latest_health.overall_status.value
            overview["uptime_seconds"] = latest_health.uptime_seconds
            overview["component_count"] = len(latest_health.components)
            overview["healthy_components"] = len([
                c for c in latest_health.components.values() 
                if c.status.value == "healthy"
            ])
        
        # Metrics overview
        if self.metrics_collector:
            all_metrics = self.metrics_collector.get_all_metrics()
            overview["metrics"] = {
                "counter_count": len(all_metrics.get("counters", {})),
                "gauge_count": len(all_metrics.get("gauges", {})),
                "histogram_count": len(all_metrics.get("histograms", {})),
                "timer_count": len(all_metrics.get("timers", {}))
            }
        
        # Performance overview
        if self.performance_monitor:
            active_alerts = self.performance_monitor.get_active_alerts()
            overview["alerts"] = {
                "active_count": len(active_alerts),
                "critical_count": len([a for a in active_alerts if a.severity.value == "critical"]),
                "warning_count": len([a for a in active_alerts if a.severity.value == "warning"])
            }
        
        # Audit overview
        if self.audit_verifier:
            recent_verifications = self.audit_verifier.get_verification_history(limit=1)
            if recent_verifications:
                latest = recent_verifications[0]
                overview["audit"] = {
                    "last_verification": latest.completed_at.isoformat(),
                    "verification_status": latest.overall_status.value,
                    "checks_passed": latest.passed_checks,
                    "checks_failed": latest.failed_checks
                }
        
        return overview
    
    def _export_metrics_data(self, hours: int) -> Dict[str, Any]:
        """Export metrics data for dashboard"""
        try:
            all_metrics = self.metrics_collector.get_all_metrics()
            
            # Get system metrics
            system_metrics = self.metrics_collector.get_system_metrics()
            
            return {
                "current_metrics": all_metrics,
                "system_metrics": system_metrics,
                "collection_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting metrics data: {str(e)}")
            return {"error": str(e)}
    
    def _export_health_data(self, hours: int) -> Dict[str, Any]:
        """Export health data for dashboard"""
        try:
            # Current health status
            current_health = self.export_health_status()
            
            # Health summary
            health_summary = self.health_monitor.get_health_summary(hours)
            
            # Recent health history
            health_history = self.health_monitor.get_health_history(hours)
            
            return {
                "current_health": current_health,
                "health_summary": health_summary,
                "health_history": [h.to_dict() for h in health_history[-50:]]  # Last 50 entries
            }
            
        except Exception as e:
            logger.error(f"Error exporting health data: {str(e)}")
            return {"error": str(e)}
    
    def _export_performance_data(self, hours: int) -> Dict[str, Any]:
        """Export performance data for dashboard"""
        try:
            # Performance summary
            performance_summary = self.performance_monitor.get_performance_summary(hours)
            
            # Active alerts
            active_alerts = self.performance_monitor.get_active_alerts()
            
            # Alert history
            alert_history = self.performance_monitor.get_alert_history(hours)
            
            return {
                "performance_summary": performance_summary,
                "active_alerts": [alert.__dict__ for alert in active_alerts],
                "alert_history": [alert.__dict__ for alert in alert_history],
                "sla_targets": self.performance_monitor.sla_targets
            }
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {str(e)}")
            return {"error": str(e)}
    
    def _export_audit_data(self, hours: int) -> Dict[str, Any]:
        """Export audit data for dashboard"""
        try:
            return self.export_audit_summary(hours)
            
        except Exception as e:
            logger.error(f"Error exporting audit data: {str(e)}")
            return {"error": str(e)}
    
    def _execute_custom_queries(self, queries: Dict[str, str], hours: int) -> Dict[str, Any]:
        """Execute custom queries for dashboard"""
        results = {}
        
        for query_name, query_config in queries.items():
            try:
                # This is a simplified implementation
                # In production, would support more complex query languages
                if query_config == "system_load":
                    if self.metrics_collector:
                        system_metrics = self.metrics_collector.get_system_metrics()
                        results[query_name] = system_metrics.get("cpu", {})
                
                elif query_config == "error_rate":
                    if self.performance_monitor:
                        perf_summary = self.performance_monitor.get_performance_summary(hours)
                        results[query_name] = perf_summary.get("error_rate", {})
                
                else:
                    results[query_name] = {"error": f"Unknown query: {query_config}"}
                    
            except Exception as e:
                results[query_name] = {"error": str(e)}
        
        return results
    
    def _analyze_verification_trend(self, verifications: List) -> Dict[str, Any]:
        """Analyze audit verification trend"""
        if len(verifications) < 2:
            return {"trend": "insufficient_data"}
        
        # Count passed vs failed verifications
        recent_passed = len([v for v in verifications[:3] if v.overall_status.value == "passed"])
        older_passed = len([v for v in verifications[3:] if v.overall_status.value == "passed"])
        
        recent_rate = recent_passed / min(3, len(verifications))
        older_rate = older_passed / max(1, len(verifications) - 3) if len(verifications) > 3 else recent_rate
        
        if recent_rate > older_rate:
            trend = "improving"
        elif recent_rate < older_rate:
            trend = "degrading"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_success_rate": recent_rate,
            "older_success_rate": older_rate,
            "total_verifications": len(verifications)
        }
    
    def _setup_default_dashboards(self):
        """Setup default dashboard configurations"""
        
        # System Overview Dashboard
        self.register_dashboard(DashboardConfig(
            name="system_overview",
            description="Comprehensive system overview dashboard",
            refresh_interval_seconds=30,
            data_retention_hours=24,
            include_metrics=True,
            include_health=True,
            include_performance=True,
            include_audit=True
        ))
        
        # Performance Dashboard
        self.register_dashboard(DashboardConfig(
            name="performance",
            description="System performance monitoring dashboard",
            refresh_interval_seconds=15,
            data_retention_hours=12,
            include_metrics=True,
            include_health=False,
            include_performance=True,
            include_audit=False,
            custom_queries={
                "response_time_trend": "response_time",
                "throughput_trend": "throughput",
                "error_rate_trend": "error_rate"
            }
        ))
        
        # Health Dashboard
        self.register_dashboard(DashboardConfig(
            name="health",
            description="System health monitoring dashboard",
            refresh_interval_seconds=60,
            data_retention_hours=48,
            include_metrics=False,
            include_health=True,
            include_performance=False,
            include_audit=False,
            custom_queries={
                "system_load": "system_load",
                "component_status": "component_status"
            }
        ))
        
        # Audit Dashboard
        self.register_dashboard(DashboardConfig(
            name="audit",
            description="Audit trail and compliance dashboard",
            refresh_interval_seconds=300,  # 5 minutes
            data_retention_hours=168,  # 1 week
            include_metrics=False,
            include_health=False,
            include_performance=False,
            include_audit=True
        ))
    
    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards"""
        return [
            {
                "name": config.name,
                "description": config.description,
                "refresh_interval_seconds": config.refresh_interval_seconds,
                "data_retention_hours": config.data_retention_hours,
                "export_format": config.export_format.value
            }
            for config in self.dashboards.values()
        ]


# Utility functions for dashboard export

def create_dashboard_exporter(metrics_collector: MetricsCollector = None,
                            performance_monitor: PerformanceMonitor = None,
                            audit_verifier: AuditVerifier = None,
                            health_monitor: HealthMonitor = None) -> DashboardExporter:
    """Create and configure dashboard exporter"""
    return DashboardExporter(
        metrics_collector=metrics_collector,
        performance_monitor=performance_monitor,
        audit_verifier=audit_verifier,
        health_monitor=health_monitor
    )


def export_to_file(exporter: DashboardExporter, dashboard_name: str, 
                  filename: str, format_type: ExportFormat = ExportFormat.JSON):
    """Export dashboard data to file"""
    try:
        data = exporter.export_dashboard_data(dashboard_name)
        
        if format_type == ExportFormat.JSON:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format_type == ExportFormat.PROMETHEUS:
            prometheus_data = exporter.export_prometheus_metrics()
            with open(filename, 'w') as f:
                f.write(prometheus_data)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Dashboard data exported to {filename}")
        
    except Exception as e:
        logger.error(f"Error exporting dashboard data: {str(e)}")
        raise
