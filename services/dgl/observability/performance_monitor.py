# Purpose: Performance monitoring and alerting for DGL system
# Author: WicketWise AI, Last Modified: 2024

"""
Performance Monitor

Monitors DGL system performance and generates alerts:
- Real-time performance tracking
- Threshold-based alerting
- Performance trend analysis
- Automated remediation suggestions
- SLA monitoring and reporting
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import statistics

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    comparison_operator: str = ">"  # >, <, >=, <=, ==, !=
    evaluation_window_minutes: int = 5
    min_data_points: int = 3
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    threshold_value: float
    current_value: float
    message: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def acknowledge(self, user: str):
        """Acknowledge the alert"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = user
        self.updated_at = datetime.now()
    
    def resolve(self):
        """Resolve the alert"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()
    
    def suppress(self):
        """Suppress the alert"""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = datetime.now()


@dataclass
class PerformanceReport:
    """Performance analysis report"""
    period_start: datetime
    period_end: datetime
    sla_compliance: Dict[str, float]
    performance_summary: Dict[str, Any]
    alerts_summary: Dict[str, int]
    recommendations: List[str]
    trends: Dict[str, str]  # "improving", "degrading", "stable"


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for DGL
    
    Monitors system performance, generates alerts, and provides
    insights for performance optimization and SLA compliance.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize performance monitor
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        
        # Alert management
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.thresholds: List[PerformanceThreshold] = []
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Performance tracking
        self.sla_targets: Dict[str, float] = {}
        self.performance_history: deque = deque(maxlen=10000)
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Default thresholds
        self._setup_default_thresholds()
        
        logger.info("Performance monitor initialized")
    
    def start(self):
        """Start performance monitoring"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        logger.info("Performance monitoring stopped")
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add performance threshold"""
        self.thresholds.append(threshold)
        logger.info(f"Added performance threshold for {threshold.metric_name}")
    
    def remove_threshold(self, metric_name: str, tags: Dict[str, str] = None):
        """Remove performance threshold"""
        self.thresholds = [
            t for t in self.thresholds 
            if not (t.metric_name == metric_name and t.tags == (tags or {}))
        ]
        logger.info(f"Removed performance threshold for {metric_name}")
    
    def set_sla_target(self, metric_name: str, target_value: float):
        """Set SLA target for a metric"""
        self.sla_targets[metric_name] = target_value
        logger.info(f"Set SLA target for {metric_name}: {target_value}")
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge(user)
            
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} resolved")
            return True
        
        return False
    
    def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.suppress()
            
            logger.info(f"Alert {alert_id} suppressed")
            return True
        
        return False
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[PerformanceAlert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alert history for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history 
            if alert.created_at >= cutoff
        ]
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Get recent performance data
        recent_data = [
            entry for entry in self.performance_history 
            if entry["timestamp"] >= cutoff
        ]
        
        if not recent_data:
            return {"error": "No performance data available"}
        
        # Calculate summary statistics
        response_times = [entry["response_time_ms"] for entry in recent_data]
        throughput_values = [entry["throughput_rps"] for entry in recent_data]
        error_rates = [entry["error_rate_pct"] for entry in recent_data]
        
        return {
            "period": {
                "start": cutoff.isoformat(),
                "end": datetime.now().isoformat(),
                "data_points": len(recent_data)
            },
            "response_time": {
                "mean_ms": statistics.mean(response_times),
                "median_ms": statistics.median(response_times),
                "p95_ms": self._percentile(response_times, 95),
                "p99_ms": self._percentile(response_times, 99),
                "max_ms": max(response_times),
                "min_ms": min(response_times)
            },
            "throughput": {
                "mean_rps": statistics.mean(throughput_values),
                "max_rps": max(throughput_values),
                "min_rps": min(throughput_values)
            },
            "error_rate": {
                "mean_pct": statistics.mean(error_rates),
                "max_pct": max(error_rates),
                "min_pct": min(error_rates)
            },
            "sla_compliance": self._calculate_sla_compliance(recent_data)
        }
    
    def generate_performance_report(self, hours: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get performance summary
        summary = self.get_performance_summary(hours)
        
        # Get alert summary
        alert_history = self.get_alert_history(hours)
        alerts_by_severity = {}
        for severity in AlertSeverity:
            alerts_by_severity[severity.value] = len([
                a for a in alert_history if a.severity == severity
            ])
        
        # Calculate SLA compliance
        sla_compliance = summary.get("sla_compliance", {})
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary, alert_history)
        
        # Analyze trends
        trends = self._analyze_trends(hours)
        
        return PerformanceReport(
            period_start=start_time,
            period_end=end_time,
            sla_compliance=sla_compliance,
            performance_summary=summary,
            alerts_summary=alerts_by_severity,
            recommendations=recommendations,
            trends=trends
        )
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Remove alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def record_performance_data(self, response_time_ms: float, throughput_rps: float, 
                               error_rate_pct: float, additional_metrics: Dict[str, float] = None):
        """Record performance data point"""
        data_point = {
            "timestamp": datetime.now(),
            "response_time_ms": response_time_ms,
            "throughput_rps": throughput_rps,
            "error_rate_pct": error_rate_pct,
            **(additional_metrics or {})
        }
        
        self.performance_history.append(data_point)
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Evaluate all thresholds
                for threshold in self.thresholds:
                    await self._evaluate_threshold(threshold)
                
                # Collect current performance data
                await self._collect_performance_data()
                
                # Auto-resolve alerts if conditions improve
                await self._check_alert_resolution()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    async def _evaluate_threshold(self, threshold: PerformanceThreshold):
        """Evaluate a performance threshold"""
        try:
            # Get metric series
            series = self.metrics_collector.get_metric_series(
                threshold.metric_name, threshold.tags
            )
            
            if not series:
                return
            
            # Get recent values
            recent_values = series.get_recent_values(threshold.evaluation_window_minutes)
            
            if len(recent_values) < threshold.min_data_points:
                return
            
            # Calculate current value (mean of recent values)
            current_value = statistics.mean([v["value"] for v in recent_values])
            
            # Determine alert severity
            severity = self._determine_alert_severity(
                current_value, threshold
            )
            
            if severity:
                await self._create_or_update_alert(
                    threshold, current_value, severity
                )
            
        except Exception as e:
            logger.error(f"Error evaluating threshold for {threshold.metric_name}: {str(e)}")
    
    def _determine_alert_severity(self, value: float, 
                                threshold: PerformanceThreshold) -> Optional[AlertSeverity]:
        """Determine alert severity based on threshold comparison"""
        
        def compare(val, thresh, op):
            if op == ">":
                return val > thresh
            elif op == "<":
                return val < thresh
            elif op == ">=":
                return val >= thresh
            elif op == "<=":
                return val <= thresh
            elif op == "==":
                return val == thresh
            elif op == "!=":
                return val != thresh
            return False
        
        op = threshold.comparison_operator
        
        # Check emergency threshold first
        if (threshold.emergency_threshold is not None and 
            compare(value, threshold.emergency_threshold, op)):
            return AlertSeverity.EMERGENCY
        
        # Check critical threshold
        if compare(value, threshold.critical_threshold, op):
            return AlertSeverity.CRITICAL
        
        # Check warning threshold
        if compare(value, threshold.warning_threshold, op):
            return AlertSeverity.WARNING
        
        return None
    
    async def _create_or_update_alert(self, threshold: PerformanceThreshold, 
                                    current_value: float, severity: AlertSeverity):
        """Create new alert or update existing one"""
        
        # Generate alert ID
        alert_key = f"{threshold.metric_name}_{hash(str(threshold.tags))}"
        
        # Check if alert already exists
        if alert_key in self.active_alerts:
            existing_alert = self.active_alerts[alert_key]
            
            # Update if severity changed
            if existing_alert.severity != severity:
                existing_alert.severity = severity
                existing_alert.current_value = current_value
                existing_alert.updated_at = datetime.now()
                
                # Trigger callbacks for severity change
                for callback in self.alert_callbacks:
                    try:
                        callback(existing_alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {str(e)}")
            
            return
        
        # Create new alert
        alert = PerformanceAlert(
            alert_id=alert_key,
            metric_name=threshold.metric_name,
            severity=severity,
            status=AlertStatus.ACTIVE,
            threshold_value=self._get_threshold_value(threshold, severity),
            current_value=current_value,
            message=self._generate_alert_message(threshold, current_value, severity),
            tags=threshold.tags
        )
        
        self.active_alerts[alert_key] = alert
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
        
        logger.warning(f"Performance alert created: {alert.message}")
    
    def _get_threshold_value(self, threshold: PerformanceThreshold, 
                           severity: AlertSeverity) -> float:
        """Get threshold value for given severity"""
        if severity == AlertSeverity.EMERGENCY and threshold.emergency_threshold is not None:
            return threshold.emergency_threshold
        elif severity == AlertSeverity.CRITICAL:
            return threshold.critical_threshold
        else:
            return threshold.warning_threshold
    
    def _generate_alert_message(self, threshold: PerformanceThreshold, 
                              current_value: float, severity: AlertSeverity) -> str:
        """Generate human-readable alert message"""
        threshold_value = self._get_threshold_value(threshold, severity)
        
        return (f"{severity.value.upper()}: {threshold.metric_name} "
                f"is {current_value:.2f}, exceeds {severity.value} threshold "
                f"of {threshold_value:.2f}")
    
    async def _collect_performance_data(self):
        """Collect current performance data"""
        try:
            # Get response time metrics
            response_time_stats = self.metrics_collector.get_timer_stats(
                "dgl.decision.processing_time"
            )
            
            # Get throughput metrics
            decision_count = self.metrics_collector.get_counter_value(
                "dgl.decisions.total"
            )
            
            # Calculate throughput (decisions per second over last minute)
            # This is a simplified calculation
            throughput_rps = decision_count / 60.0  # Approximate
            
            # Get error rate (simplified)
            error_rate_pct = 0.0  # Would calculate from actual error metrics
            
            # Record performance data
            if response_time_stats:
                self.record_performance_data(
                    response_time_ms=response_time_stats.get("mean_ms", 0),
                    throughput_rps=throughput_rps,
                    error_rate_pct=error_rate_pct
                )
            
        except Exception as e:
            logger.error(f"Error collecting performance data: {str(e)}")
    
    async def _check_alert_resolution(self):
        """Check if any alerts can be auto-resolved"""
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            # Find matching threshold
            threshold = next(
                (t for t in self.thresholds 
                 if t.metric_name == alert.metric_name and t.tags == alert.tags),
                None
            )
            
            if not threshold:
                continue
            
            # Get current metric value
            series = self.metrics_collector.get_metric_series(
                alert.metric_name, alert.tags
            )
            
            if not series:
                continue
            
            recent_values = series.get_recent_values(threshold.evaluation_window_minutes)
            
            if len(recent_values) < threshold.min_data_points:
                continue
            
            current_value = statistics.mean([v["value"] for v in recent_values])
            
            # Check if alert condition no longer applies
            severity = self._determine_alert_severity(current_value, threshold)
            
            if not severity:
                alerts_to_resolve.append(alert_id)
        
        # Resolve alerts
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id)
    
    def _calculate_sla_compliance(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate SLA compliance percentages"""
        compliance = {}
        
        for metric_name, target_value in self.sla_targets.items():
            if metric_name == "response_time_ms":
                values = [entry["response_time_ms"] for entry in performance_data]
                compliant_count = len([v for v in values if v <= target_value])
                compliance[metric_name] = (compliant_count / len(values)) * 100 if values else 0
            
            elif metric_name == "error_rate_pct":
                values = [entry["error_rate_pct"] for entry in performance_data]
                compliant_count = len([v for v in values if v <= target_value])
                compliance[metric_name] = (compliant_count / len(values)) * 100 if values else 0
        
        return compliance
    
    def _generate_recommendations(self, summary: Dict[str, Any], 
                                alert_history: List[PerformanceAlert]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Response time recommendations
        if "response_time" in summary:
            rt = summary["response_time"]
            if rt.get("p95_ms", 0) > 100:
                recommendations.append(
                    "Consider optimizing slow queries - P95 response time exceeds 100ms"
                )
            
            if rt.get("max_ms", 0) > 1000:
                recommendations.append(
                    "Investigate timeout issues - maximum response time exceeds 1 second"
                )
        
        # Error rate recommendations
        if "error_rate" in summary:
            er = summary["error_rate"]
            if er.get("mean_pct", 0) > 1.0:
                recommendations.append(
                    "Review error handling - error rate exceeds 1%"
                )
        
        # Alert-based recommendations
        critical_alerts = [a for a in alert_history if a.severity == AlertSeverity.CRITICAL]
        if len(critical_alerts) > 5:
            recommendations.append(
                "High number of critical alerts - review system stability"
            )
        
        return recommendations
    
    def _analyze_trends(self, hours: int) -> Dict[str, str]:
        """Analyze performance trends"""
        trends = {}
        
        # Get data for two periods to compare
        now = datetime.now()
        recent_start = now - timedelta(hours=hours//2)
        older_start = now - timedelta(hours=hours)
        
        recent_data = [
            entry for entry in self.performance_history 
            if recent_start <= entry["timestamp"] <= now
        ]
        
        older_data = [
            entry for entry in self.performance_history 
            if older_start <= entry["timestamp"] < recent_start
        ]
        
        if not recent_data or not older_data:
            return {"error": "Insufficient data for trend analysis"}
        
        # Compare response times
        recent_rt = statistics.mean([e["response_time_ms"] for e in recent_data])
        older_rt = statistics.mean([e["response_time_ms"] for e in older_data])
        
        if recent_rt < older_rt * 0.95:
            trends["response_time"] = "improving"
        elif recent_rt > older_rt * 1.05:
            trends["response_time"] = "degrading"
        else:
            trends["response_time"] = "stable"
        
        # Compare throughput
        recent_tp = statistics.mean([e["throughput_rps"] for e in recent_data])
        older_tp = statistics.mean([e["throughput_rps"] for e in older_data])
        
        if recent_tp > older_tp * 1.05:
            trends["throughput"] = "improving"
        elif recent_tp < older_tp * 0.95:
            trends["throughput"] = "degrading"
        else:
            trends["throughput"] = "stable"
        
        return trends
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        
        lower_index = int(index)
        upper_index = lower_index + 1
        
        if upper_index >= len(sorted_values):
            return sorted_values[-1]
        
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def _setup_default_thresholds(self):
        """Setup default performance thresholds"""
        
        # Response time thresholds
        self.add_threshold(PerformanceThreshold(
            metric_name="dgl.decision.processing_time",
            warning_threshold=50.0,
            critical_threshold=100.0,
            emergency_threshold=200.0,
            comparison_operator=">",
            evaluation_window_minutes=5
        ))
        
        # System CPU threshold
        self.add_threshold(PerformanceThreshold(
            metric_name="system.cpu.usage_percent",
            warning_threshold=70.0,
            critical_threshold=85.0,
            emergency_threshold=95.0,
            comparison_operator=">",
            evaluation_window_minutes=3
        ))
        
        # System memory threshold
        self.add_threshold(PerformanceThreshold(
            metric_name="system.memory.usage_percent",
            warning_threshold=80.0,
            critical_threshold=90.0,
            emergency_threshold=95.0,
            comparison_operator=">",
            evaluation_window_minutes=3
        ))
        
        # Set default SLA targets
        self.set_sla_target("response_time_ms", 50.0)  # 50ms SLA
        self.set_sla_target("error_rate_pct", 0.1)     # 0.1% error rate SLA


# Utility functions for performance monitoring

def create_performance_monitor(metrics_collector: MetricsCollector) -> PerformanceMonitor:
    """Create and configure performance monitor"""
    return PerformanceMonitor(metrics_collector)


def setup_alert_notifications(monitor: PerformanceMonitor):
    """Setup default alert notification handlers"""
    
    def log_alert(alert: PerformanceAlert):
        """Log alert to system logger"""
        level = logging.CRITICAL if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else logging.WARNING
        logger.log(level, f"PERFORMANCE ALERT: {alert.message}")
    
    def email_alert(alert: PerformanceAlert):
        """Send email alert (stub implementation)"""
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            logger.info(f"EMAIL ALERT: {alert.message}")
            # In production, would integrate with email service
    
    monitor.add_alert_callback(log_alert)
    monitor.add_alert_callback(email_alert)
    
    logger.info("Alert notification handlers configured")
