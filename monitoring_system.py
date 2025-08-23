#!/usr/bin/env python3
"""
Comprehensive Monitoring and Observability System
Prometheus metrics, structured logging, distributed tracing, alerting

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import asyncio
import logging
import time
import json
import traceback
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque
import psutil
import os

# Monitoring libraries
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# Local imports
from service_container import BaseService
from unified_configuration import get_config

logger = logging.getLogger(__name__)
config = get_config()

# ==================== MONITORING MODELS ====================

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """System alert"""
    id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    unit: str = ""

@dataclass
class HealthStatus:
    """Component health status"""
    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

# ==================== METRICS COLLECTOR ====================

class MetricsCollector:
    """Prometheus metrics collector"""
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Install with: pip install prometheus-client")
            return
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # System metrics
        self.http_requests_total = Counter(
            'wicketwise_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'wicketwise_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'wicketwise_active_connections',
            'Active WebSocket connections',
            registry=self.registry
        )
        
        # Business metrics
        self.matches_enriched_total = Counter(
            'wicketwise_matches_enriched_total',
            'Total matches enriched',
            ['status'],
            registry=self.registry
        )
        
        self.kg_queries_total = Counter(
            'wicketwise_kg_queries_total',
            'Total knowledge graph queries',
            ['query_type'],
            registry=self.registry
        )
        
        self.kg_query_duration = Histogram(
            'wicketwise_kg_query_duration_seconds',
            'Knowledge graph query duration',
            ['query_type'],
            registry=self.registry
        )
        
        # System resources
        self.system_cpu_usage = Gauge(
            'wicketwise_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'wicketwise_system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'wicketwise_system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Service health
        self.service_health = Gauge(
            'wicketwise_service_health',
            'Service health status (1=healthy, 0=unhealthy)',
            ['service_name'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'wicketwise_app',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': '2.0.0',
            'environment': config.environment.value,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        })
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_match_enrichment(self, status: str):
        """Record match enrichment metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.matches_enriched_total.labels(status=status).inc()
    
    def record_kg_query(self, query_type: str, duration: float):
        """Record knowledge graph query metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.kg_queries_total.labels(query_type=query_type).inc()
        self.kg_query_duration.labels(query_type=query_type).observe(duration)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.system_memory_usage.set(memory.used)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.system_disk_usage.set(disk_percent)
    
    def set_service_health(self, service_name: str, healthy: bool):
        """Set service health status"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.service_health.labels(service_name=service_name).set(1 if healthy else 0)
    
    def set_active_connections(self, count: int):
        """Set active WebSocket connections count"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.active_connections.set(count)

# ==================== STRUCTURED LOGGING ====================

class StructuredLogger:
    """Structured logging with context"""
    
    def __init__(self):
        if STRUCTLOG_AVAILABLE:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            self.logger = structlog.get_logger()
        else:
            self.logger = logging.getLogger("structured")
            logger.warning("structlog not available. Install with: pip install structlog")
    
    def log_event(self, level: str, event: str, **kwargs):
        """Log structured event"""
        if STRUCTLOG_AVAILABLE:
            getattr(self.logger, level)(event, **kwargs)
        else:
            # Fallback to standard logging
            log_data = {"event": event, **kwargs}
            getattr(self.logger, level)(json.dumps(log_data))
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance event"""
        self.log_event("info", "performance", 
                      operation=operation, 
                      duration_seconds=duration, 
                      **kwargs)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.log_event("error", "error_occurred",
                      error_type=type(error).__name__,
                      error_message=str(error),
                      traceback=traceback.format_exc(),
                      context=context or {})
    
    def log_user_action(self, user_id: str, action: str, **kwargs):
        """Log user action"""
        self.log_event("info", "user_action",
                      user_id=user_id,
                      action=action,
                      **kwargs)
    
    def log_business_event(self, event_type: str, **kwargs):
        """Log business event"""
        self.log_event("info", "business_event",
                      event_type=event_type,
                      **kwargs)

# ==================== HEALTH CHECKER ====================

class HealthChecker:
    """System health checker"""
    
    def __init__(self):
        self.health_status: Dict[str, HealthStatus] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.check_interval = 30  # seconds
        self.running = False
        self._task = None
    
    def register_health_check(self, component: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[component] = check_func
        logger.info(f"Registered health check for {component}")
    
    async def start(self):
        """Start health checking"""
        self.running = True
        self._task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checker started")
    
    async def stop(self):
        """Stop health checking"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")
    
    async def _health_check_loop(self):
        """Main health checking loop"""
        while self.running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _run_health_checks(self):
        """Run all registered health checks"""
        for component, check_func in self.health_checks.items():
            start_time = time.time()
            
            try:
                # Run health check
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                response_time = time.time() - start_time
                
                # Determine status
                if isinstance(result, dict):
                    status = result.get('status', 'unknown')
                    error_message = result.get('error')
                    metadata = {k: v for k, v in result.items() if k not in ['status', 'error']}
                else:
                    status = 'healthy' if result else 'unhealthy'
                    error_message = None
                    metadata = {}
                
                # Update health status
                self.health_status[component] = HealthStatus(
                    component=component,
                    status=status,
                    last_check=datetime.utcnow(),
                    response_time=response_time,
                    error_message=error_message,
                    metadata=metadata
                )
                
            except Exception as e:
                # Health check failed
                response_time = time.time() - start_time
                self.health_status[component] = HealthStatus(
                    component=component,
                    status='unhealthy',
                    last_check=datetime.utcnow(),
                    response_time=response_time,
                    error_message=str(e)
                )
                
                logger.warning(f"Health check failed for {component}: {e}")
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.health_status:
            return {"status": "unknown", "components": {}}
        
        healthy_count = sum(1 for h in self.health_status.values() if h.status == 'healthy')
        total_count = len(self.health_status)
        
        overall_status = "healthy"
        if healthy_count == 0:
            overall_status = "unhealthy"
        elif healthy_count < total_count:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "components": {
                name: {
                    "status": health.status,
                    "last_check": health.last_check.isoformat(),
                    "response_time": health.response_time,
                    "error": health.error_message
                }
                for name, health in self.health_status.items()
            }
        }

# ==================== ALERT MANAGER ====================

class AlertManager:
    """System alert manager"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.structured_logger = StructuredLogger()
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add alert rule"""
        self.alert_rules.append(rule)
    
    async def create_alert(self, alert: Alert):
        """Create new alert"""
        self.alerts[alert.id] = alert
        
        # Log alert
        self.structured_logger.log_event(
            alert.level.value,
            "alert_created",
            alert_id=alert.id,
            title=alert.title,
            message=alert.message,
            source=alert.source
        )
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            
            self.structured_logger.log_event(
                "info",
                "alert_resolved",
                alert_id=alert_id
            )
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    async def check_alert_rules(self, metrics: Dict[str, Any]):
        """Check alert rules against metrics"""
        for rule in self.alert_rules:
            try:
                await self._evaluate_rule(rule, metrics)
            except Exception as e:
                logger.error(f"Alert rule evaluation error: {e}")
    
    async def _evaluate_rule(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Evaluate a single alert rule"""
        rule_id = rule['id']
        condition = rule['condition']
        
        # Simple condition evaluation (would be more sophisticated in production)
        if condition['type'] == 'threshold':
            metric_name = condition['metric']
            threshold = condition['threshold']
            operator = condition['operator']
            
            if metric_name in metrics:
                value = metrics[metric_name]
                
                triggered = False
                if operator == 'gt' and value > threshold:
                    triggered = True
                elif operator == 'lt' and value < threshold:
                    triggered = True
                elif operator == 'eq' and value == threshold:
                    triggered = True
                
                if triggered:
                    alert = Alert(
                        id=f"{rule_id}_{int(time.time())}",
                        level=AlertLevel(rule['level']),
                        title=rule['title'],
                        message=rule['message'].format(value=value, threshold=threshold),
                        source=rule_id,
                        timestamp=datetime.utcnow(),
                        metadata={'metric': metric_name, 'value': value, 'threshold': threshold}
                    )
                    
                    await self.create_alert(alert)

# ==================== MONITORING SERVICE ====================

class MonitoringService(BaseService):
    """Comprehensive monitoring service"""
    
    def __init__(self):
        super().__init__("monitoring_service")
        self.metrics_collector = MetricsCollector()
        self.structured_logger = StructuredLogger()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        
        # Monitoring server
        self.metrics_server_port = config.performance.get('metrics_port', 9090)
        self.metrics_server_started = False
        
        # System monitoring
        self.system_monitor_task = None
        self.system_monitor_interval = 10  # seconds
    
    async def _start_implementation(self) -> None:
        """Start monitoring service"""
        # Start Prometheus metrics server
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self.metrics_server_port, registry=self.metrics_collector.registry)
                self.metrics_server_started = True
                logger.info(f"✅ Metrics server started on port {self.metrics_server_port}")
            except Exception as e:
                logger.warning(f"Failed to start metrics server: {e}")
        
        # Start health checker
        await self.health_checker.start()
        
        # Start system monitoring
        self.system_monitor_task = asyncio.create_task(self._system_monitor_loop())
        
        # Add default alert rules
        self._add_default_alert_rules()
        
        # Add default health checks
        self._add_default_health_checks()
        
        logger.info("✅ Monitoring service started")
    
    async def _stop_implementation(self) -> None:
        """Stop monitoring service"""
        # Stop health checker
        await self.health_checker.stop()
        
        # Stop system monitor
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Monitoring service stopped")
    
    async def _get_health_metrics(self) -> Dict[str, Any]:
        """Get monitoring service health metrics"""
        return {
            "metrics_server_running": self.metrics_server_started,
            "metrics_server_port": self.metrics_server_port,
            "health_checker_running": self.health_checker.running,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "monitored_components": len(self.health_checker.health_status)
        }
    
    async def _system_monitor_loop(self):
        """System monitoring loop"""
        while True:
            try:
                # Update system metrics
                self.metrics_collector.update_system_metrics()
                
                # Check alert rules
                system_metrics = {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }
                
                await self.alert_manager.check_alert_rules(system_metrics)
                
                await asyncio.sleep(self.system_monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitor error: {e}")
                await asyncio.sleep(5)
    
    def _add_default_alert_rules(self):
        """Add default alert rules"""
        rules = [
            {
                'id': 'high_cpu_usage',
                'condition': {
                    'type': 'threshold',
                    'metric': 'cpu_usage',
                    'operator': 'gt',
                    'threshold': 80
                },
                'level': 'warning',
                'title': 'High CPU Usage',
                'message': 'CPU usage is {value:.1f}%, above threshold of {threshold}%'
            },
            {
                'id': 'high_memory_usage',
                'condition': {
                    'type': 'threshold',
                    'metric': 'memory_usage',
                    'operator': 'gt',
                    'threshold': 85
                },
                'level': 'warning',
                'title': 'High Memory Usage',
                'message': 'Memory usage is {value:.1f}%, above threshold of {threshold}%'
            },
            {
                'id': 'high_disk_usage',
                'condition': {
                    'type': 'threshold',
                    'metric': 'disk_usage',
                    'operator': 'gt',
                    'threshold': 90
                },
                'level': 'critical',
                'title': 'High Disk Usage',
                'message': 'Disk usage is {value:.1f}%, above threshold of {threshold}%'
            }
        ]
        
        for rule in rules:
            self.alert_manager.add_alert_rule(rule)
    
    def _add_default_health_checks(self):
        """Add default health checks"""
        # System health check
        def system_health_check():
            try:
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                return {
                    'status': 'healthy',
                    'cpu_percent': cpu,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent
                }
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        
        self.health_checker.register_health_check('system', system_health_check)
    
    # Public methods for other services
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.metrics_collector.record_http_request(method, endpoint, status_code, duration)
    
    def record_business_event(self, event_type: str, **kwargs):
        """Record business event"""
        self.structured_logger.log_business_event(event_type, **kwargs)
    
    def log_user_action(self, user_id: str, action: str, **kwargs):
        """Log user action"""
        self.structured_logger.log_user_action(user_id, action, **kwargs)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.structured_logger.log_error(error, context)
    
    async def create_alert(self, level: str, title: str, message: str, source: str, **metadata):
        """Create system alert"""
        alert = Alert(
            id=f"{source}_{int(time.time())}",
            level=AlertLevel(level),
            title=title,
            message=message,
            source=source,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        await self.alert_manager.create_alert(alert)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        return self.health_checker.get_overall_health()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        alerts = self.alert_manager.get_active_alerts()
        return [asdict(alert) for alert in alerts]

# ==================== PERFORMANCE TRACER ====================

class PerformanceTracer:
    """Performance tracing context manager"""
    
    def __init__(self, monitoring_service: MonitoringService, operation: str, **labels):
        self.monitoring_service = monitoring_service
        self.operation = operation
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        # Record performance metric
        self.monitoring_service.structured_logger.log_performance(
            self.operation, 
            duration, 
            **self.labels
        )
        
        # Record error if exception occurred
        if exc_type:
            self.monitoring_service.log_error(exc_val, {
                'operation': self.operation,
                'duration': duration,
                **self.labels
            })

# Decorator for performance tracing
def trace_performance(operation: str, **labels):
    """Decorator for performance tracing"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get monitoring service from service container
            from service_container import get_container
            container = get_container()
            monitoring_service = container.resolve("monitoring_service")
            
            with PerformanceTracer(monitoring_service, operation, **labels):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    async def main():
        """Example usage of monitoring system"""
        
        # Initialize monitoring service
        monitoring = MonitoringService()
        
        try:
            await monitoring.start()
            
            # Example: Record metrics
            monitoring.record_http_request("GET", "/api/players", 200, 0.125)
            monitoring.record_business_event("player_query", player_id="123", query_type="stats")
            monitoring.log_user_action("user_001", "view_dashboard", page="main")
            
            # Example: Create alert
            await monitoring.create_alert("warning", "Test Alert", "This is a test alert", "test_system")
            
            # Get health status
            health = monitoring.get_system_health()
            logger.info(f"System health: {health}")
            
            # Get active alerts
            alerts = monitoring.get_active_alerts()
            logger.info(f"Active alerts: {len(alerts)}")
            
            # Wait a bit to see metrics
            await asyncio.sleep(5)
            
        finally:
            await monitoring.stop()
    
    # Run example
    asyncio.run(main())
