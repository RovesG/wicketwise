# Purpose: Production health monitoring and alerting system
# Author: WicketWise AI, Last Modified: 2024

import time
import threading
import requests
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import json
import socket


class HealthStatus(Enum):
    """Health check status types"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_type: str  # http, tcp, custom, process, disk, memory
    target: str  # URL, host:port, process name, etc.
    interval_seconds: int = 30
    timeout_seconds: int = 10
    retries: int = 3
    expected_status_code: int = 200
    expected_response: Optional[str] = None
    custom_check_function: Optional[Callable[[], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate health check configuration"""
        if self.check_type == "custom" and not self.custom_check_function:
            raise ValueError("Custom health check requires check function")


@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert service health to dictionary"""
        return {
            'service_name': self.service_name,
            'status': self.status.value,
            'last_check': self.last_check.isoformat(),
            'response_time_ms': self.response_time_ms,
            'error_message': self.error_message,
            'consecutive_failures': self.consecutive_failures,
            'uptime_percentage': self.uptime_percentage,
            'metadata': self.metadata
        }


@dataclass
class SystemHealth:
    """Overall system health status"""
    overall_status: HealthStatus
    services: Dict[str, ServiceHealth]
    system_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    alerts_active: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system health to dictionary"""
        return {
            'overall_status': self.overall_status.value,
            'services': {name: service.to_dict() for name, service in self.services.items()},
            'system_metrics': self.system_metrics,
            'timestamp': self.timestamp.isoformat(),
            'alerts_active': self.alerts_active
        }


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    service_name: str
    title: str
    description: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'service_name': self.service_name,
            'title': self.title,
            'description': self.description,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'metadata': self.metadata
        }


class AlertManager:
    """Alert management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Alert storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Configuration
        self.max_alerts = self.config.get('max_alerts', 1000)
        self.alert_cooldown_minutes = self.config.get('alert_cooldown_minutes', 5)
        
        # Thread safety
        self.lock = threading.RLock()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def create_alert(self, severity: AlertSeverity, service_name: str,
                    title: str, description: str, **metadata) -> Alert:
        """Create and process new alert"""
        with self.lock:
            alert_id = self._generate_alert_id()
            
            alert = Alert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                severity=severity,
                service_name=service_name,
                title=title,
                description=description,
                metadata=metadata
            )
            
            # Check for duplicate recent alerts
            if not self._is_duplicate_alert(alert):
                self.alerts[alert_id] = alert
                
                # Notify handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        self.logger.error(f"Alert handler error: {str(e)}")
                
                self.logger.warning(f"Alert created: {title} ({severity.value})")
            
            return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                self.logger.info(f"Alert resolved: {alert.title}")
                return True
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def _is_duplicate_alert(self, new_alert: Alert) -> bool:
        """Check if alert is duplicate of recent alert"""
        cutoff_time = datetime.now() - timedelta(minutes=self.alert_cooldown_minutes)
        
        for alert in self.alerts.values():
            if (alert.service_name == new_alert.service_name and
                alert.title == new_alert.title and
                alert.timestamp > cutoff_time and
                not alert.resolved):
                return True
        
        return False
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = str(int(time.time() * 1000))
        return f"alert_{timestamp}"


class HealthMonitor:
    """Production health monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert manager
        self.alert_manager = AlertManager(self.config.get('alert_config', {}))
        
        # Configuration
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.system_metrics_interval = self.config.get('system_metrics_interval', 60)
        self.health_check_timeout = self.config.get('health_check_timeout', 30)
        
        # Thresholds
        self.thresholds = {
            'cpu_percent': self.config.get('cpu_threshold', 80.0),
            'memory_percent': self.config.get('memory_threshold', 85.0),
            'disk_percent': self.config.get('disk_threshold', 90.0),
            'response_time_ms': self.config.get('response_time_threshold', 5000),
            'consecutive_failures': self.config.get('failure_threshold', 3)
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Monitoring threads
        self._monitoring_active = False
        self._health_check_threads: Dict[str, threading.Thread] = {}
        self._system_monitor_thread: Optional[threading.Thread] = None
        
        # Initialize default system checks
        self._initialize_default_checks()
    
    def _initialize_default_checks(self):
        """Initialize default system health checks"""
        # System resource checks
        self.add_health_check(HealthCheck(
            name="system_cpu",
            check_type="custom",
            target="system",
            interval_seconds=30,
            custom_check_function=self._check_system_cpu
        ))
        
        self.add_health_check(HealthCheck(
            name="system_memory",
            check_type="custom",
            target="system",
            interval_seconds=30,
            custom_check_function=self._check_system_memory
        ))
        
        self.add_health_check(HealthCheck(
            name="system_disk",
            check_type="custom",
            target="system",
            interval_seconds=60,
            custom_check_function=self._check_system_disk
        ))
    
    def add_health_check(self, health_check: HealthCheck):
        """Add health check"""
        with self.lock:
            self.health_checks[health_check.name] = health_check
            
            # Initialize service health
            self.service_health[health_check.name] = ServiceHealth(
                service_name=health_check.name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                response_time_ms=0.0
            )
            
            self.logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str) -> bool:
        """Remove health check"""
        with self.lock:
            if name in self.health_checks:
                del self.health_checks[name]
                if name in self.service_health:
                    del self.service_health[name]
                if name in self.health_history:
                    del self.health_history[name]
                
                self.logger.info(f"Removed health check: {name}")
                return True
            return False
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start health check threads
        for name, check in self.health_checks.items():
            thread = threading.Thread(
                target=self._health_check_loop,
                args=(name, check),
                daemon=True
            )
            thread.start()
            self._health_check_threads[name] = thread
        
        # Start system monitoring thread
        self._system_monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            daemon=True
        )
        self._system_monitor_thread.start()
        
        self.logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._monitoring_active = False
        
        # Wait for threads to finish
        for thread in self._health_check_threads.values():
            thread.join(timeout=5)
        
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5)
        
        self._health_check_threads.clear()
        self.logger.info("Stopped health monitoring")
    
    def _health_check_loop(self, name: str, check: HealthCheck):
        """Health check monitoring loop"""
        while self._monitoring_active:
            try:
                self._perform_health_check(name, check)
                time.sleep(check.interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in health check loop for {name}: {str(e)}")
                time.sleep(check.interval_seconds)
    
    def _perform_health_check(self, name: str, check: HealthCheck):
        """Perform individual health check"""
        start_time = time.time()
        
        try:
            if check.check_type == "http":
                success = self._check_http_health(check)
            elif check.check_type == "tcp":
                success = self._check_tcp_health(check)
            elif check.check_type == "process":
                success = self._check_process_health(check)
            elif check.check_type == "custom":
                success = self._check_custom_health(check)
            else:
                success = False
                self.logger.warning(f"Unknown health check type: {check.check_type}")
            
            response_time_ms = (time.time() - start_time) * 1000
            
            with self.lock:
                service_health = self.service_health[name]
                service_health.last_check = datetime.now()
                service_health.response_time_ms = response_time_ms
                
                if success:
                    service_health.status = HealthStatus.HEALTHY
                    service_health.consecutive_failures = 0
                    service_health.error_message = None
                else:
                    service_health.consecutive_failures += 1
                    
                    if service_health.consecutive_failures >= self.thresholds['consecutive_failures']:
                        service_health.status = HealthStatus.UNHEALTHY
                        
                        # Create alert
                        self.alert_manager.create_alert(
                            severity=AlertSeverity.ERROR,
                            service_name=name,
                            title=f"Service {name} is unhealthy",
                            description=f"Health check failed {service_health.consecutive_failures} times consecutively"
                        )
                    else:
                        service_health.status = HealthStatus.DEGRADED
                
                # Update uptime calculation
                self._update_uptime(name, success)
                
                # Store in history
                self.health_history[name].append({
                    'timestamp': datetime.now(),
                    'status': service_health.status.value,
                    'response_time_ms': response_time_ms,
                    'success': success
                })
                
        except Exception as e:
            self.logger.error(f"Error performing health check for {name}: {str(e)}")
            with self.lock:
                service_health = self.service_health[name]
                service_health.status = HealthStatus.UNKNOWN
                service_health.error_message = str(e)
    
    def _check_http_health(self, check: HealthCheck) -> bool:
        """Perform HTTP health check"""
        try:
            response = requests.get(
                check.target,
                timeout=check.timeout_seconds,
                allow_redirects=True
            )
            
            # Check status code
            if response.status_code != check.expected_status_code:
                return False
            
            # Check response content if specified
            if check.expected_response:
                if check.expected_response not in response.text:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"HTTP health check failed for {check.target}: {str(e)}")
            return False
    
    def _check_tcp_health(self, check: HealthCheck) -> bool:
        """Perform TCP health check"""
        try:
            host, port = check.target.split(':')
            port = int(port)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(check.timeout_seconds)
            
            result = sock.connect_ex((host, port))
            sock.close()
            
            return result == 0
            
        except Exception as e:
            self.logger.debug(f"TCP health check failed for {check.target}: {str(e)}")
            return False
    
    def _check_process_health(self, check: HealthCheck) -> bool:
        """Perform process health check"""
        try:
            process_name = check.target
            
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == process_name:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Process health check failed for {check.target}: {str(e)}")
            return False
    
    def _check_custom_health(self, check: HealthCheck) -> bool:
        """Perform custom health check"""
        try:
            if check.custom_check_function:
                return check.custom_check_function()
            return False
            
        except Exception as e:
            self.logger.debug(f"Custom health check failed for {check.name}: {str(e)}")
            return False
    
    def _check_system_cpu(self) -> bool:
        """Check system CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < self.thresholds['cpu_percent']
        except Exception:
            return False
    
    def _check_system_memory(self) -> bool:
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < self.thresholds['memory_percent']
        except Exception:
            return False
    
    def _check_system_disk(self) -> bool:
        """Check system disk usage"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            return disk_percent < self.thresholds['disk_percent']
        except Exception:
            return False
    
    def _update_uptime(self, service_name: str, success: bool):
        """Update service uptime percentage"""
        # Simple uptime calculation based on recent history
        if service_name in self.health_history:
            recent_checks = list(self.health_history[service_name])[-100:]  # Last 100 checks
            if recent_checks:
                successful_checks = sum(1 for check in recent_checks if check.get('success', False))
                uptime_percentage = (successful_checks / len(recent_checks)) * 100
                self.service_health[service_name].uptime_percentage = uptime_percentage
    
    def _system_monitor_loop(self):
        """System metrics monitoring loop"""
        while self._monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.system_metrics_interval)
            except Exception as e:
                self.logger.error(f"Error in system monitor loop: {str(e)}")
                time.sleep(self.system_metrics_interval)
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Store metrics
            self.system_metrics = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'disk_total_gb': disk.total / (1024**3),
                'disk_used_gb': disk.used / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check for threshold violations
            self._check_system_thresholds()
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _check_system_thresholds(self):
        """Check system metrics against thresholds"""
        if hasattr(self, 'system_metrics'):
            # CPU threshold
            if self.system_metrics['cpu_percent'] > self.thresholds['cpu_percent']:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.WARNING,
                    service_name="system",
                    title="High CPU Usage",
                    description=f"CPU usage is {self.system_metrics['cpu_percent']:.1f}%"
                )
            
            # Memory threshold
            if self.system_metrics['memory_percent'] > self.thresholds['memory_percent']:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.WARNING,
                    service_name="system",
                    title="High Memory Usage",
                    description=f"Memory usage is {self.system_metrics['memory_percent']:.1f}%"
                )
            
            # Disk threshold
            if self.system_metrics['disk_percent'] > self.thresholds['disk_percent']:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.ERROR,
                    service_name="system",
                    title="High Disk Usage",
                    description=f"Disk usage is {self.system_metrics['disk_percent']:.1f}%"
                )
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health status"""
        with self.lock:
            # Determine overall status
            statuses = [service.status for service in self.service_health.values()]
            
            if not statuses:
                overall_status = HealthStatus.UNKNOWN
            elif any(status == HealthStatus.UNHEALTHY for status in statuses):
                overall_status = HealthStatus.UNHEALTHY
            elif any(status == HealthStatus.DEGRADED for status in statuses):
                overall_status = HealthStatus.DEGRADED
            elif any(status == HealthStatus.UNKNOWN for status in statuses):
                overall_status = HealthStatus.UNKNOWN
            else:
                overall_status = HealthStatus.HEALTHY
            
            return SystemHealth(
                overall_status=overall_status,
                services=dict(self.service_health),
                system_metrics=getattr(self, 'system_metrics', {}),
                alerts_active=len(self.alert_manager.get_active_alerts())
            )
    
    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status for specific service"""
        with self.lock:
            return self.service_health.get(service_name)
    
    def get_health_history(self, service_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get health history for service"""
        with self.lock:
            if service_name not in self.health_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                entry for entry in self.health_history[service_name]
                if entry['timestamp'] > cutoff_time
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get health monitoring statistics"""
        with self.lock:
            active_alerts = self.alert_manager.get_active_alerts()
            
            return {
                'monitoring_active': self._monitoring_active,
                'total_health_checks': len(self.health_checks),
                'healthy_services': sum(1 for s in self.service_health.values() 
                                      if s.status == HealthStatus.HEALTHY),
                'unhealthy_services': sum(1 for s in self.service_health.values() 
                                        if s.status == HealthStatus.UNHEALTHY),
                'degraded_services': sum(1 for s in self.service_health.values() 
                                       if s.status == HealthStatus.DEGRADED),
                'active_alerts': len(active_alerts),
                'critical_alerts': sum(1 for a in active_alerts 
                                     if a.severity == AlertSeverity.CRITICAL),
                'system_metrics_available': hasattr(self, 'system_metrics')
            }
