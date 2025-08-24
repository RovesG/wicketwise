# Purpose: System performance monitoring and metrics collection
# Author: WicketWise AI, Last Modified: 2024

import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import json


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    alert_id: str
    severity: AlertSeverity
    component: str
    metric: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'component': self.component,
            'metric': self.metric,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'active_connections': self.active_connections
        }


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
        
    def add_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Add a custom metric"""
        if timestamp is None:
            timestamp = datetime.now()
            
        with self.lock:
            self.custom_metrics[name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def add_system_metrics(self, metrics: SystemMetrics):
        """Add system metrics to history"""
        with self.lock:
            self.metrics_history.append(metrics)
    
    def get_metric_stats(self, name: str, time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistics for a custom metric"""
        with self.lock:
            if name not in self.custom_metrics:
                return {}
            
            metrics = list(self.custom_metrics[name])
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m['timestamp'] >= cutoff_time]
            
            if not metrics:
                return {}
            
            values = [m['value'] for m in metrics]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'latest': values[-1] if values else 0.0
            }
    
    def get_system_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get system metrics statistics"""
        with self.lock:
            metrics = list(self.metrics_history)
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if not metrics:
                return {}
            
            # Calculate statistics for each metric
            stats = {}
            metric_fields = ['cpu_percent', 'memory_percent', 'disk_usage_percent']
            
            for field in metric_fields:
                values = [getattr(m, field) for m in metrics]
                if values:
                    stats[field] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': statistics.mean(values),
                        'latest': values[-1]
                    }
            
            return stats


class SystemResourceMonitor:
    """Monitors system resources (CPU, memory, disk, network)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self.metrics_collector = MetricsCollector()
        
    def start_monitoring(self, interval: float = 5.0):
        """Start system resource monitoring"""
        if self._monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        self.logger.info("Stopped system monitoring")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_collector.add_system_metrics(metrics)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Network I/O
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # Active connections
        try:
            active_connections = len(psutil.net_connections())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            active_connections = 0
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            active_connections=active_connections
        )
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics snapshot"""
        return self._collect_system_metrics()
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[SystemMetrics]:
        """Get system metrics history"""
        with self.metrics_collector.lock:
            history = list(self.metrics_collector.metrics_history)
            if limit:
                history = history[-limit:]
            return history


class PerformanceMonitor:
    """Main performance monitoring coordinator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.system_monitor = SystemResourceMonitor()
        self.metrics_collector = MetricsCollector()
        self.alerts: List[PerformanceAlert] = []
        self.alert_handlers: List[Callable[[PerformanceAlert], None]] = []
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': {'high': 80.0, 'critical': 95.0},
            'memory_percent': {'high': 85.0, 'critical': 95.0},
            'disk_usage_percent': {'high': 85.0, 'critical': 95.0},
            'response_time_ms': {'high': 2000.0, 'critical': 5000.0}
        }
        
        # Update thresholds from config
        if 'thresholds' in self.config:
            self.thresholds.update(self.config['thresholds'])
        
        self._monitoring = False
        self._alert_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, system_interval: float = 5.0, alert_interval: float = 10.0):
        """Start comprehensive performance monitoring"""
        self.logger.info("Starting performance monitoring...")
        
        # Start system resource monitoring
        self.system_monitor.start_monitoring(system_interval)
        
        # Start alert monitoring
        self._monitoring = True
        self._alert_thread = threading.Thread(
            target=self._alert_loop,
            args=(alert_interval,),
            daemon=True
        )
        self._alert_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.logger.info("Stopping performance monitoring...")
        
        # Stop system monitoring
        self.system_monitor.stop_monitoring()
        
        # Stop alert monitoring
        self._monitoring = False
        if self._alert_thread:
            self._alert_thread.join(timeout=10)
        
        self.logger.info("Performance monitoring stopped")
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def record_operation_time(self, operation: str, duration_ms: float):
        """Record operation execution time"""
        self.metrics_collector.add_metric(f"{operation}_duration_ms", duration_ms)
        
        # Check for performance alerts
        if duration_ms > self.thresholds['response_time_ms']['critical']:
            self._create_alert(
                AlertSeverity.CRITICAL,
                operation,
                'response_time_ms',
                duration_ms,
                self.thresholds['response_time_ms']['critical'],
                f"Critical response time for {operation}: {duration_ms:.1f}ms"
            )
        elif duration_ms > self.thresholds['response_time_ms']['high']:
            self._create_alert(
                AlertSeverity.HIGH,
                operation,
                'response_time_ms',
                duration_ms,
                self.thresholds['response_time_ms']['high'],
                f"High response time for {operation}: {duration_ms:.1f}ms"
            )
    
    def _alert_loop(self, interval: float):
        """Main alert monitoring loop"""
        while self._monitoring:
            try:
                self._check_system_alerts()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in alert loop: {str(e)}")
                time.sleep(interval)
    
    def _check_system_alerts(self):
        """Check system metrics for alert conditions"""
        try:
            current_metrics = self.system_monitor.get_current_metrics()
            
            # Check CPU usage
            self._check_threshold_alert(
                'system',
                'cpu_percent',
                current_metrics.cpu_percent,
                f"High CPU usage: {current_metrics.cpu_percent:.1f}%"
            )
            
            # Check memory usage
            self._check_threshold_alert(
                'system',
                'memory_percent',
                current_metrics.memory_percent,
                f"High memory usage: {current_metrics.memory_percent:.1f}%"
            )
            
            # Check disk usage
            self._check_threshold_alert(
                'system',
                'disk_usage_percent',
                current_metrics.disk_usage_percent,
                f"High disk usage: {current_metrics.disk_usage_percent:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error checking system alerts: {str(e)}")
    
    def _check_threshold_alert(self, component: str, metric: str, value: float, message: str):
        """Check if a metric exceeds thresholds"""
        if metric not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric]
        
        if value >= thresholds['critical']:
            self._create_alert(AlertSeverity.CRITICAL, component, metric, value, thresholds['critical'], message)
        elif value >= thresholds['high']:
            self._create_alert(AlertSeverity.HIGH, component, metric, value, thresholds['high'], message)
    
    def _create_alert(self, severity: AlertSeverity, component: str, metric: str, 
                     current_value: float, threshold_value: float, message: str):
        """Create and handle a performance alert"""
        alert = PerformanceAlert(
            alert_id=f"{component}_{metric}_{int(time.time())}",
            severity=severity,
            component=component,
            metric=metric,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {str(e)}")
        
        # Log the alert
        log_level = {
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.HIGH: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR
        }[severity]
        
        self.logger.log(log_level, f"Performance Alert [{severity.value.upper()}]: {message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        # System metrics summary
        system_stats = self.system_monitor.metrics_collector.get_system_stats(
            time_window=timedelta(hours=1)
        )
        
        # Custom metrics summary
        custom_stats = {}
        for metric_name in self.metrics_collector.custom_metrics.keys():
            custom_stats[metric_name] = self.metrics_collector.get_metric_stats(
                metric_name, time_window=timedelta(hours=1)
            )
        
        # Recent alerts
        recent_alerts = [
            alert.to_dict() for alert in self.alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': system_stats,
            'custom_metrics': custom_stats,
            'recent_alerts': recent_alerts,
            'total_alerts': len(self.alerts),
            'monitoring_active': self._monitoring
        }


# Utility functions for performance measurement
def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Log performance metric if monitor is available
        try:
            # This would be injected or configured in production
            monitor = getattr(wrapper, '_performance_monitor', None)
            if monitor:
                monitor.record_operation_time(func.__name__, duration_ms)
        except Exception:
            pass  # Silently ignore monitoring errors
        
        return result
    return wrapper


def create_performance_context():
    """Create a performance monitoring context manager"""
    class PerformanceContext:
        def __init__(self, operation_name: str, monitor: Optional[PerformanceMonitor] = None):
            self.operation_name = operation_name
            self.monitor = monitor
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time and self.monitor:
                duration_ms = (time.time() - self.start_time) * 1000
                self.monitor.record_operation_time(self.operation_name, duration_ms)
    
    return PerformanceContext
