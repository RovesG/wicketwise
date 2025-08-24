# Purpose: Comprehensive metrics collection for DGL system
# Author: WicketWise AI, Last Modified: 2024

"""
Metrics Collector

Collects, aggregates, and exports metrics for DGL system monitoring:
- System performance metrics
- Business logic metrics
- Custom application metrics
- Time-series data collection
- Metric aggregation and rollups
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Rate of events per time unit


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class MetricSeries:
    """Time series of metric values"""
    name: str
    metric_type: MetricType
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_value(self, value: Union[int, float], timestamp: datetime = None):
        """Add value to the series"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.values.append({
            "value": value,
            "timestamp": timestamp
        })
    
    def get_recent_values(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get values from the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [v for v in self.values if v["timestamp"] >= cutoff]
    
    def get_statistics(self, minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary of recent values"""
        recent_values = [v["value"] for v in self.get_recent_values(minutes)]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "stdev": statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
        }


class MetricsCollector:
    """
    Comprehensive metrics collection system for DGL
    
    Collects, stores, and aggregates metrics from various DGL components
    with support for real-time monitoring and historical analysis.
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to retain metric data in memory
        """
        self.retention_hours = retention_hours
        
        # Metric storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Metric callbacks for real-time processing
        self.metric_callbacks: List[Callable[[Metric], None]] = []
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Metrics collector initialized")
    
    def start(self):
        """Start background tasks"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
        logger.info("Metrics collector started")
    
    def stop(self):
        """Stop background tasks"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Metrics collector stopped")
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            self.counters[key] += value
            
            # Record in time series
            self._record_metric(name, value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric value"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            self.gauges[key] = value
            
            # Record in time series
            self._record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a value in a histogram"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            self.histograms[key].append(value)
            
            # Keep only recent values for memory efficiency
            if len(self.histograms[key]) > 10000:
                self.histograms[key] = self.histograms[key][-5000:]
            
            # Record in time series
            self._record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def record_timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timer/duration metric"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            self.timers[key].append(duration_ms)
            
            # Keep only recent values
            if len(self.timers[key]) > 10000:
                self.timers[key] = self.timers[key][-5000:]
            
            # Record in time series
            self._record_metric(name, duration_ms, MetricType.TIMER, tags)
    
    def time_function(self, name: str, tags: Dict[str, str] = None):
        """Decorator to time function execution"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    self.record_timer(name, duration_ms, tags)
            return wrapper
        return decorator
    
    def get_counter_value(self, name: str, tags: Dict[str, str] = None) -> float:
        """Get current counter value"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            return self.counters.get(key, 0.0)
    
    def get_gauge_value(self, name: str, tags: Dict[str, str] = None) -> Optional[float]:
        """Get current gauge value"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            return self.gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            values = self.histograms.get(key, [])
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0
            }
    
    def get_timer_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get timer statistics"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            values = self.timers.get(key, [])
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "p95_ms": self._percentile(values, 95),
                "p99_ms": self._percentile(values, 99),
                "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0
            }
    
    def get_metric_series(self, name: str, tags: Dict[str, str] = None) -> Optional[MetricSeries]:
        """Get time series for a metric"""
        with self._lock:
            key = self._get_metric_key(name, tags or {})
            return self.metrics.get(key)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {k: self.get_histogram_stats(k.split("|")[0], self._parse_tags(k)) 
                              for k in self.histograms.keys()},
                "timers": {k: self.get_timer_stats(k.split("|")[0], self._parse_tags(k)) 
                          for k in self.timers.keys()},
                "timestamp": datetime.now().isoformat()
            }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self._lock:
            # Export counters
            for key, value in self.counters.items():
                name, tags = self._parse_metric_key(key)
                tag_str = self._format_prometheus_tags(tags)
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{tag_str} {value}")
            
            # Export gauges
            for key, value in self.gauges.items():
                name, tags = self._parse_metric_key(key)
                tag_str = self._format_prometheus_tags(tags)
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{tag_str} {value}")
            
            # Export histogram summaries
            for key, values in self.histograms.items():
                if not values:
                    continue
                
                name, tags = self._parse_metric_key(key)
                tag_str = self._format_prometheus_tags(tags)
                
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count{tag_str} {len(values)}")
                lines.append(f"{name}_sum{tag_str} {sum(values)}")
                
                # Add percentile buckets
                for percentile in [50, 95, 99]:
                    p_value = self._percentile(values, percentile)
                    bucket_tags = self._format_prometheus_tags(tags, False)
                    lines.append(f"{name}_bucket{{le=\"{percentile}\"{bucket_tags}}} {p_value}")
        
        return "\n".join(lines)
    
    def add_metric_callback(self, callback: Callable[[Metric], None]):
        """Add callback for real-time metric processing"""
        self.metric_callbacks.append(callback)
    
    def remove_metric_callback(self, callback: Callable[[Metric], None]):
        """Remove metric callback"""
        if callback in self.metric_callbacks:
            self.metric_callbacks.remove(callback)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        import psutil
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_bytes": memory.used,
                "usage_percent": memory.percent
            },
            "disk": {
                "total_bytes": disk.total,
                "free_bytes": disk.free,
                "used_bytes": disk.used,
                "usage_percent": (disk.used / disk.total) * 100
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        }
    
    def record_dgl_metrics(self, decision_time_ms: float, rule_count: int, 
                          decision_type: str, confidence: float):
        """Record DGL-specific metrics"""
        # Decision processing time
        self.record_timer("dgl.decision.processing_time", decision_time_ms, 
                         {"decision_type": decision_type})
        
        # Rule evaluation count
        self.record_histogram("dgl.decision.rules_evaluated", rule_count,
                            {"decision_type": decision_type})
        
        # Decision confidence
        self.record_histogram("dgl.decision.confidence", confidence,
                            {"decision_type": decision_type})
        
        # Decision count
        self.increment_counter("dgl.decisions.total", 1.0, 
                             {"decision_type": decision_type})
    
    def record_governance_metrics(self, event_type: str, user: str, 
                                success: bool, processing_time_ms: float = None):
        """Record governance-specific metrics"""
        # Event count
        self.increment_counter("dgl.governance.events.total", 1.0,
                             {"event_type": event_type, "success": str(success).lower()})
        
        # Processing time if provided
        if processing_time_ms is not None:
            self.record_timer("dgl.governance.processing_time", processing_time_ms,
                            {"event_type": event_type})
        
        # User activity
        self.increment_counter("dgl.governance.user_activity", 1.0,
                             {"user": user, "event_type": event_type})
    
    def _record_metric(self, name: str, value: Union[int, float], 
                      metric_type: MetricType, tags: Dict[str, str] = None):
        """Record metric in time series and trigger callbacks"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        # Store in time series
        key = self._get_metric_key(name, tags or {})
        if key not in self.metrics:
            self.metrics[key] = MetricSeries(name, metric_type, tags=tags or {})
        
        self.metrics[key].add_value(value, metric.timestamp)
        
        # Trigger callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Error in metric callback: {str(e)}")
    
    def _get_metric_key(self, name: str, tags: Dict[str, str]) -> str:
        """Generate unique key for metric with tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"
    
    def _parse_metric_key(self, key: str) -> tuple:
        """Parse metric key back to name and tags"""
        if "|" not in key:
            return key, {}
        
        name, tag_str = key.split("|", 1)
        tags = {}
        
        if tag_str:
            for tag_pair in tag_str.split(","):
                if "=" in tag_pair:
                    k, v = tag_pair.split("=", 1)
                    tags[k] = v
        
        return name, tags
    
    def _parse_tags(self, key: str) -> Dict[str, str]:
        """Parse tags from metric key"""
        _, tags = self._parse_metric_key(key)
        return tags
    
    def _format_prometheus_tags(self, tags: Dict[str, str], include_braces: bool = True) -> str:
        """Format tags for Prometheus export"""
        if not tags:
            return ""
        
        tag_pairs = [f'{k}="{v}"' for k, v in sorted(tags.items())]
        tag_str = ",".join(tag_pairs)
        
        return f"{{{tag_str}}}" if include_braces else f",{tag_str}"
    
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
    
    async def _cleanup_old_metrics(self):
        """Background task to clean up old metric data"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
                
                with self._lock:
                    # Clean up time series data
                    for series in self.metrics.values():
                        # Remove old values
                        while (series.values and 
                               series.values[0]["timestamp"] < cutoff_time):
                            series.values.popleft()
                    
                    # Clean up histogram and timer data
                    for hist_list in self.histograms.values():
                        if len(hist_list) > 5000:
                            hist_list[:] = hist_list[-2500:]  # Keep recent half
                    
                    for timer_list in self.timers.values():
                        if len(timer_list) > 5000:
                            timer_list[:] = timer_list[-2500:]  # Keep recent half
                
                logger.debug("Cleaned up old metric data")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {str(e)}")


# Utility functions for metrics integration

def create_metrics_collector(retention_hours: int = 24) -> MetricsCollector:
    """Create and configure metrics collector"""
    return MetricsCollector(retention_hours=retention_hours)


def setup_default_metrics(collector: MetricsCollector):
    """Setup default system metrics collection"""
    
    def collect_system_metrics():
        """Collect and record system metrics"""
        try:
            sys_metrics = collector.get_system_metrics()
            
            # Record CPU metrics
            collector.set_gauge("system.cpu.usage_percent", sys_metrics["cpu"]["usage_percent"])
            collector.set_gauge("system.cpu.count", sys_metrics["cpu"]["count"])
            
            # Record memory metrics
            collector.set_gauge("system.memory.usage_percent", sys_metrics["memory"]["usage_percent"])
            collector.set_gauge("system.memory.used_bytes", sys_metrics["memory"]["used_bytes"])
            collector.set_gauge("system.memory.available_bytes", sys_metrics["memory"]["available_bytes"])
            
            # Record disk metrics
            collector.set_gauge("system.disk.usage_percent", sys_metrics["disk"]["usage_percent"])
            collector.set_gauge("system.disk.free_bytes", sys_metrics["disk"]["free_bytes"])
            
            # Record network metrics
            collector.set_gauge("system.network.bytes_sent", sys_metrics["network"]["bytes_sent"])
            collector.set_gauge("system.network.bytes_recv", sys_metrics["network"]["bytes_recv"])
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    # Collect system metrics every 30 seconds
    import threading
    
    def periodic_collection():
        while True:
            collect_system_metrics()
            time.sleep(30)
    
    thread = threading.Thread(target=periodic_collection, daemon=True)
    thread.start()
    
    logger.info("Default system metrics collection started")
