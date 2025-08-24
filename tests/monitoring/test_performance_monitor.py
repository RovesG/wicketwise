# Purpose: Unit tests for performance monitoring system
# Author: WicketWise AI, Last Modified: 2024

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crickformers.monitoring.performance_monitor import (
    PerformanceMonitor,
    MetricsCollector,
    SystemResourceMonitor,
    PerformanceAlert,
    AlertSeverity,
    SystemMetrics,
    measure_time,
    create_performance_context
)


class TestSystemMetrics:
    """Test suite for SystemMetrics data structure"""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation and basic properties"""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=45.2,
            memory_percent=67.8,
            memory_used_mb=2048.5,
            memory_available_mb=1024.3,
            disk_usage_percent=78.9,
            disk_free_gb=125.7,
            network_bytes_sent=1024000,
            network_bytes_recv=2048000,
            active_connections=15
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 45.2
        assert metrics.memory_percent == 67.8
        assert metrics.memory_used_mb == 2048.5
        assert metrics.memory_available_mb == 1024.3
        assert metrics.disk_usage_percent == 78.9
        assert metrics.disk_free_gb == 125.7
        assert metrics.network_bytes_sent == 1024000
        assert metrics.network_bytes_recv == 2048000
        assert metrics.active_connections == 15
    
    def test_system_metrics_to_dict(self):
        """Test SystemMetrics dictionary conversion"""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=45.2,
            memory_percent=67.8,
            memory_used_mb=2048.5,
            memory_available_mb=1024.3,
            disk_usage_percent=78.9,
            disk_free_gb=125.7,
            network_bytes_sent=1024000,
            network_bytes_recv=2048000,
            active_connections=15
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['timestamp'] == timestamp.isoformat()
        assert metrics_dict['cpu_percent'] == 45.2
        assert metrics_dict['memory_percent'] == 67.8
        assert metrics_dict['disk_usage_percent'] == 78.9
        assert metrics_dict['network_bytes_sent'] == 1024000
        assert metrics_dict['active_connections'] == 15


class TestPerformanceAlert:
    """Test suite for PerformanceAlert data structure"""
    
    def test_alert_creation(self):
        """Test PerformanceAlert creation"""
        timestamp = datetime.now()
        alert = PerformanceAlert(
            alert_id="test_alert_001",
            severity=AlertSeverity.HIGH,
            component="system",
            metric="cpu_percent",
            current_value=85.5,
            threshold_value=80.0,
            message="High CPU usage detected",
            timestamp=timestamp
        )
        
        assert alert.alert_id == "test_alert_001"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.component == "system"
        assert alert.metric == "cpu_percent"
        assert alert.current_value == 85.5
        assert alert.threshold_value == 80.0
        assert alert.message == "High CPU usage detected"
        assert alert.timestamp == timestamp
        assert not alert.resolved
        assert alert.resolution_time is None
    
    def test_alert_to_dict(self):
        """Test PerformanceAlert dictionary conversion"""
        timestamp = datetime.now()
        alert = PerformanceAlert(
            alert_id="test_alert_001",
            severity=AlertSeverity.CRITICAL,
            component="agent",
            metric="response_time_ms",
            current_value=5500.0,
            threshold_value=5000.0,
            message="Critical response time",
            timestamp=timestamp
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict['alert_id'] == "test_alert_001"
        assert alert_dict['severity'] == "critical"
        assert alert_dict['component'] == "agent"
        assert alert_dict['metric'] == "response_time_ms"
        assert alert_dict['current_value'] == 5500.0
        assert alert_dict['threshold_value'] == 5000.0
        assert alert_dict['message'] == "Critical response time"
        assert alert_dict['timestamp'] == timestamp.isoformat()
        assert not alert_dict['resolved']
        assert alert_dict['resolution_time'] is None


class TestMetricsCollector:
    """Test suite for MetricsCollector"""
    
    @pytest.fixture
    def collector(self):
        """Create MetricsCollector instance"""
        return MetricsCollector(max_history=100)
    
    def test_collector_initialization(self, collector):
        """Test MetricsCollector initialization"""
        assert collector.max_history == 100
        assert len(collector.metrics_history) == 0
        assert len(collector.custom_metrics) == 0
    
    def test_add_custom_metric(self, collector):
        """Test adding custom metrics"""
        collector.add_metric("test_metric", 42.5)
        collector.add_metric("test_metric", 38.2)
        collector.add_metric("another_metric", 100.0)
        
        assert len(collector.custom_metrics["test_metric"]) == 2
        assert len(collector.custom_metrics["another_metric"]) == 1
        assert collector.custom_metrics["test_metric"][0]['value'] == 42.5
        assert collector.custom_metrics["test_metric"][1]['value'] == 38.2
        assert collector.custom_metrics["another_metric"][0]['value'] == 100.0
    
    def test_add_system_metrics(self, collector):
        """Test adding system metrics"""
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=45.2,
            memory_percent=67.8,
            memory_used_mb=2048.5,
            memory_available_mb=1024.3,
            disk_usage_percent=78.9,
            disk_free_gb=125.7,
            network_bytes_sent=1024000,
            network_bytes_recv=2048000,
            active_connections=15
        )
        
        collector.add_system_metrics(metrics)
        
        assert len(collector.metrics_history) == 1
        assert collector.metrics_history[0] == metrics
    
    def test_get_metric_stats(self, collector):
        """Test getting metric statistics"""
        # Add test data
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            collector.add_metric("test_metric", value)
        
        stats = collector.get_metric_stats("test_metric")
        
        assert stats['count'] == 5
        assert stats['min'] == 10.0
        assert stats['max'] == 50.0
        assert stats['mean'] == 30.0
        assert stats['median'] == 30.0
        assert stats['latest'] == 50.0
        assert stats['std_dev'] > 0
    
    def test_get_metric_stats_with_time_window(self, collector):
        """Test getting metric statistics with time window"""
        # Add old metric
        old_time = datetime.now() - timedelta(hours=2)
        collector.add_metric("test_metric", 10.0, old_time)
        
        # Add recent metrics
        recent_time = datetime.now() - timedelta(minutes=5)
        collector.add_metric("test_metric", 20.0, recent_time)
        collector.add_metric("test_metric", 30.0)
        
        # Get stats with 1-hour window
        stats = collector.get_metric_stats("test_metric", timedelta(hours=1))
        
        assert stats['count'] == 2  # Only recent metrics
        assert stats['min'] == 20.0
        assert stats['max'] == 30.0
    
    def test_get_system_stats(self, collector):
        """Test getting system statistics"""
        # Add test system metrics
        for i in range(5):
            timestamp = datetime.now() - timedelta(minutes=i)
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_percent=40.0 + i * 5,
                memory_percent=60.0 + i * 2,
                memory_used_mb=2000.0,
                memory_available_mb=1000.0,
                disk_usage_percent=70.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000000,
                network_bytes_recv=2000000,
                active_connections=10
            )
            collector.add_system_metrics(metrics)
        
        stats = collector.get_system_stats()
        
        assert 'cpu_percent' in stats
        assert 'memory_percent' in stats
        assert 'disk_usage_percent' in stats
        
        cpu_stats = stats['cpu_percent']
        assert cpu_stats['min'] == 40.0
        assert cpu_stats['max'] == 60.0
        assert cpu_stats['mean'] == 50.0


class TestSystemResourceMonitor:
    """Test suite for SystemResourceMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create SystemResourceMonitor instance"""
        return SystemResourceMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test SystemResourceMonitor initialization"""
        assert not monitor._monitoring
        assert monitor._monitor_thread is None
        assert isinstance(monitor.metrics_collector, MetricsCollector)
    
    @patch('crickformers.monitoring.performance_monitor.psutil')
    def test_collect_system_metrics(self, mock_psutil, monitor):
        """Test system metrics collection"""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 45.2
        
        mock_memory = Mock()
        mock_memory.percent = 67.8
        mock_memory.used = 2048 * 1024 * 1024  # 2048 MB
        mock_memory.available = 1024 * 1024 * 1024  # 1024 MB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.used = 80 * 1024 * 1024 * 1024  # 80 GB
        mock_disk.total = 100 * 1024 * 1024 * 1024  # 100 GB
        mock_disk.free = 20 * 1024 * 1024 * 1024  # 20 GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_network = Mock()
        mock_network.bytes_sent = 1024000
        mock_network.bytes_recv = 2048000
        mock_psutil.net_io_counters.return_value = mock_network
        
        mock_psutil.net_connections.return_value = [Mock()] * 15
        
        # Collect metrics
        metrics = monitor._collect_system_metrics()
        
        assert metrics.cpu_percent == 45.2
        assert metrics.memory_percent == 67.8
        assert abs(metrics.memory_used_mb - 2048) < 1
        assert abs(metrics.memory_available_mb - 1024) < 1
        assert metrics.disk_usage_percent == 80.0
        assert abs(metrics.disk_free_gb - 20) < 1
        assert metrics.network_bytes_sent == 1024000
        assert metrics.network_bytes_recv == 2048000
        assert metrics.active_connections == 15
    
    def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring"""
        # Start monitoring
        monitor.start_monitoring(interval=0.1)
        assert monitor._monitoring
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor._monitoring
        
        # Wait for thread to finish
        time.sleep(0.1)
        assert not monitor._monitor_thread.is_alive()


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create PerformanceMonitor instance"""
        return PerformanceMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test PerformanceMonitor initialization"""
        assert isinstance(monitor.system_monitor, SystemResourceMonitor)
        assert isinstance(monitor.metrics_collector, MetricsCollector)
        assert len(monitor.alerts) == 0
        assert len(monitor.alert_handlers) == 0
        assert 'cpu_percent' in monitor.thresholds
        assert 'memory_percent' in monitor.thresholds
        assert not monitor._monitoring
    
    def test_add_alert_handler(self, monitor):
        """Test adding alert handlers"""
        handler1 = Mock()
        handler2 = Mock()
        
        monitor.add_alert_handler(handler1)
        monitor.add_alert_handler(handler2)
        
        assert len(monitor.alert_handlers) == 2
        assert handler1 in monitor.alert_handlers
        assert handler2 in monitor.alert_handlers
    
    def test_record_operation_time(self, monitor):
        """Test recording operation execution time"""
        # Record normal operation
        monitor.record_operation_time("test_operation", 500.0)
        
        stats = monitor.metrics_collector.get_metric_stats("test_operation_duration_ms")
        assert stats['count'] == 1
        assert stats['latest'] == 500.0
        
        # Record slow operation (should trigger alert)
        monitor.record_operation_time("slow_operation", 6000.0)
        
        # Check that alert was created
        critical_alerts = [a for a in monitor.alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) > 0
        assert critical_alerts[0].component == "slow_operation"
        assert critical_alerts[0].metric == "response_time_ms"
    
    def test_alert_creation_and_handling(self, monitor):
        """Test alert creation and handler invocation"""
        handler_mock = Mock()
        monitor.add_alert_handler(handler_mock)
        
        # Trigger a critical response time alert
        monitor.record_operation_time("critical_operation", 7000.0)
        
        # Verify alert was created
        assert len(monitor.alerts) > 0
        critical_alert = monitor.alerts[-1]
        assert critical_alert.severity == AlertSeverity.CRITICAL
        assert critical_alert.current_value == 7000.0
        
        # Verify handler was called
        handler_mock.assert_called_once()
        called_alert = handler_mock.call_args[0][0]
        assert called_alert.severity == AlertSeverity.CRITICAL
    
    def test_performance_summary(self, monitor):
        """Test getting performance summary"""
        # Add some test data
        monitor.record_operation_time("test_op", 1000.0)
        monitor.record_operation_time("test_op", 1200.0)
        
        summary = monitor.get_performance_summary()
        
        assert 'timestamp' in summary
        assert 'system_metrics' in summary
        assert 'custom_metrics' in summary
        assert 'recent_alerts' in summary
        assert 'total_alerts' in summary
        assert 'monitoring_active' in summary
        
        assert summary['monitoring_active'] == monitor._monitoring
        assert summary['total_alerts'] == len(monitor.alerts)
    
    def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping comprehensive monitoring"""
        # Start monitoring
        monitor.start_monitoring(system_interval=0.1, alert_interval=0.1)
        assert monitor._monitoring
        assert monitor.system_monitor._monitoring
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor._monitoring
        assert not monitor.system_monitor._monitoring


class TestPerformanceDecorators:
    """Test suite for performance measurement utilities"""
    
    def test_measure_time_decorator(self):
        """Test the measure_time decorator"""
        @measure_time
        def test_function():
            time.sleep(0.1)
            return "test_result"
        
        # Mock performance monitor
        mock_monitor = Mock()
        test_function._performance_monitor = mock_monitor
        
        result = test_function()
        
        assert result == "test_result"
        mock_monitor.record_operation_time.assert_called_once()
        
        # Check that duration was recorded (should be around 100ms)
        call_args = mock_monitor.record_operation_time.call_args
        assert call_args[0][0] == "test_function"
        assert 90 <= call_args[0][1] <= 150  # Allow some variance
    
    def test_performance_context_manager(self):
        """Test the performance context manager"""
        mock_monitor = Mock()
        PerformanceContext = create_performance_context()
        
        with PerformanceContext("test_operation", mock_monitor):
            time.sleep(0.05)
        
        mock_monitor.record_operation_time.assert_called_once()
        call_args = mock_monitor.record_operation_time.call_args
        assert call_args[0][0] == "test_operation"
        assert 40 <= call_args[0][1] <= 80  # Around 50ms with variance


def run_performance_monitor_tests():
    """Run all performance monitor tests"""
    print("🔍 Running Performance Monitor Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("System Metrics", TestSystemMetrics),
        ("Performance Alerts", TestPerformanceAlert),
        ("Metrics Collector", TestMetricsCollector),
        ("System Resource Monitor", TestSystemResourceMonitor),
        ("Performance Monitor", TestPerformanceMonitor),
        ("Performance Utilities", TestPerformanceDecorators)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category_name, test_class in test_categories:
        print(f"\n📊 {category_name}")
        print("-" * 30)
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        category_passed = 0
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                
                # Handle fixtures
                if hasattr(test_instance, test_method):
                    method = getattr(test_instance, test_method)
                    
                    # Check if method needs fixtures
                    import inspect
                    sig = inspect.signature(method)
                    
                    if 'collector' in sig.parameters:
                        collector = MetricsCollector(max_history=100)
                        method(collector)
                    elif 'monitor' in sig.parameters:
                        if test_class == TestSystemResourceMonitor:
                            monitor = SystemResourceMonitor()
                        else:
                            monitor = PerformanceMonitor()
                        method(monitor)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Performance Monitor Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_performance_monitor_tests()
    exit(0 if success else 1)
