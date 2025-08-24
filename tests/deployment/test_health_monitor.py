# Purpose: Unit tests for production health monitoring and alerting
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

from crickformers.deployment.health_monitor import (
    HealthMonitor,
    HealthCheck,
    ServiceHealth,
    SystemHealth,
    AlertManager,
    Alert,
    HealthStatus,
    AlertSeverity
)


class TestHealthCheck:
    """Test suite for HealthCheck configuration"""
    
    def test_health_check_creation(self):
        """Test HealthCheck creation and basic properties"""
        check = HealthCheck(
            name="api_health",
            check_type="http",
            target="http://localhost:5001/api/health",
            interval_seconds=30,
            timeout_seconds=10,
            retries=3,
            expected_status_code=200
        )
        
        assert check.name == "api_health"
        assert check.check_type == "http"
        assert check.target == "http://localhost:5001/api/health"
        assert check.interval_seconds == 30
        assert check.timeout_seconds == 10
        assert check.retries == 3
        assert check.expected_status_code == 200
    
    def test_health_check_custom_function(self):
        """Test HealthCheck with custom function"""
        def custom_check():
            return True
        
        check = HealthCheck(
            name="custom_check",
            check_type="custom",
            target="custom_service",
            custom_check_function=custom_check
        )
        
        assert check.name == "custom_check"
        assert check.check_type == "custom"
        assert check.custom_check_function == custom_check
    
    def test_health_check_invalid_custom(self):
        """Test HealthCheck validation for custom type without function"""
        with pytest.raises(ValueError, match="Custom health check requires check function"):
            HealthCheck(
                name="invalid_custom",
                check_type="custom",
                target="service"
            )


class TestServiceHealth:
    """Test suite for ServiceHealth status"""
    
    def test_service_health_creation(self):
        """Test ServiceHealth creation and basic properties"""
        last_check = datetime.now()
        
        service_health = ServiceHealth(
            service_name="api_service",
            status=HealthStatus.HEALTHY,
            last_check=last_check,
            response_time_ms=150.5,
            consecutive_failures=0,
            uptime_percentage=99.9
        )
        
        assert service_health.service_name == "api_service"
        assert service_health.status == HealthStatus.HEALTHY
        assert service_health.last_check == last_check
        assert service_health.response_time_ms == 150.5
        assert service_health.consecutive_failures == 0
        assert service_health.uptime_percentage == 99.9
        assert service_health.error_message is None
    
    def test_service_health_to_dict(self):
        """Test ServiceHealth dictionary conversion"""
        last_check = datetime.now()
        
        service_health = ServiceHealth(
            service_name="database",
            status=HealthStatus.DEGRADED,
            last_check=last_check,
            response_time_ms=2500.0,
            error_message="Connection timeout",
            consecutive_failures=2,
            uptime_percentage=95.5
        )
        
        health_dict = service_health.to_dict()
        
        assert health_dict['service_name'] == "database"
        assert health_dict['status'] == "degraded"
        assert health_dict['last_check'] == last_check.isoformat()
        assert health_dict['response_time_ms'] == 2500.0
        assert health_dict['error_message'] == "Connection timeout"
        assert health_dict['consecutive_failures'] == 2
        assert health_dict['uptime_percentage'] == 95.5


class TestAlertManager:
    """Test suite for AlertManager"""
    
    @pytest.fixture
    def alert_manager(self):
        """Create AlertManager instance"""
        config = {
            'max_alerts': 100,
            'alert_cooldown_minutes': 5
        }
        return AlertManager(config)
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test AlertManager initialization"""
        assert len(alert_manager.alerts) == 0
        assert len(alert_manager.alert_handlers) == 0
        assert alert_manager.max_alerts == 100
        assert alert_manager.alert_cooldown_minutes == 5
    
    def test_create_alert(self, alert_manager):
        """Test alert creation"""
        alert = alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            service_name="test_service",
            title="Test Alert",
            description="This is a test alert",
            test_metadata="test_value"
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.service_name == "test_service"
        assert alert.title == "Test Alert"
        assert alert.description == "This is a test alert"
        assert alert.metadata["test_metadata"] == "test_value"
        assert alert.resolved is False
        
        # Alert should be stored
        assert alert.alert_id in alert_manager.alerts
    
    def test_resolve_alert(self, alert_manager):
        """Test alert resolution"""
        alert = alert_manager.create_alert(
            severity=AlertSeverity.ERROR,
            service_name="test_service",
            title="Error Alert",
            description="Test error"
        )
        
        # Resolve alert
        success = alert_manager.resolve_alert(alert.alert_id)
        assert success is True
        
        # Check alert is resolved
        resolved_alert = alert_manager.alerts[alert.alert_id]
        assert resolved_alert.resolved is True
        assert resolved_alert.resolved_at is not None
    
    def test_resolve_nonexistent_alert(self, alert_manager):
        """Test resolving non-existent alert"""
        success = alert_manager.resolve_alert("nonexistent_alert_id")
        assert success is False
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts"""
        # Create alerts
        alert1 = alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            service_name="service1",
            title="Alert 1",
            description="First alert"
        )
        
        alert2 = alert_manager.create_alert(
            severity=AlertSeverity.ERROR,
            service_name="service2",
            title="Alert 2",
            description="Second alert"
        )
        
        # Resolve one alert
        alert_manager.resolve_alert(alert1.alert_id)
        
        # Get active alerts
        active_alerts = alert_manager.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == alert2.alert_id
        assert active_alerts[0].resolved is False
    
    def test_alert_handler(self, alert_manager):
        """Test alert handler functionality"""
        handler_called = []
        
        def test_handler(alert):
            handler_called.append(alert)
        
        # Add handler
        alert_manager.add_alert_handler(test_handler)
        
        # Create alert
        alert = alert_manager.create_alert(
            severity=AlertSeverity.CRITICAL,
            service_name="critical_service",
            title="Critical Alert",
            description="Critical issue"
        )
        
        # Handler should have been called
        assert len(handler_called) == 1
        assert handler_called[0] == alert
    
    def test_duplicate_alert_prevention(self, alert_manager):
        """Test duplicate alert prevention"""
        # Create first alert
        alert1 = alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            service_name="duplicate_service",
            title="Duplicate Alert",
            description="First occurrence"
        )
        
        initial_count = len(alert_manager.alerts)
        
        # Create duplicate alert (same service and title)
        alert2 = alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            service_name="duplicate_service",
            title="Duplicate Alert",
            description="Second occurrence"
        )
        
        # Should not create new alert due to cooldown
        assert len(alert_manager.alerts) == initial_count


class TestHealthMonitor:
    """Test suite for HealthMonitor"""
    
    @pytest.fixture
    def health_monitor(self):
        """Create HealthMonitor instance"""
        config = {
            'monitoring_enabled': True,
            'system_metrics_interval': 5,
            'health_check_timeout': 10,
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'disk_threshold': 90.0,
            'response_time_threshold': 5000,
            'failure_threshold': 3
        }
        monitor = HealthMonitor(config)
        # Don't start monitoring automatically for tests
        return monitor
    
    def test_health_monitor_initialization(self, health_monitor):
        """Test HealthMonitor initialization"""
        assert health_monitor.monitoring_enabled is True
        assert health_monitor.system_metrics_interval == 5
        assert health_monitor.thresholds['cpu_percent'] == 80.0
        assert health_monitor.thresholds['memory_percent'] == 85.0
        assert health_monitor.thresholds['disk_percent'] == 90.0
        
        # Should have default system checks
        assert "system_cpu" in health_monitor.health_checks
        assert "system_memory" in health_monitor.health_checks
        assert "system_disk" in health_monitor.health_checks
    
    def test_add_health_check(self, health_monitor):
        """Test adding health check"""
        check = HealthCheck(
            name="test_http_check",
            check_type="http",
            target="http://example.com/health",
            interval_seconds=30
        )
        
        health_monitor.add_health_check(check)
        
        assert "test_http_check" in health_monitor.health_checks
        assert "test_http_check" in health_monitor.service_health
        
        service_health = health_monitor.service_health["test_http_check"]
        assert service_health.service_name == "test_http_check"
        assert service_health.status == HealthStatus.UNKNOWN
    
    def test_remove_health_check(self, health_monitor):
        """Test removing health check"""
        check = HealthCheck(
            name="removable_check",
            check_type="tcp",
            target="localhost:5432"
        )
        
        health_monitor.add_health_check(check)
        assert "removable_check" in health_monitor.health_checks
        
        # Remove check
        success = health_monitor.remove_health_check("removable_check")
        assert success is True
        assert "removable_check" not in health_monitor.health_checks
        assert "removable_check" not in health_monitor.service_health
        
        # Try to remove non-existent check
        success = health_monitor.remove_health_check("nonexistent")
        assert success is False
    
    @patch('requests.get')
    def test_http_health_check_success(self, mock_get, health_monitor):
        """Test successful HTTP health check"""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_get.return_value = mock_response
        
        check = HealthCheck(
            name="http_success",
            check_type="http",
            target="http://example.com/health",
            expected_status_code=200
        )
        
        success = health_monitor._check_http_health(check)
        assert success is True
        
        mock_get.assert_called_once_with(
            "http://example.com/health",
            timeout=10,
            allow_redirects=True
        )
    
    @patch('requests.get')
    def test_http_health_check_failure(self, mock_get, health_monitor):
        """Test failed HTTP health check"""
        # Mock failed HTTP response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        check = HealthCheck(
            name="http_failure",
            check_type="http",
            target="http://example.com/health",
            expected_status_code=200
        )
        
        success = health_monitor._check_http_health(check)
        assert success is False
    
    @patch('socket.socket')
    def test_tcp_health_check_success(self, mock_socket_class, health_monitor):
        """Test successful TCP health check"""
        # Mock successful TCP connection
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket
        
        check = HealthCheck(
            name="tcp_success",
            check_type="tcp",
            target="localhost:5432"
        )
        
        success = health_monitor._check_tcp_health(check)
        assert success is True
        
        mock_socket.connect_ex.assert_called_once_with(('localhost', 5432))
        mock_socket.close.assert_called_once()
    
    @patch('socket.socket')
    def test_tcp_health_check_failure(self, mock_socket_class, health_monitor):
        """Test failed TCP health check"""
        # Mock failed TCP connection
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 1  # Connection refused
        mock_socket_class.return_value = mock_socket
        
        check = HealthCheck(
            name="tcp_failure",
            check_type="tcp",
            target="localhost:9999"
        )
        
        success = health_monitor._check_tcp_health(check)
        assert success is False
    
    def test_custom_health_check_success(self, health_monitor):
        """Test successful custom health check"""
        def always_healthy():
            return True
        
        check = HealthCheck(
            name="custom_success",
            check_type="custom",
            target="custom_service",
            custom_check_function=always_healthy
        )
        
        success = health_monitor._check_custom_health(check)
        assert success is True
    
    def test_custom_health_check_failure(self, health_monitor):
        """Test failed custom health check"""
        def always_unhealthy():
            return False
        
        check = HealthCheck(
            name="custom_failure",
            check_type="custom",
            target="custom_service",
            custom_check_function=always_unhealthy
        )
        
        success = health_monitor._check_custom_health(check)
        assert success is False
    
    @patch('psutil.process_iter')
    def test_process_health_check_success(self, mock_process_iter, health_monitor):
        """Test successful process health check"""
        # Mock process found
        mock_proc = Mock()
        mock_proc.info = {'name': 'nginx'}
        mock_process_iter.return_value = [mock_proc]
        
        check = HealthCheck(
            name="process_success",
            check_type="process",
            target="nginx"
        )
        
        success = health_monitor._check_process_health(check)
        assert success is True
    
    @patch('psutil.process_iter')
    def test_process_health_check_failure(self, mock_process_iter, health_monitor):
        """Test failed process health check"""
        # Mock no processes found
        mock_process_iter.return_value = []
        
        check = HealthCheck(
            name="process_failure",
            check_type="process",
            target="nonexistent_process"
        )
        
        success = health_monitor._check_process_health(check)
        assert success is False
    
    def test_get_system_health(self, health_monitor):
        """Test getting overall system health"""
        # Add some health checks with different statuses
        health_monitor.service_health["healthy_service"] = ServiceHealth(
            service_name="healthy_service",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time_ms=100.0
        )
        
        health_monitor.service_health["degraded_service"] = ServiceHealth(
            service_name="degraded_service",
            status=HealthStatus.DEGRADED,
            last_check=datetime.now(),
            response_time_ms=2000.0
        )
        
        system_health = health_monitor.get_system_health()
        
        assert isinstance(system_health, SystemHealth)
        assert system_health.overall_status == HealthStatus.DEGRADED  # Due to degraded service
        assert "healthy_service" in system_health.services
        assert "degraded_service" in system_health.services
    
    def test_get_service_health(self, health_monitor):
        """Test getting specific service health"""
        service_health = ServiceHealth(
            service_name="specific_service",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time_ms=150.0
        )
        
        health_monitor.service_health["specific_service"] = service_health
        
        retrieved_health = health_monitor.get_service_health("specific_service")
        assert retrieved_health == service_health
        
        # Test non-existent service
        nonexistent_health = health_monitor.get_service_health("nonexistent")
        assert nonexistent_health is None
    
    def test_get_statistics(self, health_monitor):
        """Test getting health monitoring statistics"""
        # Add some services with different statuses
        health_monitor.service_health["healthy1"] = ServiceHealth(
            "healthy1", HealthStatus.HEALTHY, datetime.now(), 100.0
        )
        health_monitor.service_health["healthy2"] = ServiceHealth(
            "healthy2", HealthStatus.HEALTHY, datetime.now(), 150.0
        )
        health_monitor.service_health["unhealthy1"] = ServiceHealth(
            "unhealthy1", HealthStatus.UNHEALTHY, datetime.now(), 5000.0
        )
        health_monitor.service_health["degraded1"] = ServiceHealth(
            "degraded1", HealthStatus.DEGRADED, datetime.now(), 3000.0
        )
        
        # Add some alerts
        health_monitor.alert_manager.create_alert(
            AlertSeverity.CRITICAL, "test", "Critical Alert", "Test"
        )
        health_monitor.alert_manager.create_alert(
            AlertSeverity.WARNING, "test", "Warning Alert", "Test"
        )
        
        stats = health_monitor.get_statistics()
        
        assert stats['monitoring_active'] is False  # Not started in tests
        assert stats['total_health_checks'] >= 3  # Default system checks
        assert stats['healthy_services'] == 2
        assert stats['unhealthy_services'] == 1
        assert stats['degraded_services'] == 1
        assert stats['active_alerts'] == 2
        assert stats['critical_alerts'] == 1


def run_health_monitor_tests():
    """Run all health monitor tests"""
    print("🏥 Running Health Monitor Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("Health Check", TestHealthCheck),
        ("Service Health", TestServiceHealth),
        ("Alert Manager", TestAlertManager),
        ("Health Monitor", TestHealthMonitor)
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
                    
                    if 'health_monitor' in sig.parameters:
                        config = {
                            'monitoring_enabled': True,
                            'system_metrics_interval': 5,
                            'health_check_timeout': 10,
                            'cpu_threshold': 80.0,
                            'memory_threshold': 85.0,
                            'disk_threshold': 90.0,
                            'response_time_threshold': 5000,
                            'failure_threshold': 3
                        }
                        health_monitor = HealthMonitor(config)
                        method(health_monitor)
                    elif 'alert_manager' in sig.parameters:
                        config = {
                            'max_alerts': 100,
                            'alert_cooldown_minutes': 5
                        }
                        alert_manager = AlertManager(config)
                        method(alert_manager)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Health Monitor Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_health_monitor_tests()
    exit(0 if success else 1)
