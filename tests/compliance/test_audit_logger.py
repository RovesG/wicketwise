# Purpose: Unit tests for audit logging and compliance reporting system
# Author: WicketWise AI, Last Modified: 2024

import pytest
import time
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crickformers.compliance.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditTrail,
    ComplianceReport,
    AuditLevel,
    AuditCategory
)


class TestAuditEvent:
    """Test suite for AuditEvent data structure"""
    
    def test_event_creation(self):
        """Test AuditEvent creation and basic properties"""
        timestamp = datetime.now()
        event = AuditEvent(
            event_id="audit_001",
            timestamp=timestamp,
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            actor_id="user_123",
            actor_type="user",
            action="view_profile",
            resource="user_profile_456",
            resource_type="user_data",
            outcome="success",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0...",
            session_id="sess_789",
            request_id="req_abc123",
            data_subject_id="user_456",
            processing_purpose="service_provision",
            lawful_basis="contract"
        )
        
        assert event.event_id == "audit_001"
        assert event.timestamp == timestamp
        assert event.level == AuditLevel.INFO
        assert event.category == AuditCategory.DATA_ACCESS
        assert event.actor_id == "user_123"
        assert event.actor_type == "user"
        assert event.action == "view_profile"
        assert event.resource == "user_profile_456"
        assert event.resource_type == "user_data"
        assert event.outcome == "success"
        assert event.ip_address == "192.168.1.100"
        assert event.user_agent == "Mozilla/5.0..."
        assert event.session_id == "sess_789"
        assert event.request_id == "req_abc123"
        assert event.data_subject_id == "user_456"
        assert event.processing_purpose == "service_provision"
        assert event.lawful_basis == "contract"
    
    def test_event_to_dict(self):
        """Test AuditEvent dictionary conversion"""
        timestamp = datetime.now()
        event = AuditEvent(
            event_id="audit_002",
            timestamp=timestamp,
            level=AuditLevel.WARNING,
            category=AuditCategory.SECURITY_EVENT,
            actor_id="system",
            actor_type="system",
            action="failed_login",
            resource="auth_system",
            resource_type="authentication",
            outcome="failure",
            details={"attempts": 3, "reason": "invalid_password"},
            risk_indicators=["multiple_failures", "suspicious_ip"]
        )
        
        event_dict = event.to_dict()
        
        assert event_dict['event_id'] == "audit_002"
        assert event_dict['timestamp'] == timestamp.isoformat()
        assert event_dict['level'] == "warning"
        assert event_dict['category'] == "security_event"
        assert event_dict['actor_id'] == "system"
        assert event_dict['actor_type'] == "system"
        assert event_dict['action'] == "failed_login"
        assert event_dict['resource'] == "auth_system"
        assert event_dict['resource_type'] == "authentication"
        assert event_dict['outcome'] == "failure"
        assert event_dict['details'] == {"attempts": 3, "reason": "invalid_password"}
        assert event_dict['risk_indicators'] == ["multiple_failures", "suspicious_ip"]
    
    def test_event_to_json(self):
        """Test AuditEvent JSON conversion"""
        event = AuditEvent(
            event_id="audit_003",
            timestamp=datetime.now(),
            level=AuditLevel.ERROR,
            category=AuditCategory.DATA_MODIFICATION,
            actor_id="admin_001",
            actor_type="admin",
            action="delete_user_data",
            resource="user_data_789",
            resource_type="personal_data",
            outcome="success"
        )
        
        json_str = event.to_json()
        assert isinstance(json_str, str)
        assert "audit_003" in json_str
        assert "error" in json_str
        assert "data_modification" in json_str


class TestAuditTrail:
    """Test suite for AuditTrail"""
    
    def test_trail_creation(self):
        """Test AuditTrail creation and basic properties"""
        trail = AuditTrail(
            trail_id="trail_001",
            resource_id="user_123",
            resource_type="user_data"
        )
        
        assert trail.trail_id == "trail_001"
        assert trail.resource_id == "user_123"
        assert trail.resource_type == "user_data"
        assert len(trail.events) == 0
        assert isinstance(trail.created_at, datetime)
        assert isinstance(trail.last_updated, datetime)
    
    def test_add_event_to_trail(self):
        """Test adding events to audit trail"""
        trail = AuditTrail(
            trail_id="trail_002",
            resource_id="user_456",
            resource_type="user_data"
        )
        
        event = AuditEvent(
            event_id="evt_001",
            timestamp=datetime.now(),
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            actor_id="user_456",
            actor_type="user",
            action="view_profile",
            resource="user_456",
            resource_type="user_data",
            outcome="success"
        )
        
        original_updated = trail.last_updated
        time.sleep(0.01)  # Ensure time difference
        trail.add_event(event)
        
        assert len(trail.events) == 1
        assert trail.events[0] == event
        assert trail.last_updated > original_updated
    
    def test_get_events_by_category(self):
        """Test filtering events by category"""
        trail = AuditTrail("trail_003", "resource_123", "data")
        
        # Add events with different categories
        events = [
            AuditEvent("evt1", datetime.now(), AuditLevel.INFO, AuditCategory.DATA_ACCESS, 
                      "user1", "user", "read", "res1", "data", "success"),
            AuditEvent("evt2", datetime.now(), AuditLevel.INFO, AuditCategory.DATA_MODIFICATION, 
                      "user1", "user", "update", "res1", "data", "success"),
            AuditEvent("evt3", datetime.now(), AuditLevel.INFO, AuditCategory.DATA_ACCESS, 
                      "user2", "user", "read", "res1", "data", "success")
        ]
        
        for event in events:
            trail.add_event(event)
        
        access_events = trail.get_events_by_category(AuditCategory.DATA_ACCESS)
        assert len(access_events) == 2
        assert all(e.category == AuditCategory.DATA_ACCESS for e in access_events)
        
        modification_events = trail.get_events_by_category(AuditCategory.DATA_MODIFICATION)
        assert len(modification_events) == 1
        assert modification_events[0].action == "update"
    
    def test_get_events_by_actor(self):
        """Test filtering events by actor"""
        trail = AuditTrail("trail_004", "resource_456", "data")
        
        # Add events with different actors
        events = [
            AuditEvent("evt1", datetime.now(), AuditLevel.INFO, AuditCategory.DATA_ACCESS, 
                      "user1", "user", "read", "res1", "data", "success"),
            AuditEvent("evt2", datetime.now(), AuditLevel.INFO, AuditCategory.DATA_ACCESS, 
                      "user2", "user", "read", "res1", "data", "success"),
            AuditEvent("evt3", datetime.now(), AuditLevel.INFO, AuditCategory.DATA_ACCESS, 
                      "user1", "user", "read", "res1", "data", "success")
        ]
        
        for event in events:
            trail.add_event(event)
        
        user1_events = trail.get_events_by_actor("user1")
        assert len(user1_events) == 2
        assert all(e.actor_id == "user1" for e in user1_events)
        
        user2_events = trail.get_events_by_actor("user2")
        assert len(user2_events) == 1
        assert user2_events[0].actor_id == "user2"


class TestComplianceReport:
    """Test suite for ComplianceReport"""
    
    def test_report_creation(self):
        """Test ComplianceReport creation and basic properties"""
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        generated_time = datetime.now()
        
        report = ComplianceReport(
            report_id="rpt_001",
            report_type="weekly_compliance",
            period_start=start_time,
            period_end=end_time,
            generated_at=generated_time,
            total_events=150,
            events_by_category={"data_access": 100, "data_modification": 30, "security_event": 20},
            events_by_level={"info": 120, "warning": 25, "error": 5},
            compliance_violations=[{"type": "unauthorized_access", "count": 2}],
            risk_indicators=["multiple_failed_logins", "off_hours_access"],
            recommendations=["Implement MFA", "Review access controls"]
        )
        
        assert report.report_id == "rpt_001"
        assert report.report_type == "weekly_compliance"
        assert report.period_start == start_time
        assert report.period_end == end_time
        assert report.generated_at == generated_time
        assert report.total_events == 150
        assert report.events_by_category["data_access"] == 100
        assert report.events_by_level["info"] == 120
        assert len(report.compliance_violations) == 1
        assert len(report.risk_indicators) == 2
        assert len(report.recommendations) == 2
    
    def test_report_to_dict(self):
        """Test ComplianceReport dictionary conversion"""
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        generated_time = datetime.now()
        
        report = ComplianceReport(
            report_id="rpt_002",
            report_type="daily_audit",
            period_start=start_time,
            period_end=end_time,
            generated_at=generated_time,
            total_events=50,
            events_by_category={"data_access": 40, "security_event": 10},
            events_by_level={"info": 45, "warning": 5},
            compliance_violations=[],
            risk_indicators=["off_hours_access"],
            recommendations=["Monitor off-hours activity"]
        )
        
        report_dict = report.to_dict()
        
        assert report_dict['report_id'] == "rpt_002"
        assert report_dict['report_type'] == "daily_audit"
        assert report_dict['period_start'] == start_time.isoformat()
        assert report_dict['period_end'] == end_time.isoformat()
        assert report_dict['generated_at'] == generated_time.isoformat()
        assert report_dict['total_events'] == 50
        assert report_dict['events_by_category'] == {"data_access": 40, "security_event": 10}
        assert report_dict['events_by_level'] == {"info": 45, "warning": 5}
        assert report_dict['compliance_violations'] == []
        assert report_dict['risk_indicators'] == ["off_hours_access"]
        assert report_dict['recommendations'] == ["Monitor off-hours activity"]


class TestAuditLogger:
    """Test suite for AuditLogger"""
    
    @pytest.fixture
    def logger(self):
        """Create AuditLogger instance with temporary file path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'max_events_memory': 1000,
                'audit_log_path': temp_dir,
                'enable_file_logging': True,
                'enable_compression': False,
                'rotation_size_mb': 1
            }
            yield AuditLogger(config)
    
    def test_logger_initialization(self, logger):
        """Test AuditLogger initialization"""
        assert logger.max_events_memory == 1000
        assert len(logger.events) == 0
        assert len(logger.audit_trails) == 0
        assert 'suspicious_patterns' in logger.risk_rules
        assert 'compliance_patterns' in logger.risk_rules
        assert logger.stats['total_events'] == 0
    
    def test_create_audit_event(self, logger):
        """Test creating audit events"""
        event = logger.create_audit_event(
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            actor_id="user_001",
            actor_type="user",
            action="view_data",
            resource="dataset_123",
            resource_type="dataset",
            outcome="success",
            ip_address="192.168.1.1",
            session_id="sess_001"
        )
        
        assert isinstance(event, AuditEvent)
        assert event.level == AuditLevel.INFO
        assert event.category == AuditCategory.DATA_ACCESS
        assert event.actor_id == "user_001"
        assert event.actor_type == "user"
        assert event.action == "view_data"
        assert event.resource == "dataset_123"
        assert event.resource_type == "dataset"
        assert event.outcome == "success"
        assert event.ip_address == "192.168.1.1"
        assert event.session_id == "sess_001"
        assert event.event_id.startswith("AUDIT-")
    
    def test_log_event(self, logger):
        """Test logging audit events"""
        event = logger.create_audit_event(
            level=AuditLevel.WARNING,
            category=AuditCategory.SECURITY_EVENT,
            actor_id="system",
            actor_type="system",
            action="failed_authentication",
            resource="auth_system",
            resource_type="authentication",
            outcome="failure"
        )
        
        success = logger.log_event(event)
        assert success
        
        # Check event was stored
        assert len(logger.events) == 1
        assert logger.events[0] == event
        
        # Check statistics updated
        assert logger.stats['total_events'] == 1
        assert logger.stats['events_by_level']['warning'] == 1
        assert logger.stats['events_by_category']['security_event'] == 1
    
    def test_risk_pattern_detection(self, logger):
        """Test risk pattern detection"""
        # Create multiple failed login events to trigger risk detection
        for i in range(6):  # Exceeds threshold of 5
            event = logger.create_audit_event(
                level=AuditLevel.WARNING,
                category=AuditCategory.USER_AUTHENTICATION,
                actor_id="user_suspicious",
                actor_type="user",
                action="login_attempt",
                resource="auth_system",
                resource_type="authentication",
                outcome="failure"
            )
            logger.log_event(event)
        
        # Last event should have risk indicators
        last_event = logger.events[-1]
        assert "multiple_failed_logins" in last_event.risk_indicators
    
    def test_off_hours_access_detection(self, logger):
        """Test off-hours access detection"""
        # Create event with timestamp in off-hours (3 AM)
        off_hours_time = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)
        
        event = AuditEvent(
            event_id="off_hours_test",
            timestamp=off_hours_time,
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            actor_id="user_night",
            actor_type="user",
            action="access_data",
            resource="sensitive_data",
            resource_type="dataset",
            outcome="success"
        )
        
        logger.log_event(event)
        
        # Should detect off-hours access
        assert "off_hours_access" in event.risk_indicators
    
    def test_audit_trail_creation(self, logger):
        """Test automatic audit trail creation"""
        event = logger.create_audit_event(
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            actor_id="user_001",
            actor_type="user",
            action="view_profile",
            resource="user_profile_123",
            resource_type="user_data",
            outcome="success",
            data_subject_id="user_123"
        )
        
        logger.log_event(event)
        
        # Should create audit trail for data subject
        trail_key = "subject_user_123"
        assert trail_key in logger.audit_trails
        
        trail = logger.audit_trails[trail_key]
        assert trail.resource_id == "user_123"
        assert trail.resource_type == "data_subject"
        assert len(trail.events) == 1
        assert trail.events[0] == event
    
    def test_search_events(self, logger):
        """Test event search functionality"""
        # Create various events
        events_data = [
            (AuditLevel.INFO, AuditCategory.DATA_ACCESS, "user1", "success"),
            (AuditLevel.WARNING, AuditCategory.DATA_ACCESS, "user1", "failure"),
            (AuditLevel.INFO, AuditCategory.DATA_MODIFICATION, "user2", "success"),
            (AuditLevel.ERROR, AuditCategory.SECURITY_EVENT, "admin", "failure")
        ]
        
        for level, category, actor, outcome in events_data:
            event = logger.create_audit_event(
                level=level,
                category=category,
                actor_id=actor,
                actor_type="user",
                action="test_action",
                resource="test_resource",
                resource_type="test",
                outcome=outcome
            )
            logger.log_event(event)
        
        # Search by level
        warning_events = logger.search_events({'level': 'warning'})
        assert len(warning_events) == 1
        assert warning_events[0].level == AuditLevel.WARNING
        
        # Search by category
        access_events = logger.search_events({'category': 'data_access'})
        assert len(access_events) == 2
        assert all(e.category == AuditCategory.DATA_ACCESS for e in access_events)
        
        # Search by actor
        user1_events = logger.search_events({'actor_id': 'user1'})
        assert len(user1_events) == 2
        assert all(e.actor_id == 'user1' for e in user1_events)
        
        # Search by outcome
        failure_events = logger.search_events({'outcome': 'failure'})
        assert len(failure_events) == 2
        assert all(e.outcome == 'failure' for e in failure_events)
    
    def test_compliance_report_generation(self, logger):
        """Test compliance report generation"""
        # Create events over a time period
        start_time = datetime.now() - timedelta(hours=2)
        end_time = datetime.now()
        
        # Add various events
        events_data = [
            (AuditLevel.INFO, AuditCategory.DATA_ACCESS, "success"),
            (AuditLevel.WARNING, AuditCategory.SECURITY_EVENT, "failure"),
            (AuditLevel.COMPLIANCE, AuditCategory.COMPLIANCE_CHECK, "failure"),  # Violation
            (AuditLevel.ERROR, AuditCategory.DATA_MODIFICATION, "failure")
        ]
        
        for level, category, outcome in events_data:
            event = logger.create_audit_event(
                level=level,
                category=category,
                actor_id="test_user",
                actor_type="user",
                action="test_action",
                resource="test_resource",
                resource_type="test",
                outcome=outcome
            )
            # Set timestamp within period
            event.timestamp = start_time + timedelta(minutes=30)
            logger.log_event(event)
        
        # Generate compliance report
        report = logger.generate_compliance_report(
            report_type="test_report",
            period_start=start_time,
            period_end=end_time
        )
        
        assert isinstance(report, ComplianceReport)
        assert report.report_type == "test_report"
        assert report.period_start == start_time
        assert report.period_end == end_time
        assert report.total_events == 4
        assert 'data_access' in report.events_by_category
        assert 'info' in report.events_by_level
        assert len(report.compliance_violations) == 1  # One compliance failure
        assert len(report.recommendations) > 0
    
    def test_get_statistics(self, logger):
        """Test getting audit statistics"""
        # Add some events
        for i in range(5):
            event = logger.create_audit_event(
                level=AuditLevel.INFO,
                category=AuditCategory.DATA_ACCESS,
                actor_id=f"user_{i}",
                actor_type="user",
                action="test_action",
                resource="test_resource",
                resource_type="test",
                outcome="success"
            )
            logger.log_event(event)
        
        stats = logger.get_statistics()
        
        assert stats['total_events'] == 5
        assert stats['events_in_memory'] == 5
        assert stats['events_by_level']['info'] == 5
        assert stats['events_by_category']['data_access'] == 5
        assert 'last_event_time' in stats
        assert stats['file_logging_enabled'] is True


def run_audit_logger_tests():
    """Run all audit logger tests"""
    print("📋 Running Audit Logger Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("Audit Event", TestAuditEvent),
        ("Audit Trail", TestAuditTrail),
        ("Compliance Report", TestComplianceReport),
        ("Audit Logger", TestAuditLogger)
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
                    
                    if 'logger' in sig.parameters:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            config = {
                                'max_events_memory': 1000,
                                'audit_log_path': temp_dir,
                                'enable_file_logging': True,
                                'enable_compression': False,
                                'rotation_size_mb': 1
                            }
                            logger = AuditLogger(config)
                            method(logger)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Audit Logger Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_audit_logger_tests()
    exit(0 if success else 1)
