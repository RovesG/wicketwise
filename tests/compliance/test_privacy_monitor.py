# Purpose: Unit tests for privacy compliance monitoring system
# Author: WicketWise AI, Last Modified: 2024

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crickformers.compliance.privacy_monitor import (
    PrivacyMonitor,
    DataProcessingEvent,
    PrivacyViolation,
    ComplianceStatus,
    PrivacyRisk,
    DataProcessingLawfulness
)


class TestDataProcessingEvent:
    """Test suite for DataProcessingEvent data structure"""
    
    def test_event_creation(self):
        """Test DataProcessingEvent creation and basic properties"""
        timestamp = datetime.now()
        event = DataProcessingEvent(
            event_id="evt_001",
            timestamp=timestamp,
            data_subject_id="user_123",
            processing_purpose="analytics",
            data_categories=["personal_info", "behavioral_data"],
            lawful_basis=DataProcessingLawfulness.LEGITIMATE_INTERESTS,
            processor_id="analytics_service",
            data_source="web_app",
            retention_period_days=365,
            cross_border_transfer=True,
            automated_decision_making=False,
            profiling=True,
            sensitive_data=False
        )
        
        assert event.event_id == "evt_001"
        assert event.timestamp == timestamp
        assert event.data_subject_id == "user_123"
        assert event.processing_purpose == "analytics"
        assert event.data_categories == ["personal_info", "behavioral_data"]
        assert event.lawful_basis == DataProcessingLawfulness.LEGITIMATE_INTERESTS
        assert event.processor_id == "analytics_service"
        assert event.data_source == "web_app"
        assert event.retention_period_days == 365
        assert event.cross_border_transfer is True
        assert event.automated_decision_making is False
        assert event.profiling is True
        assert event.sensitive_data is False
    
    def test_event_to_dict(self):
        """Test DataProcessingEvent dictionary conversion"""
        timestamp = datetime.now()
        event = DataProcessingEvent(
            event_id="evt_002",
            timestamp=timestamp,
            data_subject_id="user_456",
            processing_purpose="marketing",
            data_categories=["email", "preferences"],
            lawful_basis=DataProcessingLawfulness.CONSENT,
            processor_id="marketing_system",
            data_source="signup_form",
            metadata={"campaign_id": "camp_001"}
        )
        
        event_dict = event.to_dict()
        
        assert event_dict['event_id'] == "evt_002"
        assert event_dict['timestamp'] == timestamp.isoformat()
        assert event_dict['data_subject_id'] == "user_456"
        assert event_dict['processing_purpose'] == "marketing"
        assert event_dict['data_categories'] == ["email", "preferences"]
        assert event_dict['lawful_basis'] == "consent"
        assert event_dict['processor_id'] == "marketing_system"
        assert event_dict['data_source'] == "signup_form"
        assert event_dict['metadata'] == {"campaign_id": "camp_001"}


class TestPrivacyViolation:
    """Test suite for PrivacyViolation data structure"""
    
    def test_violation_creation(self):
        """Test PrivacyViolation creation and basic properties"""
        timestamp = datetime.now()
        notification_deadline = timestamp + timedelta(hours=72)
        
        violation = PrivacyViolation(
            violation_id="viol_001",
            timestamp=timestamp,
            violation_type="missing_consent",
            risk_level=PrivacyRisk.HIGH,
            description="Processing without valid consent",
            affected_data_subjects=1,
            data_categories_affected=["personal_info"],
            root_cause="System configuration error",
            remediation_actions=["Obtain consent", "Update privacy notice"],
            notification_required=True,
            notification_deadline=notification_deadline
        )
        
        assert violation.violation_id == "viol_001"
        assert violation.timestamp == timestamp
        assert violation.violation_type == "missing_consent"
        assert violation.risk_level == PrivacyRisk.HIGH
        assert violation.description == "Processing without valid consent"
        assert violation.affected_data_subjects == 1
        assert violation.data_categories_affected == ["personal_info"]
        assert violation.root_cause == "System configuration error"
        assert violation.remediation_actions == ["Obtain consent", "Update privacy notice"]
        assert violation.notification_required is True
        assert violation.notification_deadline == notification_deadline
        assert violation.resolved is False
        assert violation.resolution_timestamp is None
    
    def test_violation_to_dict(self):
        """Test PrivacyViolation dictionary conversion"""
        timestamp = datetime.now()
        violation = PrivacyViolation(
            violation_id="viol_002",
            timestamp=timestamp,
            violation_type="excessive_retention",
            risk_level=PrivacyRisk.MEDIUM,
            description="Data retained beyond policy limit",
            affected_data_subjects=5,
            data_categories_affected=["user_profiles", "activity_logs"],
            root_cause="Automated deletion failure",
            remediation_actions=["Delete expired data", "Fix deletion process"],
            notification_required=False
        )
        
        violation_dict = violation.to_dict()
        
        assert violation_dict['violation_id'] == "viol_002"
        assert violation_dict['timestamp'] == timestamp.isoformat()
        assert violation_dict['violation_type'] == "excessive_retention"
        assert violation_dict['risk_level'] == "medium"
        assert violation_dict['description'] == "Data retained beyond policy limit"
        assert violation_dict['affected_data_subjects'] == 5
        assert violation_dict['data_categories_affected'] == ["user_profiles", "activity_logs"]
        assert violation_dict['root_cause'] == "Automated deletion failure"
        assert violation_dict['remediation_actions'] == ["Delete expired data", "Fix deletion process"]
        assert violation_dict['notification_required'] is False
        assert violation_dict['notification_deadline'] is None
        assert violation_dict['resolved'] is False


class TestPrivacyMonitor:
    """Test suite for PrivacyMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create PrivacyMonitor instance"""
        config = {
            'max_events': 1000,
            'retention_policies': {
                'analytics': 365,
                'marketing': 730,
                'user_data': 2555
            }
        }
        return PrivacyMonitor(config)
    
    def test_monitor_initialization(self, monitor):
        """Test PrivacyMonitor initialization"""
        assert len(monitor.processing_events) == 0
        assert len(monitor.violations) == 0
        assert len(monitor.data_subjects) == 0
        assert len(monitor.consent_records) == 0
        assert 'gdpr_compliance' in monitor.monitoring_rules
        assert 'ccpa_compliance' in monitor.monitoring_rules
        assert 'data_minimization' in monitor.monitoring_rules
        assert monitor.retention_policies['analytics'] == 365
        assert monitor.retention_policies['marketing'] == 730
    
    def test_record_processing_event(self, monitor):
        """Test recording data processing events"""
        event = DataProcessingEvent(
            event_id="test_001",
            timestamp=datetime.now(),
            data_subject_id="user_001",
            processing_purpose="analytics",
            data_categories=["page_views", "click_events"],
            lawful_basis=DataProcessingLawfulness.LEGITIMATE_INTERESTS,
            processor_id="analytics_engine",
            data_source="web_tracker"
        )
        
        success = monitor.record_processing_event(event)
        assert success
        
        # Check event was stored
        assert len(monitor.processing_events) == 1
        assert monitor.processing_events[0] == event
        
        # Check data subject tracking
        assert "user_001" in monitor.data_subjects
        subject_data = monitor.data_subjects["user_001"]
        assert "analytics" in subject_data['processing_purposes']
        assert "page_views" in subject_data['data_categories']
        assert "click_events" in subject_data['data_categories']
        assert "legitimate_interests" in subject_data['lawful_bases']
    
    def test_consent_violation_detection(self, monitor):
        """Test detection of consent violations"""
        # Create event that requires consent but uses different lawful basis
        event = DataProcessingEvent(
            event_id="consent_test",
            timestamp=datetime.now(),
            data_subject_id="user_002",
            processing_purpose="marketing",  # Requires consent
            data_categories=["email", "preferences"],
            lawful_basis=DataProcessingLawfulness.LEGITIMATE_INTERESTS,  # Wrong basis
            processor_id="marketing_system",
            data_source="signup_form"
        )
        
        initial_violations = len(monitor.violations)
        monitor.record_processing_event(event)
        
        # Should have created a violation
        assert len(monitor.violations) > initial_violations
        
        # Check violation details
        violation = monitor.violations[-1]
        assert violation.violation_type == "missing_consent"
        assert violation.risk_level == PrivacyRisk.HIGH
        assert "marketing" in violation.description
        assert violation.notification_required is True
    
    def test_sensitive_data_violation_detection(self, monitor):
        """Test detection of sensitive data violations"""
        event = DataProcessingEvent(
            event_id="sensitive_test",
            timestamp=datetime.now(),
            data_subject_id="user_003",
            processing_purpose="research",
            data_categories=["health", "biometric"],  # Sensitive categories
            lawful_basis=DataProcessingLawfulness.LEGITIMATE_INTERESTS,  # Insufficient for sensitive data
            processor_id="research_system",
            data_source="health_app",
            sensitive_data=True
        )
        
        initial_violations = len(monitor.violations)
        monitor.record_processing_event(event)
        
        # Should have created a violation
        assert len(monitor.violations) > initial_violations
        
        violation = monitor.violations[-1]
        assert violation.violation_type == "sensitive_data_violation"
        assert violation.risk_level == PrivacyRisk.CRITICAL
        assert violation.notification_required is True
    
    def test_retention_violation_detection(self, monitor):
        """Test detection of excessive retention violations"""
        event = DataProcessingEvent(
            event_id="retention_test",
            timestamp=datetime.now(),
            data_subject_id="user_004",
            processing_purpose="analytics",
            data_categories=["usage_data"],
            lawful_basis=DataProcessingLawfulness.LEGITIMATE_INTERESTS,
            processor_id="analytics_system",
            data_source="app_logs",
            retention_period_days=3000  # Exceeds maximum
        )
        
        initial_violations = len(monitor.violations)
        monitor.record_processing_event(event)
        
        # Should have created a violation
        assert len(monitor.violations) > initial_violations
        
        violation = monitor.violations[-1]
        assert violation.violation_type == "excessive_retention"
        assert violation.risk_level == PrivacyRisk.MEDIUM
    
    def test_cross_border_transfer_violation(self, monitor):
        """Test detection of unsafe cross-border transfer violations"""
        event = DataProcessingEvent(
            event_id="transfer_test",
            timestamp=datetime.now(),
            data_subject_id="user_005",
            processing_purpose="analytics",
            data_categories=["user_data"],
            lawful_basis=DataProcessingLawfulness.LEGITIMATE_INTERESTS,
            processor_id="analytics_system",
            data_source="web_app",
            cross_border_transfer=True,
            metadata={}  # No adequacy decision or safeguards
        )
        
        initial_violations = len(monitor.violations)
        monitor.record_processing_event(event)
        
        # Should have created a violation
        assert len(monitor.violations) > initial_violations
        
        violation = monitor.violations[-1]
        assert violation.violation_type == "unsafe_cross_border_transfer"
        assert violation.risk_level == PrivacyRisk.HIGH
    
    def test_safe_cross_border_transfer(self, monitor):
        """Test that safe cross-border transfers don't create violations"""
        event = DataProcessingEvent(
            event_id="safe_transfer_test",
            timestamp=datetime.now(),
            data_subject_id="user_006",
            processing_purpose="service_provision",  # Doesn't require consent
            data_categories=["user_data"],
            lawful_basis=DataProcessingLawfulness.CONTRACT,  # Appropriate for service provision
            processor_id="service_system",
            data_source="web_app",
            cross_border_transfer=True,
            metadata={"adequacy_decision": True}  # Has adequacy decision
        )
        
        initial_violations = len(monitor.violations)
        monitor.record_processing_event(event)
        
        # Should not have created a violation
        assert len(monitor.violations) == initial_violations
    
    def test_compliance_status_compliant(self, monitor):
        """Test compliance status for compliant processing"""
        # Add compliant event
        event = DataProcessingEvent(
            event_id="compliant_test",
            timestamp=datetime.now(),
            data_subject_id="user_007",
            processing_purpose="service_provision",
            data_categories=["contact_info"],
            lawful_basis=DataProcessingLawfulness.CONTRACT,
            processor_id="service_system",
            data_source="registration"
        )
        
        monitor.record_processing_event(event)
        
        # Check overall compliance status
        status = monitor.get_compliance_status()
        assert status['overall_status'] == ComplianceStatus.COMPLIANT
        assert status['total_violations'] == 0
        assert status['compliance_score'] == 100.0
        
        # Check subject-specific compliance
        subject_status = monitor.get_compliance_status("user_007")
        assert subject_status['status'] == ComplianceStatus.COMPLIANT
        assert subject_status['violations'] == 0
    
    def test_compliance_status_non_compliant(self, monitor):
        """Test compliance status with violations"""
        # Add event that creates critical violation
        event = DataProcessingEvent(
            event_id="violation_test",
            timestamp=datetime.now(),
            data_subject_id="user_008",
            processing_purpose="research",
            data_categories=["health"],
            lawful_basis=DataProcessingLawfulness.LEGITIMATE_INTERESTS,
            processor_id="research_system",
            data_source="health_app",
            sensitive_data=True
        )
        
        monitor.record_processing_event(event)
        
        # Check overall compliance status
        status = monitor.get_compliance_status()
        assert status['overall_status'] == ComplianceStatus.NON_COMPLIANT
        assert status['total_violations'] > 0
        assert status['critical_violations'] > 0
        assert status['compliance_score'] < 100.0
    
    def test_privacy_impact_assessment(self, monitor):
        """Test Privacy Impact Assessment generation"""
        # Add multiple events for same purpose
        for i in range(3):
            event = DataProcessingEvent(
                event_id=f"pia_test_{i}",
                timestamp=datetime.now(),
                data_subject_id=f"user_{i+10}",
                processing_purpose="profiling",
                data_categories=["behavioral_data", "preferences"],
                lawful_basis=DataProcessingLawfulness.LEGITIMATE_INTERESTS,
                processor_id="profiling_system",
                data_source="web_app",
                automated_decision_making=True,
                profiling=True,
                cross_border_transfer=i > 0  # Some cross-border transfers
            )
            monitor.record_processing_event(event)
        
        # Generate PIA
        pia = monitor.generate_privacy_impact_assessment("profiling")
        
        assert pia['processing_purpose'] == "profiling"
        assert 'risk_level' in pia
        assert 'risk_factors' in pia
        assert 'data_categories' in pia
        assert 'recommendations' in pia
        assert pia['affected_data_subjects'] == 3
        assert pia['automated_decisions'] == 3
        assert pia['cross_border_transfers'] == 2
        
        # Should have medium/high risk due to automated decisions and profiling
        assert pia['risk_level'] in ['medium', 'high']
    
    def test_data_subject_rights_report(self, monitor):
        """Test data subject rights report generation"""
        # Add events for data subject
        events_data = [
            ("analytics", ["page_views"], DataProcessingLawfulness.LEGITIMATE_INTERESTS),
            ("marketing", ["email"], DataProcessingLawfulness.CONSENT),
            ("service", ["profile"], DataProcessingLawfulness.CONTRACT)
        ]
        
        for i, (purpose, categories, basis) in enumerate(events_data):
            event = DataProcessingEvent(
                event_id=f"rights_test_{i}",
                timestamp=datetime.now() - timedelta(days=i),
                data_subject_id="user_rights_test",
                processing_purpose=purpose,
                data_categories=categories,
                lawful_basis=basis,
                processor_id=f"system_{i}",
                data_source="web_app"
            )
            monitor.record_processing_event(event)
        
        # Generate rights report
        report = monitor.get_data_subject_rights_report("user_rights_test")
        
        assert report['data_subject_id'] == "user_rights_test"
        assert len(report['processing_activities']) == 3
        assert 'first_processing_date' in report
        assert 'last_processing_date' in report
        assert 'rights_information' in report
        
        # Check processing activities
        purposes = [activity['purpose'] for activity in report['processing_activities']]
        assert 'analytics' in purposes
        assert 'marketing' in purposes
        assert 'service' in purposes
    
    def test_nonexistent_data_subject_report(self, monitor):
        """Test rights report for nonexistent data subject"""
        report = monitor.get_data_subject_rights_report("nonexistent_user")
        assert 'error' in report
        assert 'not found' in report['error'].lower()


def run_privacy_monitor_tests():
    """Run all privacy monitor tests"""
    print("🔒 Running Privacy Monitor Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("Data Processing Event", TestDataProcessingEvent),
        ("Privacy Violation", TestPrivacyViolation),
        ("Privacy Monitor", TestPrivacyMonitor)
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
                    
                    if 'monitor' in sig.parameters:
                        config = {
                            'max_events': 1000,
                            'retention_policies': {
                                'analytics': 365,
                                'marketing': 730,
                                'user_data': 2555
                            }
                        }
                        monitor = PrivacyMonitor(config)
                        method(monitor)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Privacy Monitor Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_privacy_monitor_tests()
    exit(0 if success else 1)
