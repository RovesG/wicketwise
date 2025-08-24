# Purpose: Data privacy and compliance tests module initialization
# Author: WicketWise AI, Last Modified: 2024

"""
Data Privacy & Compliance Tests Module

This module contains comprehensive tests for the WicketWise data privacy
and compliance system, including GDPR/CCPA compliance monitoring,
audit logging, and data protection measures.

Test Categories:
- Privacy compliance monitoring and violation detection
- Audit logging and compliance reporting
- Data anonymization and pseudonymization
- Consent management and tracking
- Data subject rights and reporting
- Cross-border transfer compliance
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Test utilities and fixtures
from .test_privacy_monitor import (
    TestPrivacyMonitor,
    TestDataProcessingEvent,
    TestPrivacyViolation
)

from .test_audit_logger import (
    TestAuditLogger,
    TestAuditEvent,
    TestComplianceReport
)

__all__ = [
    'TestPrivacyMonitor',
    'TestDataProcessingEvent',
    'TestPrivacyViolation',
    'TestAuditLogger',
    'TestAuditEvent',
    'TestComplianceReport'
]
