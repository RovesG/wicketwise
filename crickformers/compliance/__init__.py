# Purpose: Data privacy and compliance monitoring module
# Author: WicketWise AI, Last Modified: 2024

"""
Data Privacy & Compliance Module

This module provides comprehensive data privacy and compliance monitoring
for the WicketWise cricket intelligence platform, ensuring adherence to
GDPR, CCPA, and other privacy regulations.

Key Components:
- GDPR/CCPA compliance monitoring and reporting
- Data anonymization and pseudonymization utilities
- Audit logging and compliance trail management
- Data retention and deletion policy enforcement
- Privacy impact assessment tools
- Consent management and tracking
- Data breach detection and notification
- Cross-border data transfer compliance
"""

__version__ = "1.0.0"
__author__ = "WicketWise AI"

# Core compliance components
from .privacy_monitor import (
    PrivacyMonitor,
    DataProcessingEvent,
    PrivacyViolation,
    ComplianceStatus,
    PrivacyRisk
)

from .audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditLevel,
    ComplianceReport,
    AuditTrail
)

# Data anonymization (to be implemented in future phases)
# from .data_anonymizer import (
#     DataAnonymizer,
#     AnonymizationMethod,
#     PseudonymizationEngine,
#     DataMaskingPolicy
# )

# Consent management (to be implemented in future phases)
# from .consent_manager import (
#     ConsentManager,
#     ConsentRecord,
#     ConsentStatus,
#     DataProcessingPurpose,
#     ConsentWithdrawal
# )

__all__ = [
    # Privacy monitoring
    'PrivacyMonitor',
    'DataProcessingEvent',
    'PrivacyViolation',
    'ComplianceStatus',
    'PrivacyRisk',
    
    # Audit logging
    'AuditLogger',
    'AuditEvent',
    'AuditLevel',
    'ComplianceReport',
    'AuditTrail',
    
    # Data anonymization (to be implemented in future phases)
    # 'DataAnonymizer',
    # 'AnonymizationMethod',
    # 'PseudonymizationEngine',
    # 'DataMaskingPolicy',
    
    # Consent management (to be implemented in future phases)
    # 'ConsentManager',
    # 'ConsentRecord',
    # 'ConsentStatus',
    # 'DataProcessingPurpose',
    # 'ConsentWithdrawal'
]
