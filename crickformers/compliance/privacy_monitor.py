# Purpose: Privacy compliance monitoring and violation detection
# Author: WicketWise AI, Last Modified: 2024

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import hashlib
import json


class PrivacyRisk(Enum):
    """Privacy risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status types"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    REMEDIATION_REQUIRED = "remediation_required"


class DataProcessingLawfulness(Enum):
    """GDPR Article 6 lawful bases for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataProcessingEvent:
    """Data processing event for compliance tracking"""
    event_id: str
    timestamp: datetime
    data_subject_id: str
    processing_purpose: str
    data_categories: List[str]
    lawful_basis: DataProcessingLawfulness
    processor_id: str
    data_source: str
    retention_period_days: Optional[int] = None
    cross_border_transfer: bool = False
    automated_decision_making: bool = False
    profiling: bool = False
    sensitive_data: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'data_subject_id': self.data_subject_id,
            'processing_purpose': self.processing_purpose,
            'data_categories': self.data_categories,
            'lawful_basis': self.lawful_basis.value,
            'processor_id': self.processor_id,
            'data_source': self.data_source,
            'retention_period_days': self.retention_period_days,
            'cross_border_transfer': self.cross_border_transfer,
            'automated_decision_making': self.automated_decision_making,
            'profiling': self.profiling,
            'sensitive_data': self.sensitive_data,
            'metadata': self.metadata
        }


@dataclass
class PrivacyViolation:
    """Privacy violation record"""
    violation_id: str
    timestamp: datetime
    violation_type: str
    risk_level: PrivacyRisk
    description: str
    affected_data_subjects: int
    data_categories_affected: List[str]
    root_cause: str
    remediation_actions: List[str]
    notification_required: bool
    notification_deadline: Optional[datetime] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary"""
        return {
            'violation_id': self.violation_id,
            'timestamp': self.timestamp.isoformat(),
            'violation_type': self.violation_type,
            'risk_level': self.risk_level.value,
            'description': self.description,
            'affected_data_subjects': self.affected_data_subjects,
            'data_categories_affected': self.data_categories_affected,
            'root_cause': self.root_cause,
            'remediation_actions': self.remediation_actions,
            'notification_required': self.notification_required,
            'notification_deadline': self.notification_deadline.isoformat() if self.notification_deadline else None,
            'resolved': self.resolved,
            'resolution_timestamp': self.resolution_timestamp.isoformat() if self.resolution_timestamp else None,
            'metadata': self.metadata
        }


class PrivacyMonitor:
    """Privacy compliance monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Event storage
        self.processing_events: deque = deque(maxlen=self.config.get('max_events', 100000))
        self.violations: List[PrivacyViolation] = []
        
        # Monitoring rules
        self.monitoring_rules = self._initialize_monitoring_rules()
        
        # Data subject tracking
        self.data_subjects: Dict[str, Dict[str, Any]] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        
        # Retention policies
        self.retention_policies: Dict[str, int] = self.config.get('retention_policies', {
            'user_data': 2555,  # 7 years in days
            'analytics_data': 1095,  # 3 years
            'audit_logs': 2555,  # 7 years
            'consent_records': 2555  # 7 years
        })
        
        # Compliance thresholds
        self.compliance_thresholds = {
            'max_processing_without_consent': 0,
            'max_retention_period_days': 2555,  # 7 years
            'max_cross_border_transfers_per_day': 1000,
            'max_automated_decisions_per_hour': 100
        }
        
        # Monitoring state
        self.lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def _initialize_monitoring_rules(self) -> Dict[str, Any]:
        """Initialize privacy monitoring rules"""
        return {
            'gdpr_compliance': {
                'consent_required_purposes': [
                    'marketing', 'profiling', 'analytics', 'personalization'
                ],
                'sensitive_data_categories': [
                    'biometric', 'health', 'genetic', 'political_opinions',
                    'religious_beliefs', 'trade_union_membership'
                ],
                'notification_deadlines': {
                    'supervisory_authority': 72,  # hours
                    'data_subjects': 720  # hours (30 days)
                }
            },
            'ccpa_compliance': {
                'sale_opt_out_required': True,
                'deletion_request_deadline': 45,  # days
                'disclosure_categories': [
                    'identifiers', 'personal_info', 'commercial_info',
                    'biometric_info', 'internet_activity', 'geolocation'
                ]
            },
            'data_minimization': {
                'max_data_categories_per_purpose': 5,
                'purpose_limitation_check': True,
                'storage_limitation_check': True
            }
        }
    
    def record_processing_event(self, event: DataProcessingEvent) -> bool:
        """Record a data processing event"""
        try:
            with self.lock:
                self.processing_events.append(event)
                
                # Update data subject tracking
                if event.data_subject_id not in self.data_subjects:
                    self.data_subjects[event.data_subject_id] = {
                        'first_processing': event.timestamp,
                        'processing_purposes': set(),
                        'data_categories': set(),
                        'lawful_bases': set()
                    }
                
                subject_data = self.data_subjects[event.data_subject_id]
                subject_data['last_processing'] = event.timestamp
                subject_data['processing_purposes'].update([event.processing_purpose])
                subject_data['data_categories'].update(event.data_categories)
                subject_data['lawful_bases'].add(event.lawful_basis.value)
                
                # Check for violations
                self._check_compliance_violations(event)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error recording processing event: {str(e)}")
            return False
    
    def _check_compliance_violations(self, event: DataProcessingEvent):
        """Check for compliance violations in processing event"""
        violations = []
        
        # Check consent requirement
        gdpr_rules = self.monitoring_rules['gdpr_compliance']
        if (event.processing_purpose in gdpr_rules['consent_required_purposes'] and
            event.lawful_basis != DataProcessingLawfulness.CONSENT):
            violations.append({
                'type': 'missing_consent',
                'risk': PrivacyRisk.HIGH,
                'description': f"Processing for {event.processing_purpose} requires consent but using {event.lawful_basis.value}"
            })
        
        # Check sensitive data processing
        sensitive_categories = gdpr_rules['sensitive_data_categories']
        if any(cat in sensitive_categories for cat in event.data_categories):
            if event.lawful_basis not in [DataProcessingLawfulness.CONSENT, 
                                        DataProcessingLawfulness.VITAL_INTERESTS,
                                        DataProcessingLawfulness.LEGAL_OBLIGATION]:
                violations.append({
                    'type': 'sensitive_data_violation',
                    'risk': PrivacyRisk.CRITICAL,
                    'description': f"Sensitive data processing requires explicit consent or other specific lawful basis"
                })
        
        # Check retention period
        if event.retention_period_days and event.retention_period_days > self.compliance_thresholds['max_retention_period_days']:
            violations.append({
                'type': 'excessive_retention',
                'risk': PrivacyRisk.MEDIUM,
                'description': f"Retention period {event.retention_period_days} days exceeds maximum allowed"
            })
        
        # Check cross-border transfers
        if event.cross_border_transfer:
            # Check if adequate safeguards are in place
            if not event.metadata.get('adequacy_decision') and not event.metadata.get('safeguards'):
                violations.append({
                    'type': 'unsafe_cross_border_transfer',
                    'risk': PrivacyRisk.HIGH,
                    'description': "Cross-border transfer without adequate safeguards"
                })
        
        # Record violations
        for violation_data in violations:
            violation = PrivacyViolation(
                violation_id=self._generate_violation_id(),
                timestamp=datetime.now(),
                violation_type=violation_data['type'],
                risk_level=violation_data['risk'],
                description=violation_data['description'],
                affected_data_subjects=1,
                data_categories_affected=event.data_categories,
                root_cause=f"Processing event {event.event_id}",
                remediation_actions=self._get_remediation_actions(violation_data['type']),
                notification_required=violation_data['risk'] in [PrivacyRisk.HIGH, PrivacyRisk.CRITICAL]
            )
            
            if violation.notification_required:
                violation.notification_deadline = datetime.now() + timedelta(hours=72)
            
            self.violations.append(violation)
            self.logger.warning(f"Privacy violation detected: {violation.description}")
    
    def _generate_violation_id(self) -> str:
        """Generate unique violation ID"""
        timestamp = str(int(time.time() * 1000))
        return f"PRIV-{timestamp}-{hashlib.md5(timestamp.encode()).hexdigest()[:8].upper()}"
    
    def _get_remediation_actions(self, violation_type: str) -> List[str]:
        """Get recommended remediation actions for violation type"""
        remediation_map = {
            'missing_consent': [
                'Obtain explicit consent from data subject',
                'Review lawful basis for processing',
                'Update privacy notice',
                'Implement consent management system'
            ],
            'sensitive_data_violation': [
                'Stop processing sensitive data immediately',
                'Obtain explicit consent if required',
                'Review data minimization practices',
                'Implement additional security measures'
            ],
            'excessive_retention': [
                'Review and update retention policies',
                'Delete data that exceeds retention period',
                'Implement automated deletion processes',
                'Update privacy notices with retention periods'
            ],
            'unsafe_cross_border_transfer': [
                'Implement Standard Contractual Clauses (SCCs)',
                'Obtain adequacy decision confirmation',
                'Review transfer impact assessment',
                'Consider data localization options'
            ]
        }
        
        return remediation_map.get(violation_type, ['Review and remediate violation'])
    
    def get_compliance_status(self, data_subject_id: Optional[str] = None) -> Dict[str, Any]:
        """Get compliance status for data subject or overall system"""
        with self.lock:
            if data_subject_id:
                return self._get_subject_compliance_status(data_subject_id)
            else:
                return self._get_overall_compliance_status()
    
    def _get_subject_compliance_status(self, data_subject_id: str) -> Dict[str, Any]:
        """Get compliance status for specific data subject"""
        if data_subject_id not in self.data_subjects:
            return {'status': ComplianceStatus.COMPLIANT, 'message': 'No processing recorded'}
        
        subject_data = self.data_subjects[data_subject_id]
        subject_violations = [v for v in self.violations 
                            if data_subject_id in v.metadata.get('affected_subjects', [])]
        
        # Determine compliance status
        if any(v.risk_level == PrivacyRisk.CRITICAL and not v.resolved for v in subject_violations):
            status = ComplianceStatus.NON_COMPLIANT
        elif any(v.risk_level in [PrivacyRisk.HIGH, PrivacyRisk.MEDIUM] and not v.resolved for v in subject_violations):
            status = ComplianceStatus.REMEDIATION_REQUIRED
        elif subject_violations:
            status = ComplianceStatus.UNDER_REVIEW
        else:
            status = ComplianceStatus.COMPLIANT
        
        return {
            'data_subject_id': data_subject_id,
            'status': status,
            'processing_purposes': list(subject_data['processing_purposes']),
            'data_categories': list(subject_data['data_categories']),
            'lawful_bases': list(subject_data['lawful_bases']),
            'violations': len(subject_violations),
            'unresolved_violations': len([v for v in subject_violations if not v.resolved]),
            'first_processing': subject_data['first_processing'].isoformat(),
            'last_processing': subject_data.get('last_processing', subject_data['first_processing']).isoformat()
        }
    
    def _get_overall_compliance_status(self) -> Dict[str, Any]:
        """Get overall system compliance status"""
        total_violations = len(self.violations)
        unresolved_violations = len([v for v in self.violations if not v.resolved])
        critical_violations = len([v for v in self.violations 
                                 if v.risk_level == PrivacyRisk.CRITICAL and not v.resolved])
        
        # Determine overall status
        if critical_violations > 0:
            status = ComplianceStatus.NON_COMPLIANT
        elif unresolved_violations > 0:
            status = ComplianceStatus.REMEDIATION_REQUIRED
        elif total_violations > 0:
            status = ComplianceStatus.UNDER_REVIEW
        else:
            status = ComplianceStatus.COMPLIANT
        
        # Calculate compliance metrics
        total_events = len(self.processing_events)
        violation_rate = (total_violations / total_events) if total_events > 0 else 0
        
        return {
            'overall_status': status,
            'total_processing_events': total_events,
            'total_violations': total_violations,
            'unresolved_violations': unresolved_violations,
            'critical_violations': critical_violations,
            'violation_rate': violation_rate,
            'data_subjects_tracked': len(self.data_subjects),
            'compliance_score': max(0, 100 - (violation_rate * 100)),
            'last_assessment': datetime.now().isoformat()
        }
    
    def generate_privacy_impact_assessment(self, processing_purpose: str) -> Dict[str, Any]:
        """Generate Privacy Impact Assessment (PIA) for processing purpose"""
        with self.lock:
            # Get events for this purpose
            purpose_events = [e for e in self.processing_events 
                            if e.processing_purpose == processing_purpose]
            
            if not purpose_events:
                return {'error': f'No processing events found for purpose: {processing_purpose}'}
            
            # Analyze data categories
            data_categories = set()
            sensitive_data = False
            cross_border_transfers = 0
            automated_decisions = 0
            
            for event in purpose_events:
                data_categories.update(event.data_categories)
                if event.sensitive_data:
                    sensitive_data = True
                if event.cross_border_transfer:
                    cross_border_transfers += 1
                if event.automated_decision_making:
                    automated_decisions += 1
            
            # Assess risks
            risk_factors = []
            risk_level = PrivacyRisk.LOW
            
            if sensitive_data:
                risk_factors.append("Processing of sensitive personal data")
                risk_level = PrivacyRisk.HIGH
            
            if cross_border_transfers > 0:
                risk_factors.append(f"Cross-border data transfers ({cross_border_transfers} events)")
                if risk_level == PrivacyRisk.LOW:
                    risk_level = PrivacyRisk.MEDIUM
            
            if automated_decisions > 0:
                risk_factors.append(f"Automated decision-making ({automated_decisions} events)")
                if risk_level == PrivacyRisk.LOW:
                    risk_level = PrivacyRisk.MEDIUM
            
            if len(data_categories) > 5:
                risk_factors.append(f"Large number of data categories ({len(data_categories)})")
                if risk_level == PrivacyRisk.LOW:
                    risk_level = PrivacyRisk.MEDIUM
            
            # Generate recommendations
            recommendations = []
            if sensitive_data:
                recommendations.append("Implement additional security measures for sensitive data")
                recommendations.append("Ensure explicit consent for sensitive data processing")
            
            if cross_border_transfers > 0:
                recommendations.append("Review adequacy decisions and transfer safeguards")
                recommendations.append("Consider data localization where possible")
            
            if automated_decisions > 0:
                recommendations.append("Implement human review processes for automated decisions")
                recommendations.append("Provide clear information about automated decision-making")
            
            return {
                'processing_purpose': processing_purpose,
                'assessment_date': datetime.now().isoformat(),
                'risk_level': risk_level.value,
                'risk_factors': risk_factors,
                'data_categories': list(data_categories),
                'sensitive_data_involved': sensitive_data,
                'cross_border_transfers': cross_border_transfers,
                'automated_decisions': automated_decisions,
                'affected_data_subjects': len(set(e.data_subject_id for e in purpose_events)),
                'recommendations': recommendations,
                'compliance_measures_required': len(risk_factors) > 2
            }
    
    def get_data_subject_rights_report(self, data_subject_id: str) -> Dict[str, Any]:
        """Generate data subject rights report (Article 15 GDPR)"""
        with self.lock:
            if data_subject_id not in self.data_subjects:
                return {'error': 'Data subject not found'}
            
            subject_data = self.data_subjects[data_subject_id]
            subject_events = [e for e in self.processing_events 
                            if e.data_subject_id == data_subject_id]
            
            # Group by processing purpose
            purposes = defaultdict(list)
            for event in subject_events:
                purposes[event.processing_purpose].append(event)
            
            processing_activities = []
            for purpose, events in purposes.items():
                activity = {
                    'purpose': purpose,
                    'lawful_basis': list(set(e.lawful_basis.value for e in events)),
                    'data_categories': list(set(cat for e in events for cat in e.data_categories)),
                    'retention_periods': list(set(e.retention_period_days for e in events if e.retention_period_days)),
                    'recipients': list(set(e.processor_id for e in events)),
                    'cross_border_transfers': any(e.cross_border_transfer for e in events),
                    'automated_decision_making': any(e.automated_decision_making for e in events),
                    'profiling': any(e.profiling for e in events)
                }
                processing_activities.append(activity)
            
            return {
                'data_subject_id': data_subject_id,
                'report_date': datetime.now().isoformat(),
                'processing_activities': processing_activities,
                'data_sources': list(set(e.data_source for e in subject_events)),
                'first_processing_date': subject_data['first_processing'].isoformat(),
                'last_processing_date': subject_data.get('last_processing', subject_data['first_processing']).isoformat(),
                'total_processing_events': len(subject_events),
                'rights_information': {
                    'right_to_rectification': 'Contact data protection officer',
                    'right_to_erasure': 'Submit deletion request',
                    'right_to_restrict_processing': 'Contact data protection officer',
                    'right_to_data_portability': 'Available for consent-based processing',
                    'right_to_object': 'Available for legitimate interest processing'
                }
            }
    
    def start_monitoring(self, interval_hours: float = 1.0):
        """Start continuous privacy monitoring"""
        if self._monitoring_active:
            self.logger.warning("Privacy monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_hours,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Started privacy monitoring with {interval_hours}h interval")
    
    def stop_monitoring(self):
        """Stop privacy monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        self.logger.info("Stopped privacy monitoring")
    
    def _monitoring_loop(self, interval_hours: float):
        """Main privacy monitoring loop"""
        interval_seconds = interval_hours * 3600
        
        while self._monitoring_active:
            try:
                self._perform_periodic_checks()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in privacy monitoring loop: {str(e)}")
                time.sleep(interval_seconds)
    
    def _perform_periodic_checks(self):
        """Perform periodic privacy compliance checks"""
        with self.lock:
            # Check for data retention violations
            self._check_retention_violations()
            
            # Check for notification deadlines
            self._check_notification_deadlines()
            
            # Check for consent expiration
            self._check_consent_expiration()
    
    def _check_retention_violations(self):
        """Check for data retention policy violations"""
        current_time = datetime.now()
        
        for subject_id, subject_data in self.data_subjects.items():
            last_processing = subject_data.get('last_processing', subject_data['first_processing'])
            
            for purpose in subject_data['processing_purposes']:
                retention_days = self.retention_policies.get(purpose, self.retention_policies.get('default', 2555))
                retention_deadline = last_processing + timedelta(days=retention_days)
                
                if current_time > retention_deadline:
                    violation = PrivacyViolation(
                        violation_id=self._generate_violation_id(),
                        timestamp=current_time,
                        violation_type='retention_violation',
                        risk_level=PrivacyRisk.MEDIUM,
                        description=f"Data retention period exceeded for subject {subject_id}, purpose {purpose}",
                        affected_data_subjects=1,
                        data_categories_affected=list(subject_data['data_categories']),
                        root_cause='Automated retention check',
                        remediation_actions=['Delete expired data', 'Update retention policies'],
                        notification_required=False
                    )
                    self.violations.append(violation)
    
    def _check_notification_deadlines(self):
        """Check for missed notification deadlines"""
        current_time = datetime.now()
        
        for violation in self.violations:
            if (violation.notification_required and 
                violation.notification_deadline and 
                current_time > violation.notification_deadline and 
                not violation.resolved):
                
                self.logger.critical(f"Notification deadline missed for violation {violation.violation_id}")
    
    def _check_consent_expiration(self):
        """Check for expired consent records"""
        # This would integrate with the consent manager
        # For now, it's a placeholder for future implementation
        pass
