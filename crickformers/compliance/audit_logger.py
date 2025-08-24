# Purpose: Comprehensive audit logging and compliance reporting
# Author: WicketWise AI, Last Modified: 2024

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import hashlib
import gzip
import os


class AuditLevel(Enum):
    """Audit event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class AuditCategory(Enum):
    """Audit event categories"""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    USER_AUTHENTICATION = "user_authentication"
    SYSTEM_CONFIGURATION = "system_configuration"
    PRIVACY_OPERATION = "privacy_operation"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_EXPORT = "data_export"
    CONSENT_MANAGEMENT = "consent_management"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    timestamp: datetime
    level: AuditLevel
    category: AuditCategory
    actor_id: str
    actor_type: str  # user, system, agent, api
    action: str
    resource: str
    resource_type: str
    outcome: str  # success, failure, partial
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    data_subject_id: Optional[str] = None
    processing_purpose: Optional[str] = None
    lawful_basis: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'category': self.category.value,
            'actor_id': self.actor_id,
            'actor_type': self.actor_type,
            'action': self.action,
            'resource': self.resource,
            'resource_type': self.resource_type,
            'outcome': self.outcome,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'data_subject_id': self.data_subject_id,
            'processing_purpose': self.processing_purpose,
            'lawful_basis': self.lawful_basis,
            'details': self.details,
            'risk_indicators': self.risk_indicators
        }
    
    def to_json(self) -> str:
        """Convert audit event to JSON string"""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class AuditTrail:
    """Audit trail for a specific resource or data subject"""
    trail_id: str
    resource_id: str
    resource_type: str
    events: List[AuditEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_event(self, event: AuditEvent):
        """Add event to audit trail"""
        self.events.append(event)
        self.last_updated = datetime.now()
    
    def get_events_by_category(self, category: AuditCategory) -> List[AuditEvent]:
        """Get events by category"""
        return [e for e in self.events if e.category == category]
    
    def get_events_by_actor(self, actor_id: str) -> List[AuditEvent]:
        """Get events by actor"""
        return [e for e in self.events if e.actor_id == actor_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit trail to dictionary"""
        return {
            'trail_id': self.trail_id,
            'resource_id': self.resource_id,
            'resource_type': self.resource_type,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'total_events': len(self.events),
            'events': [event.to_dict() for event in self.events]
        }


@dataclass
class ComplianceReport:
    """Compliance audit report"""
    report_id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    total_events: int
    events_by_category: Dict[str, int]
    events_by_level: Dict[str, int]
    compliance_violations: List[Dict[str, Any]]
    risk_indicators: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert compliance report to dictionary"""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'generated_at': self.generated_at.isoformat(),
            'total_events': self.total_events,
            'events_by_category': self.events_by_category,
            'events_by_level': self.events_by_level,
            'compliance_violations': self.compliance_violations,
            'risk_indicators': self.risk_indicators,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }


class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Storage configuration
        self.max_events_memory = self.config.get('max_events_memory', 50000)
        self.audit_log_path = self.config.get('audit_log_path', 'logs/audit')
        self.enable_file_logging = self.config.get('enable_file_logging', True)
        self.enable_compression = self.config.get('enable_compression', True)
        self.rotation_size_mb = self.config.get('rotation_size_mb', 100)
        
        # Event storage
        self.events: deque = deque(maxlen=self.max_events_memory)
        self.audit_trails: Dict[str, AuditTrail] = {}
        
        # Risk detection rules
        self.risk_rules = self._initialize_risk_rules()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # File logging setup
        if self.enable_file_logging:
            self._setup_file_logging()
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'events_by_level': defaultdict(int),
            'events_by_category': defaultdict(int),
            'risk_events': 0,
            'compliance_events': 0
        }
    
    def _initialize_risk_rules(self) -> Dict[str, Any]:
        """Initialize risk detection rules"""
        return {
            'suspicious_patterns': {
                'multiple_failed_logins': {
                    'threshold': 5,
                    'time_window_minutes': 15,
                    'risk_level': 'high'
                },
                'bulk_data_access': {
                    'threshold': 100,
                    'time_window_minutes': 60,
                    'risk_level': 'medium'
                },
                'off_hours_access': {
                    'hours': [22, 23, 0, 1, 2, 3, 4, 5, 6],
                    'risk_level': 'low'
                },
                'unusual_ip_access': {
                    'track_ips': True,
                    'risk_level': 'medium'
                }
            },
            'compliance_patterns': {
                'data_access_without_purpose': {
                    'risk_level': 'high'
                },
                'cross_border_transfer': {
                    'risk_level': 'medium'
                },
                'sensitive_data_access': {
                    'risk_level': 'high'
                }
            }
        }
    
    def _setup_file_logging(self):
        """Setup file-based audit logging"""
        try:
            os.makedirs(self.audit_log_path, exist_ok=True)
            
            # Setup rotating file handler
            from logging.handlers import RotatingFileHandler
            
            log_file = os.path.join(self.audit_log_path, 'audit.log')
            max_bytes = self.rotation_size_mb * 1024 * 1024
            
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=max_bytes, 
                backupCount=10
            )
            
            formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Create audit file logger
            self.file_logger = logging.getLogger('audit_file')
            self.file_logger.setLevel(logging.INFO)
            self.file_logger.addHandler(file_handler)
            
        except Exception as e:
            self.logger.error(f"Failed to setup file logging: {str(e)}")
            self.enable_file_logging = False
    
    def log_event(self, event: AuditEvent) -> bool:
        """Log an audit event"""
        try:
            with self.lock:
                # Add event to memory storage
                self.events.append(event)
                
                # Update statistics
                self.stats['total_events'] += 1
                self.stats['events_by_level'][event.level.value] += 1
                self.stats['events_by_category'][event.category.value] += 1
                
                if event.level in [AuditLevel.SECURITY, AuditLevel.CRITICAL]:
                    self.stats['risk_events'] += 1
                
                if event.level == AuditLevel.COMPLIANCE:
                    self.stats['compliance_events'] += 1
                
                # Check for risk patterns
                risk_indicators = self._detect_risk_patterns(event)
                if risk_indicators:
                    event.risk_indicators.extend(risk_indicators)
                
                # Add to audit trail
                self._add_to_audit_trail(event)
                
                # File logging
                if self.enable_file_logging:
                    self.file_logger.info(event.to_json())
                
                # Real-time alerting for critical events
                if event.level in [AuditLevel.CRITICAL, AuditLevel.SECURITY]:
                    self._handle_critical_event(event)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging audit event: {str(e)}")
            return False
    
    def _detect_risk_patterns(self, event: AuditEvent) -> List[str]:
        """Detect risk patterns in audit events"""
        risk_indicators = []
        
        # Check for suspicious patterns
        suspicious_rules = self.risk_rules['suspicious_patterns']
        
        # Multiple failed logins
        if (event.category == AuditCategory.USER_AUTHENTICATION and 
            event.outcome == 'failure'):
            recent_failures = self._count_recent_events(
                category=AuditCategory.USER_AUTHENTICATION,
                outcome='failure',
                actor_id=event.actor_id,
                minutes=suspicious_rules['multiple_failed_logins']['time_window_minutes']
            )
            
            if recent_failures >= suspicious_rules['multiple_failed_logins']['threshold']:
                risk_indicators.append('multiple_failed_logins')
        
        # Bulk data access
        if event.category == AuditCategory.DATA_ACCESS:
            recent_access = self._count_recent_events(
                category=AuditCategory.DATA_ACCESS,
                actor_id=event.actor_id,
                minutes=suspicious_rules['bulk_data_access']['time_window_minutes']
            )
            
            if recent_access >= suspicious_rules['bulk_data_access']['threshold']:
                risk_indicators.append('bulk_data_access')
        
        # Off-hours access
        if event.timestamp.hour in suspicious_rules['off_hours_access']['hours']:
            risk_indicators.append('off_hours_access')
        
        # Compliance patterns
        compliance_rules = self.risk_rules['compliance_patterns']
        
        # Data access without purpose
        if (event.category == AuditCategory.DATA_ACCESS and 
            not event.processing_purpose):
            risk_indicators.append('data_access_without_purpose')
        
        # Cross-border transfer
        if 'cross_border' in event.details:
            risk_indicators.append('cross_border_transfer')
        
        # Sensitive data access
        if event.details.get('sensitive_data', False):
            risk_indicators.append('sensitive_data_access')
        
        return risk_indicators
    
    def _count_recent_events(self, category: Optional[AuditCategory] = None,
                           outcome: Optional[str] = None,
                           actor_id: Optional[str] = None,
                           minutes: int = 60) -> int:
        """Count recent events matching criteria"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        count = 0
        
        for event in reversed(self.events):
            if event.timestamp < cutoff_time:
                break
            
            if category and event.category != category:
                continue
            if outcome and event.outcome != outcome:
                continue
            if actor_id and event.actor_id != actor_id:
                continue
            
            count += 1
        
        return count
    
    def _add_to_audit_trail(self, event: AuditEvent):
        """Add event to appropriate audit trail"""
        # Create trail key based on resource or data subject
        trail_key = None
        
        if event.data_subject_id:
            trail_key = f"subject_{event.data_subject_id}"
        elif event.resource:
            trail_key = f"resource_{event.resource}"
        
        if trail_key:
            if trail_key not in self.audit_trails:
                self.audit_trails[trail_key] = AuditTrail(
                    trail_id=trail_key,
                    resource_id=event.data_subject_id or event.resource,
                    resource_type=event.resource_type or 'data_subject'
                )
            
            self.audit_trails[trail_key].add_event(event)
    
    def _handle_critical_event(self, event: AuditEvent):
        """Handle critical audit events"""
        # Log critical event
        self.logger.critical(f"Critical audit event: {event.action} by {event.actor_id}")
        
        # Here you could integrate with alerting systems
        # For example: send to SIEM, trigger notifications, etc.
    
    def create_audit_event(self, level: AuditLevel, category: AuditCategory,
                          actor_id: str, actor_type: str, action: str,
                          resource: str, resource_type: str, outcome: str,
                          **kwargs) -> AuditEvent:
        """Create a new audit event"""
        event_id = self._generate_event_id()
        
        return AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            level=level,
            category=category,
            actor_id=actor_id,
            actor_type=actor_type,
            action=action,
            resource=resource,
            resource_type=resource_type,
            outcome=outcome,
            **kwargs
        )
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = str(int(time.time() * 1000000))
        return f"AUDIT-{timestamp}-{hashlib.md5(timestamp.encode()).hexdigest()[:8].upper()}"
    
    def get_audit_trail(self, resource_id: str, resource_type: str = None) -> Optional[AuditTrail]:
        """Get audit trail for resource or data subject"""
        with self.lock:
            trail_key = f"subject_{resource_id}" if resource_type == 'data_subject' else f"resource_{resource_id}"
            return self.audit_trails.get(trail_key)
    
    def search_events(self, filters: Dict[str, Any], limit: int = 1000) -> List[AuditEvent]:
        """Search audit events with filters"""
        with self.lock:
            results = []
            
            for event in reversed(self.events):
                if len(results) >= limit:
                    break
                
                # Apply filters
                if 'level' in filters and event.level.value != filters['level']:
                    continue
                if 'category' in filters and event.category.value != filters['category']:
                    continue
                if 'actor_id' in filters and event.actor_id != filters['actor_id']:
                    continue
                if 'action' in filters and filters['action'] not in event.action:
                    continue
                if 'outcome' in filters and event.outcome != filters['outcome']:
                    continue
                if 'data_subject_id' in filters and event.data_subject_id != filters['data_subject_id']:
                    continue
                
                # Time range filters
                if 'start_time' in filters and event.timestamp < filters['start_time']:
                    continue
                if 'end_time' in filters and event.timestamp > filters['end_time']:
                    continue
                
                results.append(event)
            
            return results
    
    def generate_compliance_report(self, report_type: str, 
                                 period_start: datetime,
                                 period_end: datetime) -> ComplianceReport:
        """Generate compliance audit report"""
        with self.lock:
            # Filter events by time period
            period_events = [
                e for e in self.events 
                if period_start <= e.timestamp <= period_end
            ]
            
            # Analyze events
            events_by_category = defaultdict(int)
            events_by_level = defaultdict(int)
            compliance_violations = []
            risk_indicators = set()
            
            for event in period_events:
                events_by_category[event.category.value] += 1
                events_by_level[event.level.value] += 1
                
                # Collect risk indicators
                risk_indicators.update(event.risk_indicators)
                
                # Identify compliance violations
                if event.level == AuditLevel.COMPLIANCE and event.outcome == 'failure':
                    compliance_violations.append({
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'violation_type': event.action,
                        'actor_id': event.actor_id,
                        'details': event.details
                    })
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                period_events, list(risk_indicators), compliance_violations
            )
            
            report = ComplianceReport(
                report_id=self._generate_report_id(),
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                generated_at=datetime.now(),
                total_events=len(period_events),
                events_by_category=dict(events_by_category),
                events_by_level=dict(events_by_level),
                compliance_violations=compliance_violations,
                risk_indicators=list(risk_indicators),
                recommendations=recommendations
            )
            
            return report
    
    def _generate_recommendations(self, events: List[AuditEvent], 
                                risk_indicators: List[str],
                                violations: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if 'multiple_failed_logins' in risk_indicators:
            recommendations.append("Implement account lockout policies and MFA")
        
        if 'bulk_data_access' in risk_indicators:
            recommendations.append("Review data access patterns and implement access controls")
        
        if 'off_hours_access' in risk_indicators:
            recommendations.append("Monitor and restrict off-hours access to sensitive data")
        
        if 'data_access_without_purpose' in risk_indicators:
            recommendations.append("Enforce processing purpose documentation for all data access")
        
        if 'cross_border_transfer' in risk_indicators:
            recommendations.append("Review cross-border transfer safeguards and adequacy decisions")
        
        # Violation-based recommendations
        if violations:
            recommendations.append("Address identified compliance violations immediately")
            recommendations.append("Implement additional compliance monitoring controls")
        
        # General recommendations based on event patterns
        auth_failures = sum(1 for e in events 
                           if e.category == AuditCategory.USER_AUTHENTICATION 
                           and e.outcome == 'failure')
        
        if auth_failures > len(events) * 0.1:  # More than 10% auth failures
            recommendations.append("Review authentication mechanisms and user training")
        
        return recommendations
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID"""
        timestamp = str(int(time.time() * 1000))
        return f"RPT-{timestamp}-{hashlib.md5(timestamp.encode()).hexdigest()[:8].upper()}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        with self.lock:
            return {
                'total_events': self.stats['total_events'],
                'events_in_memory': len(self.events),
                'events_by_level': dict(self.stats['events_by_level']),
                'events_by_category': dict(self.stats['events_by_category']),
                'risk_events': self.stats['risk_events'],
                'compliance_events': self.stats['compliance_events'],
                'audit_trails': len(self.audit_trails),
                'file_logging_enabled': self.enable_file_logging,
                'last_event_time': self.events[-1].timestamp.isoformat() if self.events else None
            }
