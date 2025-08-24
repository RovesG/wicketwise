# Purpose: Security monitoring and threat detection system
# Author: WicketWise AI, Last Modified: 2024

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import hashlib
import ipaddress
import re


class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SUSPICIOUS_LOGIN = "suspicious_login"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNUSUAL_ACCESS_PATTERN = "unusual_access_pattern"
    MALICIOUS_REQUEST = "malicious_request"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ACCOUNT_TAKEOVER = "account_takeover"
    DDoS_ATTACK = "ddos_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_ATTEMPT = "csrf_attempt"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: datetime
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security event to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'user_agent': self.user_agent,
            'endpoint': self.endpoint,
            'description': self.description,
            'details': self.details,
            'indicators': self.indicators,
            'mitigations': self.mitigations,
            'resolved': self.resolved,
            'resolution_timestamp': self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
        }


@dataclass
class SecurityThreat:
    """Identified security threat"""
    threat_id: str
    threat_type: SecurityEventType
    threat_level: ThreatLevel
    first_seen: datetime
    last_seen: datetime
    event_count: int
    affected_resources: Set[str] = field(default_factory=set)
    source_ips: Set[str] = field(default_factory=set)
    user_ids: Set[str] = field(default_factory=set)
    indicators: List[str] = field(default_factory=list)
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security threat to dictionary"""
        return {
            'threat_id': self.threat_id,
            'threat_type': self.threat_type.value,
            'threat_level': self.threat_level.value,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'event_count': self.event_count,
            'affected_resources': list(self.affected_resources),
            'source_ips': list(self.source_ips),
            'user_ids': list(self.user_ids),
            'indicators': self.indicators,
            'active': self.active
        }


@dataclass
class SecurityAlert:
    """Security alert for notifications"""
    alert_id: str
    timestamp: datetime
    threat_level: ThreatLevel
    title: str
    description: str
    events: List[str] = field(default_factory=list)  # Event IDs
    recommended_actions: List[str] = field(default_factory=list)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'threat_level': self.threat_level.value,
            'title': self.title,
            'description': self.description,
            'events': self.events,
            'recommended_actions': self.recommended_actions,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }


class SecurityMonitor:
    """Comprehensive security monitoring and threat detection system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Event storage
        self.events: deque = deque(maxlen=self.config.get('max_events', 100000))
        self.threats: Dict[str, SecurityThreat] = {}
        self.alerts: Dict[str, SecurityAlert] = {}
        
        # Detection rules and patterns
        self.detection_rules = self._initialize_detection_rules()
        self.ip_reputation = self._initialize_ip_reputation()
        
        # Tracking data structures
        self.failed_logins: Dict[str, List[datetime]] = defaultdict(list)
        self.request_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.monitoring_config = {
            'brute_force_threshold': self.config.get('brute_force_threshold', 5),
            'brute_force_window_minutes': self.config.get('brute_force_window_minutes', 15),
            'unusual_access_threshold': self.config.get('unusual_access_threshold', 10),
            'rate_limit_threshold': self.config.get('rate_limit_threshold', 100),
            'geolocation_check': self.config.get('geolocation_check', True),
            'user_agent_analysis': self.config.get('user_agent_analysis', True)
        }
        
        # Alert handlers
        self.alert_handlers: List[Callable[[SecurityAlert], None]] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def _initialize_detection_rules(self) -> Dict[str, Any]:
        """Initialize security detection rules"""
        return {
            'sql_injection_patterns': [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
                r"(\b(UNION|OR|AND)\b.*\b(SELECT|INSERT|UPDATE|DELETE)\b)",
                r"(\'|\")(\s)*(OR|AND)(\s)*(\d+|\w+)(\s)*=(\s)*(\d+|\w+)",
                r"(\'|\");(\s)*(DROP|DELETE|INSERT|UPDATE)"
            ],
            'xss_patterns': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>",
                r"eval\s*\(",
                r"document\.(cookie|write|location)"
            ],
            'path_traversal_patterns': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e\\",
                r"..%2f",
                r"..%5c"
            ],
            'suspicious_user_agents': [
                r"sqlmap",
                r"nikto",
                r"nmap",
                r"masscan",
                r"burp",
                r"zap",
                r"crawler",
                r"bot.*attack"
            ],
            'malicious_ips': set(),  # Would be populated from threat intelligence
            'tor_exit_nodes': set()  # Would be populated from Tor network data
        }
    
    def _initialize_ip_reputation(self) -> Dict[str, Dict[str, Any]]:
        """Initialize IP reputation database"""
        return {
            # Format: IP -> {'reputation': score, 'categories': [], 'last_updated': datetime}
            # This would be populated from threat intelligence feeds
        }
    
    def record_security_event(self, event: SecurityEvent) -> bool:
        """Record a security event and analyze for threats"""
        try:
            with self.lock:
                # Add event to storage
                self.events.append(event)
                
                # Analyze event for threat patterns
                self._analyze_event_for_threats(event)
                
                # Update tracking data
                self._update_tracking_data(event)
                
                # Check for immediate threats
                threats = self._detect_immediate_threats(event)
                
                # Generate alerts if necessary
                for threat in threats:
                    self._generate_alert(threat, [event])
                
                self.logger.info(f"Recorded security event: {event.event_type.value} from {event.source_ip}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error recording security event: {str(e)}")
            return False
    
    def _analyze_event_for_threats(self, event: SecurityEvent):
        """Analyze event for threat indicators"""
        indicators = []
        
        # Check for malicious patterns in request data
        if 'request_data' in event.details:
            request_data = str(event.details['request_data']).lower()
            
            # SQL injection detection
            for pattern in self.detection_rules['sql_injection_patterns']:
                if re.search(pattern, request_data, re.IGNORECASE):
                    indicators.append('sql_injection_pattern')
                    event.event_type = SecurityEventType.SQL_INJECTION
                    event.threat_level = ThreatLevel.HIGH
                    break
            
            # XSS detection
            for pattern in self.detection_rules['xss_patterns']:
                if re.search(pattern, request_data, re.IGNORECASE):
                    indicators.append('xss_pattern')
                    event.event_type = SecurityEventType.XSS_ATTEMPT
                    event.threat_level = ThreatLevel.HIGH
                    break
            
            # Path traversal detection
            for pattern in self.detection_rules['path_traversal_patterns']:
                if re.search(pattern, request_data, re.IGNORECASE):
                    indicators.append('path_traversal_pattern')
                    event.event_type = SecurityEventType.MALICIOUS_REQUEST
                    event.threat_level = ThreatLevel.MEDIUM
                    break
        
        # Check user agent
        if event.user_agent:
            for pattern in self.detection_rules['suspicious_user_agents']:
                if re.search(pattern, event.user_agent, re.IGNORECASE):
                    indicators.append('suspicious_user_agent')
                    if event.threat_level == ThreatLevel.LOW:
                        event.threat_level = ThreatLevel.MEDIUM
                    break
        
        # Check IP reputation
        if event.source_ip:
            if event.source_ip in self.detection_rules['malicious_ips']:
                indicators.append('known_malicious_ip')
                event.threat_level = ThreatLevel.HIGH
            
            if event.source_ip in self.detection_rules['tor_exit_nodes']:
                indicators.append('tor_exit_node')
                if event.threat_level == ThreatLevel.LOW:
                    event.threat_level = ThreatLevel.MEDIUM
            
            # Check for private/internal IP accessing from external
            try:
                ip = ipaddress.ip_address(event.source_ip)
                if ip.is_private and event.details.get('external_access', False):
                    indicators.append('internal_ip_external_access')
                    event.threat_level = ThreatLevel.MEDIUM
            except ValueError:
                pass
        
        # Update event indicators
        event.indicators.extend(indicators)
    
    def _update_tracking_data(self, event: SecurityEvent):
        """Update tracking data structures"""
        current_time = datetime.now()
        
        # Track failed logins
        if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
            key = f"{event.source_ip}:{event.user_id or 'unknown'}"
            self.failed_logins[key].append(current_time)
            
            # Clean old entries
            cutoff_time = current_time - timedelta(minutes=self.monitoring_config['brute_force_window_minutes'])
            self.failed_logins[key] = [t for t in self.failed_logins[key] if t > cutoff_time]
        
        # Track request patterns
        if event.source_ip:
            self.request_patterns[event.source_ip].append({
                'timestamp': current_time,
                'endpoint': event.endpoint,
                'event_type': event.event_type.value,
                'threat_level': event.threat_level.value
            })
        
        # Track user sessions
        if event.user_id:
            if event.user_id not in self.user_sessions:
                self.user_sessions[event.user_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'ip_addresses': set(),
                    'user_agents': set(),
                    'event_count': 0
                }
            
            session = self.user_sessions[event.user_id]
            session['last_seen'] = current_time
            session['event_count'] += 1
            
            if event.source_ip:
                session['ip_addresses'].add(event.source_ip)
            if event.user_agent:
                session['user_agents'].add(event.user_agent)
    
    def _detect_immediate_threats(self, event: SecurityEvent) -> List[SecurityThreat]:
        """Detect immediate threats from event"""
        threats = []
        current_time = datetime.now()
        
        # Brute force detection
        if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
            key = f"{event.source_ip}:{event.user_id or 'unknown'}"
            if len(self.failed_logins[key]) >= self.monitoring_config['brute_force_threshold']:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    threat_type=SecurityEventType.BRUTE_FORCE_ATTACK,
                    threat_level=ThreatLevel.HIGH,
                    first_seen=self.failed_logins[key][0],
                    last_seen=current_time,
                    event_count=len(self.failed_logins[key]),
                    source_ips={event.source_ip} if event.source_ip else set(),
                    user_ids={event.user_id} if event.user_id else set(),
                    indicators=['multiple_failed_logins', 'brute_force_pattern']
                )
                threats.append(threat)
                self.threats[threat.threat_id] = threat
        
        # DDoS detection (high request rate from single IP)
        if event.source_ip:
            recent_requests = [
                req for req in self.request_patterns[event.source_ip]
                if req['timestamp'] > current_time - timedelta(minutes=5)
            ]
            
            if len(recent_requests) > self.monitoring_config['rate_limit_threshold']:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    threat_type=SecurityEventType.DDoS_ATTACK,
                    threat_level=ThreatLevel.HIGH,
                    first_seen=recent_requests[0]['timestamp'],
                    last_seen=current_time,
                    event_count=len(recent_requests),
                    source_ips={event.source_ip},
                    indicators=['high_request_rate', 'ddos_pattern']
                )
                threats.append(threat)
                self.threats[threat.threat_id] = threat
        
        # Unusual access pattern detection
        if event.user_id and event.user_id in self.user_sessions:
            session = self.user_sessions[event.user_id]
            
            # Multiple IP addresses for same user
            if len(session['ip_addresses']) > self.monitoring_config['unusual_access_threshold']:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    threat_type=SecurityEventType.UNUSUAL_ACCESS_PATTERN,
                    threat_level=ThreatLevel.MEDIUM,
                    first_seen=session['first_seen'],
                    last_seen=current_time,
                    event_count=session['event_count'],
                    source_ips=session['ip_addresses'],
                    user_ids={event.user_id},
                    indicators=['multiple_ip_addresses', 'unusual_access_pattern']
                )
                threats.append(threat)
                self.threats[threat.threat_id] = threat
        
        return threats
    
    def _generate_alert(self, threat: SecurityThreat, events: List[SecurityEvent]):
        """Generate security alert for threat"""
        alert = SecurityAlert(
            alert_id=self._generate_alert_id(),
            timestamp=datetime.now(),
            threat_level=threat.threat_level,
            title=f"{threat.threat_type.value.replace('_', ' ').title()} Detected",
            description=f"Security threat detected: {threat.threat_type.value}",
            events=[event.event_id for event in events],
            recommended_actions=self._get_recommended_actions(threat.threat_type)
        )
        
        self.alerts[alert.alert_id] = alert
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {str(e)}")
        
        self.logger.warning(f"Security alert generated: {alert.title}")
    
    def _get_recommended_actions(self, threat_type: SecurityEventType) -> List[str]:
        """Get recommended actions for threat type"""
        actions_map = {
            SecurityEventType.BRUTE_FORCE_ATTACK: [
                "Block source IP address",
                "Implement account lockout policy",
                "Enable multi-factor authentication",
                "Review authentication logs"
            ],
            SecurityEventType.DDoS_ATTACK: [
                "Enable rate limiting",
                "Block source IP address",
                "Scale infrastructure",
                "Contact DDoS protection service"
            ],
            SecurityEventType.SQL_INJECTION: [
                "Block malicious requests",
                "Review database access logs",
                "Update input validation",
                "Patch application vulnerabilities"
            ],
            SecurityEventType.XSS_ATTEMPT: [
                "Block malicious requests",
                "Update input sanitization",
                "Review content security policy",
                "Scan for XSS vulnerabilities"
            ],
            SecurityEventType.UNUSUAL_ACCESS_PATTERN: [
                "Verify user identity",
                "Review account activity",
                "Enable additional authentication",
                "Monitor user behavior"
            ]
        }
        
        return actions_map.get(threat_type, [
            "Investigate security event",
            "Review system logs",
            "Implement additional monitoring"
        ])
    
    def add_alert_handler(self, handler: Callable[[SecurityAlert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge security alert"""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                
                self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
            return False
    
    def resolve_threat(self, threat_id: str) -> bool:
        """Mark threat as resolved"""
        with self.lock:
            if threat_id in self.threats:
                self.threats[threat_id].active = False
                self.logger.info(f"Threat resolved: {threat_id}")
                return True
            return False
    
    def get_active_threats(self) -> List[SecurityThreat]:
        """Get all active threats"""
        with self.lock:
            return [threat for threat in self.threats.values() if threat.active]
    
    def get_unacknowledged_alerts(self) -> List[SecurityAlert]:
        """Get all unacknowledged alerts"""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.acknowledged]
    
    def search_events(self, filters: Dict[str, Any], limit: int = 1000) -> List[SecurityEvent]:
        """Search security events with filters"""
        with self.lock:
            results = []
            
            for event in reversed(self.events):
                if len(results) >= limit:
                    break
                
                # Apply filters
                if 'event_type' in filters and event.event_type.value != filters['event_type']:
                    continue
                if 'threat_level' in filters and event.threat_level.value != filters['threat_level']:
                    continue
                if 'source_ip' in filters and event.source_ip != filters['source_ip']:
                    continue
                if 'user_id' in filters and event.user_id != filters['user_id']:
                    continue
                
                # Time range filters
                if 'start_time' in filters and event.timestamp < filters['start_time']:
                    continue
                if 'end_time' in filters and event.timestamp > filters['end_time']:
                    continue
                
                results.append(event)
            
            return results
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter recent events
            recent_events = [e for e in self.events if e.timestamp > cutoff_time]
            
            # Count by type and threat level
            events_by_type = defaultdict(int)
            events_by_threat_level = defaultdict(int)
            
            for event in recent_events:
                events_by_type[event.event_type.value] += 1
                events_by_threat_level[event.threat_level.value] += 1
            
            # Active threats
            active_threats = self.get_active_threats()
            recent_threats = [t for t in active_threats if t.last_seen > cutoff_time]
            
            # Unacknowledged alerts
            unack_alerts = self.get_unacknowledged_alerts()
            recent_alerts = [a for a in unack_alerts if a.timestamp > cutoff_time]
            
            return {
                'time_period_hours': hours,
                'total_events': len(recent_events),
                'events_by_type': dict(events_by_type),
                'events_by_threat_level': dict(events_by_threat_level),
                'active_threats': len(recent_threats),
                'unacknowledged_alerts': len(recent_alerts),
                'top_source_ips': self._get_top_source_ips(recent_events),
                'top_targeted_users': self._get_top_targeted_users(recent_events)
            }
    
    def _get_top_source_ips(self, events: List[SecurityEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top source IPs by event count"""
        ip_counts = defaultdict(int)
        for event in events:
            if event.source_ip:
                ip_counts[event.source_ip] += 1
        
        return [
            {'ip': ip, 'event_count': count}
            for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _get_top_targeted_users(self, events: List[SecurityEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top targeted users by event count"""
        user_counts = defaultdict(int)
        for event in events:
            if event.user_id:
                user_counts[event.user_id] += 1
        
        return [
            {'user_id': user_id, 'event_count': count}
            for user_id, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _generate_threat_id(self) -> str:
        """Generate unique threat ID"""
        timestamp = str(int(time.time() * 1000))
        return f"threat_{timestamp}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = str(int(time.time() * 1000))
        return f"alert_{timestamp}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
    
    def start_monitoring(self):
        """Start security monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        self.logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                time.sleep(60)  # Check every minute
                self._periodic_analysis()
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
    
    def _periodic_analysis(self):
        """Perform periodic security analysis"""
        with self.lock:
            # Clean old data
            self._cleanup_old_data()
            
            # Analyze trends
            self._analyze_security_trends()
    
    def _cleanup_old_data(self):
        """Clean up old tracking data"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)
        
        # Clean failed login tracking
        for key in list(self.failed_logins.keys()):
            self.failed_logins[key] = [t for t in self.failed_logins[key] if t > cutoff_time]
            if not self.failed_logins[key]:
                del self.failed_logins[key]
        
        # Clean request patterns
        for ip in list(self.request_patterns.keys()):
            old_len = len(self.request_patterns[ip])
            self.request_patterns[ip] = deque(
                [req for req in self.request_patterns[ip] if req['timestamp'] > cutoff_time],
                maxlen=1000
            )
            if not self.request_patterns[ip]:
                del self.request_patterns[ip]
    
    def _analyze_security_trends(self):
        """Analyze security trends and patterns"""
        # This would implement more sophisticated trend analysis
        # For now, it's a placeholder for future enhancements
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security monitoring statistics"""
        with self.lock:
            return {
                'total_events': len(self.events),
                'total_threats': len(self.threats),
                'active_threats': len([t for t in self.threats.values() if t.active]),
                'total_alerts': len(self.alerts),
                'unacknowledged_alerts': len([a for a in self.alerts.values() if not a.acknowledged]),
                'tracked_ips': len(self.request_patterns),
                'tracked_users': len(self.user_sessions),
                'monitoring_active': self._monitoring_active
            }
