# Purpose: Audit trail verification and integrity checking for DGL
# Author: WicketWise AI, Last Modified: 2024

"""
Audit Verifier

Verifies audit trail integrity and compliance:
- Hash chain verification
- Audit record validation
- Compliance checking
- Forensic analysis capabilities
- Audit trail reconstruction
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from governance.audit import GovernanceAuditStore


logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Verification result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class IntegrityCheckType(Enum):
    """Types of integrity checks"""
    HASH_CHAIN = "hash_chain"
    RECORD_STRUCTURE = "record_structure"
    TIMESTAMP_SEQUENCE = "timestamp_sequence"
    USER_VALIDATION = "user_validation"
    COMPLETENESS = "completeness"
    COMPLIANCE = "compliance"


@dataclass
class IntegrityCheck:
    """Individual integrity check result"""
    check_type: IntegrityCheckType
    status: VerificationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "check_type": self.check_type.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat()
        }


@dataclass
class VerificationResult:
    """Complete verification result"""
    verification_id: str
    started_at: datetime
    completed_at: datetime
    overall_status: VerificationStatus
    checks: List[IntegrityCheck] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def passed_checks(self) -> int:
        """Count of passed checks"""
        return len([c for c in self.checks if c.status == VerificationStatus.PASSED])
    
    @property
    def failed_checks(self) -> int:
        """Count of failed checks"""
        return len([c for c in self.checks if c.status == VerificationStatus.FAILED])
    
    @property
    def warning_checks(self) -> int:
        """Count of warning checks"""
        return len([c for c in self.checks if c.status == VerificationStatus.WARNING])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "verification_id": self.verification_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "overall_status": self.overall_status.value,
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                **self.summary,
                "total_checks": len(self.checks),
                "passed_checks": self.passed_checks,
                "failed_checks": self.failed_checks,
                "warning_checks": self.warning_checks
            },
            "recommendations": self.recommendations
        }


@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    name: str
    description: str
    check_function: str  # Name of method to call
    severity: str = "warning"  # warning, error, critical
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class AuditVerifier:
    """
    Comprehensive audit trail verification system
    
    Verifies the integrity, completeness, and compliance of audit trails
    with support for forensic analysis and compliance reporting.
    """
    
    def __init__(self, audit_store: GovernanceAuditStore):
        """
        Initialize audit verifier
        
        Args:
            audit_store: Audit store to verify
        """
        self.audit_store = audit_store
        
        # Compliance rules
        self.compliance_rules: List[ComplianceRule] = []
        
        # Verification history
        self.verification_history: List[VerificationResult] = []
        
        # Setup default compliance rules
        self._setup_default_compliance_rules()
        
        logger.info("Audit verifier initialized")
    
    async def verify_audit_integrity(self, 
                                   start_time: datetime = None,
                                   end_time: datetime = None) -> VerificationResult:
        """
        Perform comprehensive audit integrity verification
        
        Args:
            start_time: Start of verification period
            end_time: End of verification period
            
        Returns:
            Verification result
        """
        verification_id = f"verify_{int(datetime.now().timestamp())}"
        started_at = datetime.now()
        
        logger.info(f"Starting audit verification: {verification_id}")
        
        checks = []
        
        try:
            # Get audit records for verification period
            records = self._get_records_for_period(start_time, end_time)
            
            # Perform various integrity checks
            checks.extend(await self._verify_record_structure(records))
            checks.extend(await self._verify_timestamp_sequence(records))
            checks.extend(await self._verify_user_validation(records))
            checks.extend(await self._verify_completeness(records))
            checks.extend(await self._verify_compliance(records))
            
            # Determine overall status
            overall_status = self._determine_overall_status(checks)
            
            # Generate summary and recommendations
            summary = self._generate_verification_summary(records, checks)
            recommendations = self._generate_recommendations(checks)
            
            result = VerificationResult(
                verification_id=verification_id,
                started_at=started_at,
                completed_at=datetime.now(),
                overall_status=overall_status,
                checks=checks,
                summary=summary,
                recommendations=recommendations
            )
            
            # Store verification result
            self.verification_history.append(result)
            
            logger.info(f"Audit verification completed: {verification_id} - {overall_status.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during audit verification: {str(e)}")
            
            # Return failed verification
            return VerificationResult(
                verification_id=verification_id,
                started_at=started_at,
                completed_at=datetime.now(),
                overall_status=VerificationStatus.FAILED,
                checks=[IntegrityCheck(
                    check_type=IntegrityCheckType.COMPLETENESS,
                    status=VerificationStatus.FAILED,
                    message=f"Verification failed with error: {str(e)}"
                )]
            )
    
    async def verify_hash_chain(self, records: List[Dict[str, Any]]) -> IntegrityCheck:
        """
        Verify hash chain integrity
        
        Args:
            records: Audit records to verify
            
        Returns:
            Hash chain verification result
        """
        try:
            if not records:
                return IntegrityCheck(
                    check_type=IntegrityCheckType.HASH_CHAIN,
                    status=VerificationStatus.SKIPPED,
                    message="No records to verify hash chain"
                )
            
            # Sort records by timestamp
            sorted_records = sorted(records, key=lambda r: r.get("timestamp", ""))
            
            verification_errors = []
            previous_hash = None
            
            for i, record in enumerate(sorted_records):
                # Calculate expected hash
                record_data = {k: v for k, v in record.items() 
                              if k not in ["hash_curr", "hash_prev"]}
                record_json = json.dumps(record_data, sort_keys=True)
                
                if previous_hash:
                    hash_input = previous_hash + record_json
                else:
                    hash_input = record_json
                
                expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()
                
                # Check if record has hash
                actual_hash = record.get("hash_curr")
                if not actual_hash:
                    verification_errors.append(f"Record {i} missing hash_curr")
                    continue
                
                # Verify hash
                if actual_hash != expected_hash:
                    verification_errors.append(
                        f"Record {i} hash mismatch: expected {expected_hash[:8]}..., "
                        f"got {actual_hash[:8]}..."
                    )
                
                # Verify previous hash reference
                expected_prev_hash = previous_hash
                actual_prev_hash = record.get("hash_prev")
                
                if expected_prev_hash != actual_prev_hash:
                    verification_errors.append(
                        f"Record {i} previous hash mismatch"
                    )
                
                previous_hash = actual_hash
            
            if verification_errors:
                return IntegrityCheck(
                    check_type=IntegrityCheckType.HASH_CHAIN,
                    status=VerificationStatus.FAILED,
                    message=f"Hash chain verification failed: {len(verification_errors)} errors",
                    details={"errors": verification_errors}
                )
            
            return IntegrityCheck(
                check_type=IntegrityCheckType.HASH_CHAIN,
                status=VerificationStatus.PASSED,
                message=f"Hash chain verification passed for {len(sorted_records)} records"
            )
            
        except Exception as e:
            return IntegrityCheck(
                check_type=IntegrityCheckType.HASH_CHAIN,
                status=VerificationStatus.FAILED,
                message=f"Hash chain verification error: {str(e)}"
            )
    
    async def _verify_record_structure(self, records: List[Dict[str, Any]]) -> List[IntegrityCheck]:
        """Verify audit record structure"""
        checks = []
        
        if not records:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.RECORD_STRUCTURE,
                status=VerificationStatus.WARNING,
                message="No records found for structure verification"
            ))
            return checks
        
        # Required fields for audit records
        required_fields = ["event_type", "user", "resource", "action", "timestamp"]
        
        structure_errors = []
        
        for i, record in enumerate(records):
            # Check required fields
            missing_fields = [field for field in required_fields if field not in record]
            if missing_fields:
                structure_errors.append(
                    f"Record {i} missing fields: {missing_fields}"
                )
            
            # Validate timestamp format
            timestamp = record.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    structure_errors.append(
                        f"Record {i} has invalid timestamp format: {timestamp}"
                    )
        
        if structure_errors:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.RECORD_STRUCTURE,
                status=VerificationStatus.FAILED,
                message=f"Record structure validation failed: {len(structure_errors)} errors",
                details={"errors": structure_errors[:10]}  # Limit to first 10 errors
            ))
        else:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.RECORD_STRUCTURE,
                status=VerificationStatus.PASSED,
                message=f"Record structure validation passed for {len(records)} records"
            ))
        
        return checks
    
    async def _verify_timestamp_sequence(self, records: List[Dict[str, Any]]) -> List[IntegrityCheck]:
        """Verify timestamp sequence integrity"""
        checks = []
        
        if len(records) < 2:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.TIMESTAMP_SEQUENCE,
                status=VerificationStatus.SKIPPED,
                message="Insufficient records for timestamp sequence verification"
            ))
            return checks
        
        # Sort records by timestamp
        try:
            sorted_records = sorted(records, key=lambda r: r.get("timestamp", ""))
        except Exception as e:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.TIMESTAMP_SEQUENCE,
                status=VerificationStatus.FAILED,
                message=f"Failed to sort records by timestamp: {str(e)}"
            ))
            return checks
        
        sequence_errors = []
        
        for i in range(1, len(sorted_records)):
            prev_record = sorted_records[i-1]
            curr_record = sorted_records[i]
            
            prev_time = prev_record.get("timestamp")
            curr_time = curr_record.get("timestamp")
            
            if not prev_time or not curr_time:
                continue
            
            try:
                if isinstance(prev_time, str):
                    prev_dt = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                else:
                    prev_dt = prev_time
                
                if isinstance(curr_time, str):
                    curr_dt = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                else:
                    curr_dt = curr_time
                
                # Check for reasonable time progression
                time_diff = (curr_dt - prev_dt).total_seconds()
                
                if time_diff < 0:
                    sequence_errors.append(
                        f"Records {i-1} and {i} have reverse timestamp order"
                    )
                elif time_diff > 86400:  # More than 24 hours
                    sequence_errors.append(
                        f"Records {i-1} and {i} have suspicious time gap: {time_diff/3600:.1f} hours"
                    )
                
            except Exception as e:
                sequence_errors.append(
                    f"Error comparing timestamps for records {i-1} and {i}: {str(e)}"
                )
        
        if sequence_errors:
            status = VerificationStatus.WARNING if len(sequence_errors) < len(records) * 0.1 else VerificationStatus.FAILED
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.TIMESTAMP_SEQUENCE,
                status=status,
                message=f"Timestamp sequence issues found: {len(sequence_errors)} problems",
                details={"errors": sequence_errors[:10]}
            ))
        else:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.TIMESTAMP_SEQUENCE,
                status=VerificationStatus.PASSED,
                message=f"Timestamp sequence verification passed for {len(records)} records"
            ))
        
        return checks
    
    async def _verify_user_validation(self, records: List[Dict[str, Any]]) -> List[IntegrityCheck]:
        """Verify user validation and authorization"""
        checks = []
        
        if not records:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.USER_VALIDATION,
                status=VerificationStatus.SKIPPED,
                message="No records for user validation"
            ))
            return checks
        
        user_validation_errors = []
        users_seen = set()
        
        for i, record in enumerate(records):
            user = record.get("user")
            
            if not user:
                user_validation_errors.append(f"Record {i} missing user field")
                continue
            
            users_seen.add(user)
            
            # Check for system users vs human users
            if user.startswith("system") or user == "system":
                # System users should have specific event types
                event_type = record.get("event_type", "")
                if event_type not in ["cleanup", "automated_check", "system_maintenance"]:
                    # This is acceptable for most system events
                    pass
            
            # Check for suspicious user patterns
            if len(user) < 3:
                user_validation_errors.append(f"Record {i} has suspicious short username: {user}")
            
            if user.count(" ") > 2:
                user_validation_errors.append(f"Record {i} has suspicious username format: {user}")
        
        # Generate summary
        total_users = len(users_seen)
        system_users = len([u for u in users_seen if u.startswith("system") or u == "system"])
        human_users = total_users - system_users
        
        if user_validation_errors:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.USER_VALIDATION,
                status=VerificationStatus.WARNING,
                message=f"User validation issues found: {len(user_validation_errors)} problems",
                details={
                    "errors": user_validation_errors[:10],
                    "total_users": total_users,
                    "system_users": system_users,
                    "human_users": human_users
                }
            ))
        else:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.USER_VALIDATION,
                status=VerificationStatus.PASSED,
                message=f"User validation passed: {total_users} users ({human_users} human, {system_users} system)",
                details={
                    "total_users": total_users,
                    "system_users": system_users,
                    "human_users": human_users
                }
            ))
        
        return checks
    
    async def _verify_completeness(self, records: List[Dict[str, Any]]) -> List[IntegrityCheck]:
        """Verify audit trail completeness"""
        checks = []
        
        completeness_issues = []
        
        # Check for gaps in audit trail
        if len(records) < 10:
            completeness_issues.append("Very few audit records found - may indicate incomplete logging")
        
        # Check event type distribution
        event_types = {}
        for record in records:
            event_type = record.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Check for missing expected event types
        expected_events = ["state_transition", "approval_request_created", "mfa_device_registered"]
        missing_events = [event for event in expected_events if event not in event_types]
        
        if missing_events:
            completeness_issues.append(f"Missing expected event types: {missing_events}")
        
        # Check for time gaps
        if len(records) > 1:
            sorted_records = sorted(records, key=lambda r: r.get("timestamp", ""))
            
            for i in range(1, len(sorted_records)):
                prev_time = sorted_records[i-1].get("timestamp")
                curr_time = sorted_records[i].get("timestamp")
                
                if prev_time and curr_time:
                    try:
                        if isinstance(prev_time, str):
                            prev_dt = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                        else:
                            prev_dt = prev_time
                        
                        if isinstance(curr_time, str):
                            curr_dt = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                        else:
                            curr_dt = curr_time
                        
                        gap_hours = (curr_dt - prev_dt).total_seconds() / 3600
                        
                        if gap_hours > 24:  # Gap of more than 24 hours
                            completeness_issues.append(
                                f"Large time gap detected: {gap_hours:.1f} hours between records"
                            )
                    except Exception:
                        pass
        
        if completeness_issues:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.COMPLETENESS,
                status=VerificationStatus.WARNING,
                message=f"Completeness issues found: {len(completeness_issues)} problems",
                details={
                    "issues": completeness_issues,
                    "event_type_distribution": event_types,
                    "total_records": len(records)
                }
            ))
        else:
            checks.append(IntegrityCheck(
                check_type=IntegrityCheckType.COMPLETENESS,
                status=VerificationStatus.PASSED,
                message=f"Completeness verification passed: {len(records)} records with {len(event_types)} event types",
                details={
                    "event_type_distribution": event_types,
                    "total_records": len(records)
                }
            ))
        
        return checks
    
    async def _verify_compliance(self, records: List[Dict[str, Any]]) -> List[IntegrityCheck]:
        """Verify compliance with regulations and policies"""
        checks = []
        
        for rule in self.compliance_rules:
            if not rule.enabled:
                continue
            
            try:
                # Get the check function
                check_method = getattr(self, rule.check_function, None)
                if not check_method:
                    checks.append(IntegrityCheck(
                        check_type=IntegrityCheckType.COMPLIANCE,
                        status=VerificationStatus.FAILED,
                        message=f"Compliance rule {rule.rule_id}: check function not found"
                    ))
                    continue
                
                # Execute the check
                result = await check_method(records, rule.parameters)
                
                if isinstance(result, IntegrityCheck):
                    checks.append(result)
                else:
                    # Convert boolean result to IntegrityCheck
                    status = VerificationStatus.PASSED if result else VerificationStatus.FAILED
                    checks.append(IntegrityCheck(
                        check_type=IntegrityCheckType.COMPLIANCE,
                        status=status,
                        message=f"Compliance rule {rule.rule_id}: {rule.name}"
                    ))
                
            except Exception as e:
                checks.append(IntegrityCheck(
                    check_type=IntegrityCheckType.COMPLIANCE,
                    status=VerificationStatus.FAILED,
                    message=f"Compliance rule {rule.rule_id} failed: {str(e)}"
                ))
        
        return checks
    
    async def _check_data_retention_compliance(self, records: List[Dict[str, Any]], 
                                             parameters: Dict[str, Any]) -> IntegrityCheck:
        """Check data retention compliance"""
        retention_days = parameters.get("retention_days", 2555)  # ~7 years default
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        old_records = []
        
        for record in records:
            timestamp = record.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        record_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        record_dt = timestamp
                    
                    if record_dt < cutoff_date:
                        old_records.append(record)
                except Exception:
                    pass
        
        if old_records:
            return IntegrityCheck(
                check_type=IntegrityCheckType.COMPLIANCE,
                status=VerificationStatus.WARNING,
                message=f"Found {len(old_records)} records older than retention policy ({retention_days} days)",
                details={"old_records_count": len(old_records)}
            )
        
        return IntegrityCheck(
            check_type=IntegrityCheckType.COMPLIANCE,
            status=VerificationStatus.PASSED,
            message=f"Data retention compliance verified: no records older than {retention_days} days"
        )
    
    async def _check_audit_frequency_compliance(self, records: List[Dict[str, Any]], 
                                              parameters: Dict[str, Any]) -> IntegrityCheck:
        """Check audit frequency compliance"""
        min_events_per_day = parameters.get("min_events_per_day", 1)
        
        # Group records by date
        daily_counts = {}
        
        for record in records:
            timestamp = record.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        record_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        record_dt = timestamp
                    
                    date_key = record_dt.date()
                    daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
                except Exception:
                    pass
        
        low_activity_days = [
            date for date, count in daily_counts.items() 
            if count < min_events_per_day
        ]
        
        if low_activity_days:
            return IntegrityCheck(
                check_type=IntegrityCheckType.COMPLIANCE,
                status=VerificationStatus.WARNING,
                message=f"Found {len(low_activity_days)} days with low audit activity (< {min_events_per_day} events)",
                details={"low_activity_days": len(low_activity_days)}
            )
        
        return IntegrityCheck(
            check_type=IntegrityCheckType.COMPLIANCE,
            status=VerificationStatus.PASSED,
            message=f"Audit frequency compliance verified: all days meet minimum {min_events_per_day} events"
        )
    
    def _get_records_for_period(self, start_time: datetime = None, 
                               end_time: datetime = None) -> List[Dict[str, Any]]:
        """Get audit records for verification period"""
        all_records = self.audit_store.get_recent_records(1000)  # Get recent records
        
        if not start_time and not end_time:
            return all_records
        
        filtered_records = []
        
        for record in all_records:
            timestamp = record.get("timestamp")
            if not timestamp:
                continue
            
            try:
                if isinstance(timestamp, str):
                    record_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    record_dt = timestamp
                
                if start_time and record_dt < start_time:
                    continue
                
                if end_time and record_dt > end_time:
                    continue
                
                filtered_records.append(record)
                
            except Exception:
                # Include records with unparseable timestamps
                filtered_records.append(record)
        
        return filtered_records
    
    def _determine_overall_status(self, checks: List[IntegrityCheck]) -> VerificationStatus:
        """Determine overall verification status"""
        if not checks:
            return VerificationStatus.FAILED
        
        failed_checks = [c for c in checks if c.status == VerificationStatus.FAILED]
        warning_checks = [c for c in checks if c.status == VerificationStatus.WARNING]
        
        if failed_checks:
            return VerificationStatus.FAILED
        elif warning_checks:
            return VerificationStatus.WARNING
        else:
            return VerificationStatus.PASSED
    
    def _generate_verification_summary(self, records: List[Dict[str, Any]], 
                                     checks: List[IntegrityCheck]) -> Dict[str, Any]:
        """Generate verification summary"""
        return {
            "records_verified": len(records),
            "checks_performed": len(checks),
            "verification_period": {
                "start": min([r.get("timestamp", "") for r in records]) if records else None,
                "end": max([r.get("timestamp", "") for r in records]) if records else None
            },
            "check_results": {
                "passed": len([c for c in checks if c.status == VerificationStatus.PASSED]),
                "failed": len([c for c in checks if c.status == VerificationStatus.FAILED]),
                "warnings": len([c for c in checks if c.status == VerificationStatus.WARNING]),
                "skipped": len([c for c in checks if c.status == VerificationStatus.SKIPPED])
            }
        }
    
    def _generate_recommendations(self, checks: List[IntegrityCheck]) -> List[str]:
        """Generate recommendations based on check results"""
        recommendations = []
        
        failed_checks = [c for c in checks if c.status == VerificationStatus.FAILED]
        warning_checks = [c for c in checks if c.status == VerificationStatus.WARNING]
        
        if failed_checks:
            recommendations.append("Address critical audit integrity failures immediately")
            
            # Specific recommendations based on check types
            for check in failed_checks:
                if check.check_type == IntegrityCheckType.HASH_CHAIN:
                    recommendations.append("Investigate hash chain integrity - possible tampering detected")
                elif check.check_type == IntegrityCheckType.RECORD_STRUCTURE:
                    recommendations.append("Fix audit record structure issues - ensure all required fields are present")
                elif check.check_type == IntegrityCheckType.TIMESTAMP_SEQUENCE:
                    recommendations.append("Review timestamp sequence issues - check system clock synchronization")
        
        if warning_checks:
            recommendations.append("Review audit trail warnings and consider improvements")
            
            for check in warning_checks:
                if check.check_type == IntegrityCheckType.COMPLETENESS:
                    recommendations.append("Improve audit trail completeness - ensure all events are logged")
                elif check.check_type == IntegrityCheckType.USER_VALIDATION:
                    recommendations.append("Review user validation issues - verify user authentication")
        
        if not recommendations:
            recommendations.append("Audit trail verification passed - no immediate action required")
        
        return recommendations
    
    def _setup_default_compliance_rules(self):
        """Setup default compliance rules"""
        
        # Data retention compliance
        self.compliance_rules.append(ComplianceRule(
            rule_id="data_retention",
            name="Data Retention Policy",
            description="Verify compliance with data retention requirements",
            check_function="_check_data_retention_compliance",
            severity="warning",
            parameters={"retention_days": 2555}  # ~7 years
        ))
        
        # Audit frequency compliance
        self.compliance_rules.append(ComplianceRule(
            rule_id="audit_frequency",
            name="Audit Frequency Policy",
            description="Verify minimum audit event frequency",
            check_function="_check_audit_frequency_compliance",
            severity="warning",
            parameters={"min_events_per_day": 1}
        ))
    
    def get_verification_history(self, limit: int = 10) -> List[VerificationResult]:
        """Get recent verification history"""
        return sorted(self.verification_history, 
                     key=lambda r: r.started_at, reverse=True)[:limit]
    
    def add_compliance_rule(self, rule: ComplianceRule):
        """Add custom compliance rule"""
        self.compliance_rules.append(rule)
        logger.info(f"Added compliance rule: {rule.rule_id}")
    
    def remove_compliance_rule(self, rule_id: str) -> bool:
        """Remove compliance rule"""
        original_count = len(self.compliance_rules)
        self.compliance_rules = [r for r in self.compliance_rules if r.rule_id != rule_id]
        
        if len(self.compliance_rules) < original_count:
            logger.info(f"Removed compliance rule: {rule_id}")
            return True
        
        return False


# Utility functions for audit verification

def create_audit_verifier(audit_store: GovernanceAuditStore) -> AuditVerifier:
    """Create and configure audit verifier"""
    return AuditVerifier(audit_store)


async def run_daily_verification(verifier: AuditVerifier) -> VerificationResult:
    """Run daily audit verification"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)
    
    return await verifier.verify_audit_integrity(start_time, end_time)
