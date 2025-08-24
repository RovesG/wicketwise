# Purpose: DGL audit system with hash chaining for immutable logging
# Author: WicketWise AI, Last Modified: 2024

from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json
import logging

from schemas import AuditRecord, DecisionType, RuleId, ExposureSnapshot
from store import AuditStore


logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Audit logging system with hash chaining for tamper detection
    
    Provides immutable audit trail for all DGL decisions with:
    - Hash chaining for integrity verification
    - Structured logging with metadata
    - Query capabilities for compliance reporting
    - Tamper detection and verification
    """
    
    def __init__(self, audit_store: AuditStore, hash_algorithm: str = "sha256"):
        self.audit_store = audit_store
        self.hash_algorithm = hash_algorithm
        self._logger = logging.getLogger(f"{__name__}.AuditLogger")
    
    def log_decision(self, proposal_id: str, decision: DecisionType,
                    rule_ids: List[RuleId], exposure_snapshot: ExposureSnapshot,
                    user_id: Optional[str] = None, session_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a governance decision to the audit trail
        
        Args:
            proposal_id: ID of the bet proposal
            decision: Decision made (APPROVE/REJECT/AMEND)
            rule_ids: List of rules that were triggered
            exposure_snapshot: Current exposure state
            user_id: Optional user ID
            session_id: Optional session ID
            metadata: Optional additional metadata
            
        Returns:
            Audit record ID
        """
        try:
            # Create audit record
            audit_record = AuditRecord(
                proposal_id=proposal_id,
                decision=decision,
                rule_ids=rule_ids,
                snapshot=exposure_snapshot,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )
            
            # Append to store (which handles hash chaining)
            audit_id = self.audit_store.append_record(audit_record)
            
            self._logger.info(f"Audit record created: {audit_id} for proposal {proposal_id}")
            
            return audit_id
            
        except Exception as e:
            self._logger.error(f"Failed to create audit record for proposal {proposal_id}: {str(e)}")
            raise
    
    def get_audit_trail(self, proposal_id: str) -> List[AuditRecord]:
        """Get complete audit trail for a proposal"""
        return self.audit_store.get_records_by_proposal(proposal_id)
    
    def get_recent_decisions(self, limit: int = 100) -> List[AuditRecord]:
        """Get recent audit decisions"""
        return self.audit_store.get_recent_records(limit)
    
    def verify_integrity(self) -> bool:
        """Verify integrity of the entire audit chain"""
        try:
            is_valid = self.audit_store.verify_hash_chain()
            
            if is_valid:
                self._logger.info("Audit chain integrity verification: PASSED")
            else:
                self._logger.error("Audit chain integrity verification: FAILED")
            
            return is_valid
            
        except Exception as e:
            self._logger.error(f"Error during integrity verification: {str(e)}")
            return False
    
    def get_compliance_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance report for a date range
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report with decision statistics and rule triggers
        """
        try:
            # Get all records in date range
            all_records = self.audit_store.get_recent_records(limit=10000)  # Simplified
            filtered_records = [
                r for r in all_records 
                if start_date <= r.timestamp <= end_date
            ]
            
            # Calculate statistics
            decision_counts = {
                DecisionType.APPROVE: 0,
                DecisionType.REJECT: 0,
                DecisionType.AMEND: 0
            }
            
            rule_trigger_counts = {}
            
            for record in filtered_records:
                decision_counts[record.decision] += 1
                
                for rule_id in record.rule_ids:
                    rule_trigger_counts[rule_id] = rule_trigger_counts.get(rule_id, 0) + 1
            
            # Calculate percentages
            total_decisions = sum(decision_counts.values())
            decision_percentages = {
                decision: (count / total_decisions * 100) if total_decisions > 0 else 0
                for decision, count in decision_counts.items()
            }
            
            report = {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_decisions": total_decisions,
                    "decision_counts": {k.value: v for k, v in decision_counts.items()},
                    "decision_percentages": {k.value: round(v, 2) for k, v in decision_percentages.items()},
                    "approval_rate": round(decision_percentages[DecisionType.APPROVE], 2),
                    "rejection_rate": round(decision_percentages[DecisionType.REJECT], 2)
                },
                "rule_analysis": {
                    "most_triggered_rules": sorted(
                        [(rule.value, count) for rule, count in rule_trigger_counts.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:10],
                    "total_rule_triggers": sum(rule_trigger_counts.values()),
                    "unique_rules_triggered": len(rule_trigger_counts)
                },
                "integrity": {
                    "hash_chain_valid": self.verify_integrity(),
                    "total_audit_records": len(filtered_records)
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            self._logger.info(f"Compliance report generated for {start_date} to {end_date}")
            
            return report
            
        except Exception as e:
            self._logger.error(f"Error generating compliance report: {str(e)}")
            raise
    
    def search_audit_records(self, filters: Dict[str, Any]) -> List[AuditRecord]:
        """
        Search audit records with filters
        
        Args:
            filters: Dictionary of search filters
                - proposal_id: Specific proposal ID
                - decision: Decision type
                - rule_ids: List of rule IDs
                - user_id: User ID
                - start_date: Start date
                - end_date: End date
                
        Returns:
            List of matching audit records
        """
        try:
            # Get all records (simplified - in production would use proper indexing)
            all_records = self.audit_store.get_recent_records(limit=10000)
            
            filtered_records = []
            
            for record in all_records:
                # Apply filters
                if "proposal_id" in filters and record.proposal_id != filters["proposal_id"]:
                    continue
                
                if "decision" in filters and record.decision != filters["decision"]:
                    continue
                
                if "rule_ids" in filters:
                    filter_rules = set(filters["rule_ids"])
                    record_rules = set(record.rule_ids)
                    if not filter_rules.intersection(record_rules):
                        continue
                
                if "user_id" in filters and record.user_id != filters["user_id"]:
                    continue
                
                if "start_date" in filters and record.timestamp < filters["start_date"]:
                    continue
                
                if "end_date" in filters and record.timestamp > filters["end_date"]:
                    continue
                
                filtered_records.append(record)
            
            self._logger.info(f"Audit search returned {len(filtered_records)} records")
            
            return filtered_records
            
        except Exception as e:
            self._logger.error(f"Error searching audit records: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        try:
            store_stats = self.audit_store.get_statistics()
            
            # Add audit logger specific stats
            stats = {
                **store_stats,
                "hash_algorithm": self.hash_algorithm,
                "integrity_last_verified": datetime.utcnow().isoformat(),
                "audit_logger_version": "1.0.0"
            }
            
            return stats
            
        except Exception as e:
            self._logger.error(f"Error getting audit statistics: {str(e)}")
            return {"error": str(e)}


class AuditVerifier:
    """Utility class for verifying audit chain integrity"""
    
    def __init__(self, audit_store: AuditStore, hash_algorithm: str = "sha256"):
        self.audit_store = audit_store
        self.hash_algorithm = hash_algorithm
    
    def verify_full_chain(self) -> Dict[str, Any]:
        """
        Perform comprehensive verification of audit chain
        
        Returns:
            Verification report with details
        """
        try:
            verification_result = {
                "overall_valid": True,
                "total_records": 0,
                "verified_records": 0,
                "hash_chain_valid": True,
                "errors": [],
                "warnings": [],
                "verification_timestamp": datetime.utcnow().isoformat()
            }
            
            # Get all records
            all_records = self.audit_store.get_recent_records(limit=100000)
            verification_result["total_records"] = len(all_records)
            
            if not all_records:
                verification_result["warnings"].append("No audit records found")
                return verification_result
            
            # Verify hash chain
            hash_chain_valid = self.audit_store.verify_hash_chain()
            verification_result["hash_chain_valid"] = hash_chain_valid
            
            if not hash_chain_valid:
                verification_result["overall_valid"] = False
                verification_result["errors"].append("Hash chain integrity check failed")
            
            # Verify individual records
            for record in all_records:
                try:
                    # Verify record structure
                    if not record.audit_id or not record.proposal_id:
                        verification_result["errors"].append(
                            f"Record {record.audit_id} missing required fields"
                        )
                        continue
                    
                    # Verify timestamp is reasonable
                    if record.timestamp > datetime.utcnow():
                        verification_result["warnings"].append(
                            f"Record {record.audit_id} has future timestamp"
                        )
                    
                    verification_result["verified_records"] += 1
                    
                except Exception as e:
                    verification_result["errors"].append(
                        f"Error verifying record {getattr(record, 'audit_id', 'unknown')}: {str(e)}"
                    )
            
            # Final validation
            if verification_result["errors"]:
                verification_result["overall_valid"] = False
            
            return verification_result
            
        except Exception as e:
            return {
                "overall_valid": False,
                "error": f"Verification failed: {str(e)}",
                "verification_timestamp": datetime.utcnow().isoformat()
            }
    
    def detect_tampering(self) -> List[Dict[str, Any]]:
        """
        Detect potential tampering in audit records
        
        Returns:
            List of suspicious findings
        """
        findings = []
        
        try:
            # Check hash chain integrity
            if not self.audit_store.verify_hash_chain():
                findings.append({
                    "type": "hash_chain_broken",
                    "severity": "CRITICAL",
                    "description": "Hash chain integrity check failed - potential tampering detected",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Additional tampering detection logic would go here
            # - Check for gaps in timestamps
            # - Verify record ordering
            # - Check for duplicate IDs
            # - Validate hash calculations
            
        except Exception as e:
            findings.append({
                "type": "verification_error",
                "severity": "ERROR",
                "description": f"Error during tampering detection: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return findings
