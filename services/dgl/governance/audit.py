# Purpose: Simple audit store for governance events
# Author: WicketWise AI, Last Modified: 2024

"""
Governance Audit Store

Simple audit logging for governance events that doesn't require
the complex DGL AuditRecord schema.
"""

from typing import Dict, Any, List
from datetime import datetime
import uuid


class GovernanceAuditStore:
    """Simple in-memory audit store for governance events"""
    
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
    
    def append_record(self, record: Dict[str, Any]) -> str:
        """Append new audit record and return its ID"""
        # Add ID if not present
        if "audit_id" not in record:
            record["audit_id"] = str(uuid.uuid4())
        
        # Add timestamp if not present
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()
        
        self.records.append(record.copy())
        return record["audit_id"]
    
    def get_record(self, audit_id: str) -> Dict[str, Any]:
        """Get specific audit record by ID"""
        for record in self.records:
            if record.get("audit_id") == audit_id:
                return record
        return None
    
    def get_records_by_proposal(self, proposal_id: str) -> List[Dict[str, Any]]:
        """Get all audit records for a proposal"""
        return [r for r in self.records if r.get("proposal_id") == proposal_id]
    
    def get_recent_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent audit records"""
        return self.records[-limit:]
    
    def get_recent_decisions_for_market(self, market_id: str, seconds: int = 300) -> List[Dict[str, Any]]:
        """Get recent decisions for a specific market"""
        # Simple implementation - return records mentioning the market
        return [r for r in self.records if market_id in str(r.get("details", {}))]
    
    def verify_integrity(self) -> bool:
        """Verify audit log integrity (simplified)"""
        return True  # Always valid for this simple implementation
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit store statistics"""
        return {
            "total_records": len(self.records),
            "event_types": list(set(r.get("event_type") for r in self.records)),
            "users": list(set(r.get("user") for r in self.records))
        }
