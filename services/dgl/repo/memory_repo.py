# Purpose: In-memory mock implementations of DGL storage interfaces
# Author: WicketWise AI, Last Modified: 2024

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import json
import threading

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import ExposureSnapshot, AuditRecord, DecisionType
from store import ExposureStore, PnLStore, AuditStore


class MemoryExposureStore(ExposureStore):
    """In-memory implementation of ExposureStore"""
    
    def __init__(self, initial_bankroll: float = 100000.0):
        self._bankroll = initial_bankroll
        self._open_exposure = 0.0
        self._per_match_exposure: Dict[str, float] = defaultdict(float)
        self._per_market_exposure: Dict[str, float] = defaultdict(float)
        self._per_correlation_group: Dict[str, float] = defaultdict(float)
        self._session_start = datetime.utcnow()
        self._lock = threading.RLock()
    
    def get_current_exposure(self) -> ExposureSnapshot:
        """Get current exposure snapshot"""
        with self._lock:
            return ExposureSnapshot(
                bankroll=self._bankroll,
                open_exposure=self._open_exposure,
                daily_pnl=0.0,  # Would be calculated from PnL store
                session_pnl=0.0,  # Would be calculated from PnL store
                per_match_exposure=dict(self._per_match_exposure),
                per_market_exposure=dict(self._per_market_exposure),
                per_correlation_group=dict(self._per_correlation_group),
                session_start=self._session_start
            )
    
    def update_exposure(self, match_id: str, market_id: str, 
                       correlation_group: Optional[str], 
                       exposure_delta: float) -> None:
        """Update exposure for match/market/correlation group"""
        with self._lock:
            self._open_exposure += exposure_delta
            self._per_match_exposure[match_id] += exposure_delta
            self._per_market_exposure[market_id] += exposure_delta
            
            if correlation_group:
                self._per_correlation_group[correlation_group] += exposure_delta
            
            # Clean up zero exposures
            if self._per_match_exposure[match_id] == 0:
                del self._per_match_exposure[match_id]
            if self._per_market_exposure[market_id] == 0:
                del self._per_market_exposure[market_id]
            if correlation_group and self._per_correlation_group[correlation_group] == 0:
                del self._per_correlation_group[correlation_group]
    
    def set_bankroll(self, bankroll: float) -> None:
        """Set current bankroll amount"""
        with self._lock:
            self._bankroll = bankroll
    
    def get_match_exposure(self, match_id: str) -> float:
        """Get total exposure for a specific match"""
        with self._lock:
            return self._per_match_exposure.get(match_id, 0.0)
    
    def get_market_exposure(self, market_id: str) -> float:
        """Get total exposure for a specific market"""
        with self._lock:
            return self._per_market_exposure.get(market_id, 0.0)
    
    def get_correlation_group_exposure(self, correlation_group: str) -> float:
        """Get total exposure for a correlation group"""
        with self._lock:
            return self._per_correlation_group.get(correlation_group, 0.0)


class MemoryPnLStore(PnLStore):
    """In-memory implementation of PnLStore"""
    
    def __init__(self):
        self._daily_pnl: Dict[str, float] = defaultdict(float)  # date -> pnl
        self._session_pnl = 0.0
        self._session_start = datetime.utcnow()
        self._pnl_history: List[tuple] = []  # (timestamp, amount)
        self._lock = threading.RLock()
    
    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Get P&L for a specific date (default: today)"""
        with self._lock:
            if date is None:
                date = datetime.utcnow()
            
            date_key = date.strftime("%Y-%m-%d")
            return self._daily_pnl[date_key]
    
    def get_session_pnl(self) -> float:
        """Get P&L for current session"""
        with self._lock:
            return self._session_pnl
    
    def update_pnl(self, amount: float, timestamp: Optional[datetime] = None) -> None:
        """Update P&L with realized gain/loss"""
        with self._lock:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            date_key = timestamp.strftime("%Y-%m-%d")
            self._daily_pnl[date_key] += amount
            
            # Only add to session P&L if it's from current session
            if timestamp >= self._session_start:
                self._session_pnl += amount
            
            self._pnl_history.append((timestamp, amount))
            
            # Keep only last 10000 entries
            if len(self._pnl_history) > 10000:
                self._pnl_history = self._pnl_history[-10000:]
    
    def start_new_session(self) -> None:
        """Start a new trading session"""
        with self._lock:
            self._session_pnl = 0.0
            self._session_start = datetime.utcnow()
    
    def get_pnl_history(self, days: int = 30) -> Dict[str, float]:
        """Get P&L history for specified number of days"""
        with self._lock:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            result = {}
            for date_key, pnl in self._daily_pnl.items():
                date_obj = datetime.strptime(date_key, "%Y-%m-%d")
                if date_obj >= cutoff_date:
                    result[date_key] = pnl
            
            return result


class MemoryAuditStore(AuditStore):
    """In-memory implementation of AuditStore with hash chaining"""
    
    def __init__(self):
        self._records: Dict[str, AuditRecord] = {}
        self._records_by_proposal: Dict[str, List[str]] = defaultdict(list)
        self._ordered_records: List[str] = []  # Maintains insertion order
        self._last_hash: Optional[str] = None
        self._lock = threading.RLock()
    
    def append_record(self, record: AuditRecord) -> str:
        """Append new audit record and return its ID"""
        with self._lock:
            # Set hash chain
            record.hash_prev = self._last_hash
            record.hash_curr = self._calculate_record_hash(record)
            
            # Store record
            self._records[record.audit_id] = record
            self._records_by_proposal[record.proposal_id].append(record.audit_id)
            self._ordered_records.append(record.audit_id)
            
            # Update last hash
            self._last_hash = record.hash_curr
            
            return record.audit_id
    
    def get_record(self, audit_id: str) -> Optional[AuditRecord]:
        """Get specific audit record by ID"""
        with self._lock:
            return self._records.get(audit_id)
    
    def get_records_by_proposal(self, proposal_id: str) -> List[AuditRecord]:
        """Get all audit records for a proposal"""
        with self._lock:
            record_ids = self._records_by_proposal.get(proposal_id, [])
            return [self._records[rid] for rid in record_ids if rid in self._records]
    
    def get_recent_records(self, limit: int = 100) -> List[AuditRecord]:
        """Get most recent audit records"""
        with self._lock:
            recent_ids = self._ordered_records[-limit:] if self._ordered_records else []
            return [self._records[rid] for rid in reversed(recent_ids) if rid in self._records]
    
    def get_recent_decisions_for_market(self, market_id: str, 
                                      seconds: int = 300) -> List[AuditRecord]:
        """Get recent decisions for a specific market"""
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=seconds)
            
            recent_records = []
            for record in self._records.values():
                if (record.timestamp >= cutoff_time and 
                    hasattr(record, 'market_id') and 
                    getattr(record, 'market_id', None) == market_id):
                    recent_records.append(record)
            
            # Sort by timestamp, most recent first
            return sorted(recent_records, key=lambda r: r.timestamp, reverse=True)
    
    def verify_hash_chain(self) -> bool:
        """Verify integrity of hash chain"""
        with self._lock:
            if not self._ordered_records:
                return True
            
            prev_hash = None
            for record_id in self._ordered_records:
                record = self._records.get(record_id)
                if not record:
                    return False
                
                # Check previous hash matches
                if record.hash_prev != prev_hash:
                    return False
                
                # Verify current hash
                calculated_hash = self._calculate_record_hash(record)
                if record.hash_curr != calculated_hash:
                    return False
                
                prev_hash = record.hash_curr
            
            return True
    
    def get_statistics(self) -> Dict:
        """Get audit store statistics"""
        with self._lock:
            decision_counts = defaultdict(int)
            for record in self._records.values():
                decision_counts[record.decision.value] += 1
            
            return {
                "total_records": len(self._records),
                "decision_counts": dict(decision_counts),
                "hash_chain_valid": self.verify_hash_chain(),
                "oldest_record": (
                    min(r.timestamp for r in self._records.values()).isoformat() 
                    if self._records else None
                ),
                "newest_record": (
                    max(r.timestamp for r in self._records.values()).isoformat() 
                    if self._records else None
                )
            }
    
    def _calculate_record_hash(self, record: AuditRecord) -> str:
        """Calculate SHA-256 hash of audit record"""
        # Create a deterministic representation of the record
        hash_data = {
            "audit_id": record.audit_id,
            "timestamp": record.timestamp.isoformat(),
            "entity": record.entity,
            "proposal_id": record.proposal_id,
            "decision": record.decision.value,
            "rule_ids": [rid.value for rid in record.rule_ids],
            "snapshot": {
                "bankroll": record.snapshot.bankroll,
                "open_exposure": record.snapshot.open_exposure,
                "daily_pnl": record.snapshot.daily_pnl,
                "session_pnl": record.snapshot.session_pnl
            },
            "hash_prev": record.hash_prev
        }
        
        # Convert to JSON string with sorted keys for deterministic hashing
        json_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        
        # Calculate SHA-256 hash
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


class MemoryRepositoryFactory:
    """Factory for creating memory-based repository instances"""
    
    @staticmethod
    def create_exposure_store(initial_bankroll: float = 100000.0) -> MemoryExposureStore:
        """Create memory-based exposure store"""
        return MemoryExposureStore(initial_bankroll)
    
    @staticmethod
    def create_pnl_store() -> MemoryPnLStore:
        """Create memory-based P&L store"""
        return MemoryPnLStore()
    
    @staticmethod
    def create_audit_store() -> MemoryAuditStore:
        """Create memory-based audit store"""
        return MemoryAuditStore()
    
    @staticmethod
    def create_all_stores(initial_bankroll: float = 100000.0) -> tuple:
        """Create all memory-based stores"""
        return (
            MemoryRepositoryFactory.create_exposure_store(initial_bankroll),
            MemoryRepositoryFactory.create_pnl_store(),
            MemoryRepositoryFactory.create_audit_store()
        )
