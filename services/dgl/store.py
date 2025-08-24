# Purpose: DGL storage interfaces for exposures, P&L, and audit data
# Author: WicketWise AI, Last Modified: 2024

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from schemas import ExposureSnapshot, AuditRecord, DecisionType


class ExposureStore(ABC):
    """Abstract interface for exposure data storage"""
    
    @abstractmethod
    def get_current_exposure(self) -> ExposureSnapshot:
        """Get current exposure snapshot"""
        pass
    
    @abstractmethod
    def update_exposure(self, match_id: str, market_id: str, 
                       correlation_group: Optional[str], 
                       exposure_delta: float) -> None:
        """Update exposure for match/market/correlation group"""
        pass
    
    @abstractmethod
    def set_bankroll(self, bankroll: float) -> None:
        """Set current bankroll amount"""
        pass
    
    @abstractmethod
    def get_match_exposure(self, match_id: str) -> float:
        """Get total exposure for a specific match"""
        pass
    
    @abstractmethod
    def get_market_exposure(self, market_id: str) -> float:
        """Get total exposure for a specific market"""
        pass
    
    @abstractmethod
    def get_correlation_group_exposure(self, correlation_group: str) -> float:
        """Get total exposure for a correlation group"""
        pass


class PnLStore(ABC):
    """Abstract interface for P&L data storage"""
    
    @abstractmethod
    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """Get P&L for a specific date (default: today)"""
        pass
    
    @abstractmethod
    def get_session_pnl(self) -> float:
        """Get P&L for current session"""
        pass
    
    @abstractmethod
    def update_pnl(self, amount: float, timestamp: Optional[datetime] = None) -> None:
        """Update P&L with realized gain/loss"""
        pass
    
    @abstractmethod
    def start_new_session(self) -> None:
        """Start a new trading session"""
        pass
    
    @abstractmethod
    def get_pnl_history(self, days: int = 30) -> Dict[str, float]:
        """Get P&L history for specified number of days"""
        pass


class AuditStore(ABC):
    """Abstract interface for audit log storage"""
    
    @abstractmethod
    def append_record(self, record: AuditRecord) -> str:
        """Append new audit record and return its ID"""
        pass
    
    @abstractmethod
    def get_record(self, audit_id: str) -> Optional[AuditRecord]:
        """Get specific audit record by ID"""
        pass
    
    @abstractmethod
    def get_records_by_proposal(self, proposal_id: str) -> List[AuditRecord]:
        """Get all audit records for a proposal"""
        pass
    
    @abstractmethod
    def get_recent_records(self, limit: int = 100) -> List[AuditRecord]:
        """Get most recent audit records"""
        pass
    
    @abstractmethod
    def get_recent_decisions_for_market(self, market_id: str, 
                                      seconds: int = 300) -> List[AuditRecord]:
        """Get recent decisions for a specific market"""
        pass
    
    @abstractmethod
    def verify_hash_chain(self) -> bool:
        """Verify integrity of hash chain"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict:
        """Get audit store statistics"""
        pass
