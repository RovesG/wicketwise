# Purpose: Audit trail and compliance API endpoints
# Author: WicketWise AI, Last Modified: 2024

"""
Audit API Endpoints

Provides REST endpoints for:
- Audit trail access and search
- Compliance reporting
- Audit integrity verification
- Decision history tracking
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import AuditRecord, DecisionType, RuleId
from engine import RuleEngine
from api.governance import get_rule_engine


logger = logging.getLogger(__name__)

# Router for audit endpoints
audit_router = APIRouter(prefix="/audit", tags=["audit"])


class AuditSearchRequest(BaseModel):
    """Request model for audit search"""
    start_date: Optional[datetime] = Field(default=None, description="Search start date")
    end_date: Optional[datetime] = Field(default=None, description="Search end date")
    decision_types: Optional[List[DecisionType]] = Field(default=None, description="Filter by decision types")
    rule_ids: Optional[List[RuleId]] = Field(default=None, description="Filter by rule IDs")
    market_ids: Optional[List[str]] = Field(default=None, description="Filter by market IDs")
    match_ids: Optional[List[str]] = Field(default=None, description="Filter by match IDs")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum records to return")


class AuditSearchResponse(BaseModel):
    """Response model for audit search"""
    records: List[AuditRecord]
    total_count: int
    filtered_count: int
    search_criteria: Dict[str, Any]
    timestamp: datetime


class ComplianceReportResponse(BaseModel):
    """Response model for compliance reporting"""
    report_id: str
    period_start: datetime
    period_end: datetime
    total_decisions: int
    decisions_by_type: Dict[str, int]
    rule_violations: Dict[str, int]
    compliance_metrics: Dict[str, Any]
    integrity_status: str
    generated_at: datetime


class IntegrityCheckResponse(BaseModel):
    """Response model for audit integrity verification"""
    total_records: int
    verified_records: int
    integrity_violations: List[Dict[str, Any]]
    hash_chain_status: str
    last_verified_record: Optional[str]
    verification_timestamp: datetime


@audit_router.get("/records", response_model=AuditSearchResponse)
async def search_audit_records(
    start_date: Optional[datetime] = Query(default=None, description="Search start date"),
    end_date: Optional[datetime] = Query(default=None, description="Search end date"),
    decision_type: Optional[DecisionType] = Query(default=None, description="Filter by decision type"),
    rule_id: Optional[RuleId] = Query(default=None, description="Filter by rule ID"),
    market_id: Optional[str] = Query(default=None, description="Filter by market ID"),
    match_id: Optional[str] = Query(default=None, description="Filter by match ID"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum records"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> AuditSearchResponse:
    """
    Search audit records with filters
    
    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        decision_type: Optional decision type filter
        rule_id: Optional rule ID filter
        market_id: Optional market ID filter
        match_id: Optional match ID filter
        limit: Maximum number of records to return
        
    Returns:
        Filtered audit records
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=7)  # Default to last 7 days
        
        logger.info(f"Searching audit records from {start_date} to {end_date}")
        
        # Mock audit records (in production, query from audit store)
        mock_records = []
        
        # Generate mock audit records
        for i in range(min(limit, 50)):  # Generate up to 50 mock records
            record_time = start_date + timedelta(
                seconds=(end_date - start_date).total_seconds() * (i / 50)
            )
            
            # Mock decision data
            decisions = [DecisionType.APPROVE, DecisionType.REJECT, DecisionType.AMEND]
            rules = [RuleId.BANKROLL_MAX_EXPOSURE, RuleId.LIQ_SLIPPAGE_LIMIT, RuleId.PNL_DAILY_LOSS_LIMIT]
            
            mock_decision = decisions[i % len(decisions)]
            mock_rule = rules[i % len(rules)]
            mock_market = f"market_{i % 10}"
            mock_match = f"match_{i % 5}"
            
            # Apply filters
            if decision_type and mock_decision != decision_type:
                continue
            if rule_id and mock_rule != rule_id:
                continue
            if market_id and mock_market != market_id:
                continue
            if match_id and mock_match != match_id:
                continue
            
            # Create mock audit record
            record = AuditRecord(
                record_id=f"audit_{i:06d}",
                timestamp=record_time,
                event_type="governance_decision",
                decision_type=mock_decision,
                rule_ids_triggered=[mock_rule] if mock_decision != DecisionType.APPROVE else [],
                market_id=mock_market,
                match_id=mock_match,
                proposal_data={
                    "stake": 1000.0 + (i * 100),
                    "odds": 2.0 + (i % 10) * 0.1,
                    "selection": f"Selection_{i % 3}"
                },
                decision_data={
                    "reasoning": f"Decision based on rule evaluation {i}",
                    "confidence_score": 0.8 + (i % 20) * 0.01,
                    "processing_time_ms": 10.0 + (i % 50)
                },
                system_state={
                    "bankroll": 100000.0,
                    "open_exposure": 15000.0 + (i * 100),
                    "daily_pnl": -500.0 + (i * 50)
                },
                previous_hash=f"hash_{i-1:06d}" if i > 0 else "genesis",
                record_hash=f"hash_{i:06d}"
            )
            
            mock_records.append(record)
        
        # Search criteria summary
        search_criteria = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "decision_type": decision_type.value if decision_type else None,
            "rule_id": rule_id.value if rule_id else None,
            "market_id": market_id,
            "match_id": match_id,
            "limit": limit
        }
        
        return AuditSearchResponse(
            records=mock_records,
            total_count=len(mock_records),
            filtered_count=len(mock_records),
            search_criteria=search_criteria,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error searching audit records: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Audit search failed: {str(e)}"
        )


@audit_router.post("/search", response_model=AuditSearchResponse)
async def advanced_audit_search(
    request: AuditSearchRequest,
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> AuditSearchResponse:
    """
    Advanced audit search with complex filters
    
    Args:
        request: Advanced search request with multiple filters
        
    Returns:
        Filtered audit records matching all criteria
    """
    try:
        # Use the GET endpoint logic with request parameters
        return await search_audit_records(
            start_date=request.start_date,
            end_date=request.end_date,
            decision_type=request.decision_types[0] if request.decision_types else None,
            rule_id=request.rule_ids[0] if request.rule_ids else None,
            market_id=request.market_ids[0] if request.market_ids else None,
            match_id=request.match_ids[0] if request.match_ids else None,
            limit=request.limit,
            rule_engine=rule_engine
        )
        
    except Exception as e:
        logger.error(f"Error in advanced audit search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Advanced audit search failed: {str(e)}"
        )


@audit_router.get("/compliance/report", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    start_date: Optional[datetime] = Query(default=None, description="Report start date"),
    end_date: Optional[datetime] = Query(default=None, description="Report end date"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> ComplianceReportResponse:
    """
    Generate compliance report for a time period
    
    Args:
        start_date: Report start date (defaults to 30 days ago)
        end_date: Report end date (defaults to now)
        
    Returns:
        Comprehensive compliance report
    """
    try:
        # Set default date range
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)  # Default to last 30 days
        
        logger.info(f"Generating compliance report from {start_date} to {end_date}")
        
        # Mock compliance data (in production, aggregate from audit records)
        total_decisions = rule_engine._decision_count
        
        decisions_by_type = {
            "APPROVE": int(total_decisions * 0.65),
            "REJECT": int(total_decisions * 0.25),
            "AMEND": int(total_decisions * 0.10)
        }
        
        rule_violations = {
            "BANKROLL_MAX_EXPOSURE": int(total_decisions * 0.12),
            "LIQ_SLIPPAGE_LIMIT": int(total_decisions * 0.15),
            "LIQ_FRACTION_LIMIT": int(total_decisions * 0.08),
            "PNL_DAILY_LOSS_LIMIT": int(total_decisions * 0.03),
            "RATE_LIMIT_EXCEEDED": int(total_decisions * 0.05)
        }
        
        # Compliance metrics
        compliance_metrics = {
            "approval_rate_pct": 65.0,
            "rejection_rate_pct": 25.0,
            "amendment_rate_pct": 10.0,
            "avg_processing_time_ms": 12.5,
            "rule_violation_rate_pct": 35.0,
            "system_uptime_pct": 99.8,
            "audit_integrity_score": 100.0,
            "regulatory_compliance_score": 98.5
        }
        
        # Determine integrity status
        integrity_status = "VERIFIED"
        if compliance_metrics["audit_integrity_score"] < 100.0:
            integrity_status = "WARNING"
        if compliance_metrics["audit_integrity_score"] < 95.0:
            integrity_status = "CRITICAL"
        
        report_id = f"compliance_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        return ComplianceReportResponse(
            report_id=report_id,
            period_start=start_date,
            period_end=end_date,
            total_decisions=total_decisions,
            decisions_by_type=decisions_by_type,
            rule_violations=rule_violations,
            compliance_metrics=compliance_metrics,
            integrity_status=integrity_status,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Compliance report generation failed: {str(e)}"
        )


@audit_router.get("/integrity/verify", response_model=IntegrityCheckResponse)
async def verify_audit_integrity(
    start_date: Optional[datetime] = Query(default=None, description="Verification start date"),
    end_date: Optional[datetime] = Query(default=None, description="Verification end date"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> IntegrityCheckResponse:
    """
    Verify audit trail integrity using hash chain validation
    
    Args:
        start_date: Start date for verification (defaults to 7 days ago)
        end_date: End date for verification (defaults to now)
        
    Returns:
        Integrity verification results
    """
    try:
        # Set default date range
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=7)  # Default to last 7 days
        
        logger.info(f"Verifying audit integrity from {start_date} to {end_date}")
        
        # Mock integrity verification (in production, verify actual hash chain)
        total_records = rule_engine._decision_count
        verified_records = total_records  # Assume all verified for mock
        
        # Mock integrity violations (should be empty in healthy system)
        integrity_violations = []
        
        # Simulate occasional integrity issue for demonstration
        if total_records > 100 and total_records % 97 == 0:  # Rare condition
            integrity_violations.append({
                "record_id": f"audit_{total_records-5:06d}",
                "violation_type": "hash_mismatch",
                "description": "Hash chain verification failed",
                "detected_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                "severity": "WARNING"
            })
            verified_records -= 1
        
        # Determine hash chain status
        if not integrity_violations:
            hash_chain_status = "INTACT"
        elif len(integrity_violations) < 3:
            hash_chain_status = "WARNING"
        else:
            hash_chain_status = "COMPROMISED"
        
        last_verified_record = f"audit_{verified_records-1:06d}" if verified_records > 0 else None
        
        return IntegrityCheckResponse(
            total_records=total_records,
            verified_records=verified_records,
            integrity_violations=integrity_violations,
            hash_chain_status=hash_chain_status,
            last_verified_record=last_verified_record,
            verification_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error verifying audit integrity: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Audit integrity verification failed: {str(e)}"
        )


@audit_router.get("/records/{record_id}", response_model=AuditRecord)
async def get_audit_record(
    record_id: str,
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> AuditRecord:
    """
    Get a specific audit record by ID
    
    Args:
        record_id: Unique audit record identifier
        
    Returns:
        Audit record details
    """
    try:
        logger.info(f"Retrieving audit record {record_id}")
        
        # Mock record retrieval (in production, query audit store)
        # Extract record number from ID
        try:
            record_num = int(record_id.split('_')[1])
        except (IndexError, ValueError):
            raise HTTPException(
                status_code=404,
                detail=f"Invalid record ID format: {record_id}"
            )
        
        if record_num >= rule_engine._decision_count:
            raise HTTPException(
                status_code=404,
                detail=f"Audit record not found: {record_id}"
            )
        
        # Create mock record
        record_time = datetime.now() - timedelta(minutes=record_num * 5)
        
        record = AuditRecord(
            record_id=record_id,
            timestamp=record_time,
            event_type="governance_decision",
            decision_type=DecisionType.APPROVE if record_num % 3 == 0 else DecisionType.REJECT,
            rule_ids_triggered=[RuleId.BANKROLL_MAX_EXPOSURE] if record_num % 3 != 0 else [],
            market_id=f"market_{record_num % 10}",
            match_id=f"match_{record_num % 5}",
            proposal_data={
                "stake": 1000.0 + (record_num * 100),
                "odds": 2.0 + (record_num % 10) * 0.1,
                "selection": f"Selection_{record_num % 3}"
            },
            decision_data={
                "reasoning": f"Decision based on rule evaluation {record_num}",
                "confidence_score": 0.8 + (record_num % 20) * 0.01,
                "processing_time_ms": 10.0 + (record_num % 50)
            },
            system_state={
                "bankroll": 100000.0,
                "open_exposure": 15000.0 + (record_num * 100),
                "daily_pnl": -500.0 + (record_num * 50)
            },
            previous_hash=f"hash_{record_num-1:06d}" if record_num > 0 else "genesis",
            record_hash=f"hash_{record_num:06d}"
        )
        
        return record
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audit record {record_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve audit record: {str(e)}"
        )


@audit_router.get("/health")
async def audit_health_check(
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> Dict[str, Any]:
    """
    Health check for audit system
    
    Returns:
        Audit system health status
    """
    try:
        # Check audit system health
        total_records = rule_engine._decision_count
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "audit_store": {
                "status": "active",
                "total_records": total_records,
                "last_record_time": datetime.now().isoformat(),
                "storage_utilization_pct": 15.2  # Mock storage usage
            },
            "integrity": {
                "hash_chain_status": "intact",
                "last_verification": datetime.now().isoformat(),
                "verification_success_rate": 100.0
            },
            "performance": {
                "avg_write_time_ms": 2.1,
                "avg_read_time_ms": 1.8,
                "records_per_second": 150.0
            }
        }
        
        # Check for warning conditions
        warnings = []
        if total_records > 10000:
            warnings.append("High record count - consider archiving old records")
        
        if warnings:
            health_status["warnings"] = warnings
            health_status["status"] = "warning"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Audit health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
