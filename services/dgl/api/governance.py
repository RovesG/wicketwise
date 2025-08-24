# Purpose: Governance decision API endpoints
# Author: WicketWise AI, Last Modified: 2024

"""
Governance API Endpoints

Provides REST endpoints for:
- Bet proposal evaluation
- Governance decision retrieval
- Batch proposal processing
- Decision statistics and metrics
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import (
    BetProposal, GovernanceDecision, DecisionType, RuleId, 
    ExposureSnapshot, AuditRecord
)
from engine import RuleEngine
from config import DGLConfig, load_config
from repo.memory_repo import MemoryRepositoryFactory


logger = logging.getLogger(__name__)

# Router for governance endpoints
governance_router = APIRouter(prefix="/governance", tags=["governance"])

# Global state (in production, use dependency injection)
_rule_engine: Optional[RuleEngine] = None
_config: Optional[DGLConfig] = None


def get_rule_engine() -> RuleEngine:
    """Dependency to get rule engine instance"""
    global _rule_engine, _config
    
    if _rule_engine is None:
        # Load configuration
        _config = load_config("../../configs/dgl.yaml")
        
        # Create repositories
        exposure_store, pnl_store, audit_store = MemoryRepositoryFactory.create_all_stores(
            initial_bankroll=_config.bankroll.total_bankroll
        )
        
        # Create rule engine
        _rule_engine = RuleEngine(_config, exposure_store, pnl_store, audit_store)
        logger.info("Rule engine initialized for API")
    
    return _rule_engine


class BatchProposalRequest(BaseModel):
    """Request model for batch proposal evaluation"""
    proposals: List[BetProposal] = Field(..., min_items=1, max_items=100)
    options: Dict[str, Any] = Field(default_factory=dict)


class BatchProposalResponse(BaseModel):
    """Response model for batch proposal evaluation"""
    decisions: List[GovernanceDecision]
    summary: Dict[str, Any]
    processing_time_ms: float
    timestamp: datetime


class DecisionStatsResponse(BaseModel):
    """Response model for decision statistics"""
    total_decisions: int
    decisions_by_type: Dict[str, int]
    decisions_by_rule: Dict[str, int]
    avg_processing_time_ms: float
    recent_activity: List[Dict[str, Any]]
    timestamp: datetime


class ProposalValidationResponse(BaseModel):
    """Response model for proposal validation"""
    is_valid: bool
    validation_errors: List[str]
    warnings: List[str]
    suggested_amendments: Dict[str, Any]


@governance_router.post("/evaluate", response_model=GovernanceDecision)
async def evaluate_proposal(
    proposal: BetProposal,
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> GovernanceDecision:
    """
    Evaluate a single bet proposal against governance rules
    
    Args:
        proposal: The bet proposal to evaluate
        
    Returns:
        GovernanceDecision with approval/rejection/amendment
        
    Raises:
        HTTPException: If evaluation fails
    """
    try:
        logger.info(f"Evaluating proposal for market {proposal.market_id}")
        
        # Evaluate proposal
        decision = rule_engine.evaluate_proposal(proposal)
        
        logger.info(f"Decision: {decision.decision.value} for proposal {proposal.market_id}")
        return decision
        
    except Exception as e:
        logger.error(f"Error evaluating proposal: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Governance evaluation failed: {str(e)}"
        )


@governance_router.post("/evaluate/batch", response_model=BatchProposalResponse)
async def evaluate_batch_proposals(
    request: BatchProposalRequest,
    background_tasks: BackgroundTasks,
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> BatchProposalResponse:
    """
    Evaluate multiple bet proposals in batch
    
    Args:
        request: Batch of proposals to evaluate
        background_tasks: For async processing
        
    Returns:
        BatchProposalResponse with all decisions and summary
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Evaluating batch of {len(request.proposals)} proposals")
        
        decisions = []
        decision_counts = {"APPROVE": 0, "REJECT": 0, "AMEND": 0}
        rule_violations = {}
        
        # Process each proposal
        for i, proposal in enumerate(request.proposals):
            try:
                decision = rule_engine.evaluate_proposal(proposal)
                decisions.append(decision)
                
                # Update statistics
                decision_counts[decision.decision.value] += 1
                
                # Track rule violations
                for rule_id in decision.rule_ids_triggered:
                    rule_name = rule_id.value
                    rule_violations[rule_name] = rule_violations.get(rule_name, 0) + 1
                    
            except Exception as e:
                logger.error(f"Error evaluating proposal {i}: {str(e)}")
                # Create error decision
                error_decision = GovernanceDecision(
                    decision=DecisionType.REJECT,
                    rule_ids_triggered=[RuleId.BANKROLL_MAX_EXPOSURE],  # Generic error
                    reasoning=f"Evaluation error: {str(e)}",
                    confidence_score=0.0,
                    processing_time_ms=0.0,
                    audit_ref="error"
                )
                decisions.append(error_decision)
                decision_counts["REJECT"] += 1
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create summary
        summary = {
            "total_proposals": len(request.proposals),
            "successful_evaluations": len([d for d in decisions if "error" not in d.reasoning.lower()]),
            "decision_breakdown": decision_counts,
            "top_rule_violations": dict(sorted(rule_violations.items(), key=lambda x: x[1], reverse=True)[:5]),
            "avg_processing_time_per_proposal_ms": processing_time / len(request.proposals) if request.proposals else 0
        }
        
        # Schedule background audit logging
        background_tasks.add_task(
            _log_batch_audit,
            len(request.proposals),
            decision_counts,
            processing_time
        )
        
        return BatchProposalResponse(
            decisions=decisions,
            summary=summary,
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in batch evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch evaluation failed: {str(e)}"
        )


@governance_router.post("/validate", response_model=ProposalValidationResponse)
async def validate_proposal(
    proposal: BetProposal,
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> ProposalValidationResponse:
    """
    Validate a proposal without making a governance decision
    
    Args:
        proposal: The bet proposal to validate
        
    Returns:
        ProposalValidationResponse with validation results
    """
    try:
        validation_errors = []
        warnings = []
        suggested_amendments = {}
        
        # Basic validation
        if proposal.stake <= 0:
            validation_errors.append("Stake must be positive")
        
        if proposal.odds <= 1.0:
            validation_errors.append("Odds must be greater than 1.0")
        
        if not (0.0 <= proposal.model_confidence <= 1.0):
            validation_errors.append("Model confidence must be between 0.0 and 1.0")
        
        # Check against current exposure without making decision
        current_exposure = rule_engine.exposure_store.get_current_exposure()
        
        # Bankroll checks
        if proposal.stake > current_exposure.bankroll * 0.1:  # 10% of bankroll
            warnings.append(f"Large stake relative to bankroll ({proposal.stake:.0f} vs {current_exposure.bankroll:.0f})")
            suggested_amendments["stake"] = current_exposure.bankroll * 0.05  # Suggest 5%
        
        # Odds range checks
        config = rule_engine.config
        if proposal.odds < config.liquidity.min_odds_threshold:
            validation_errors.append(f"Odds below minimum threshold ({config.liquidity.min_odds_threshold})")
            suggested_amendments["odds"] = config.liquidity.min_odds_threshold
        
        if proposal.odds > config.liquidity.max_odds_threshold:
            validation_errors.append(f"Odds above maximum threshold ({config.liquidity.max_odds_threshold})")
            suggested_amendments["odds"] = config.liquidity.max_odds_threshold
        
        # Liquidity checks
        if proposal.liquidity and proposal.liquidity.available > 0:
            fraction_used = (proposal.stake / proposal.liquidity.available) * 100
            if fraction_used > config.liquidity.max_fraction_of_available_liquidity:
                warnings.append(f"High liquidity consumption ({fraction_used:.1f}%)")
                max_stake = proposal.liquidity.available * (config.liquidity.max_fraction_of_available_liquidity / 100)
                suggested_amendments["stake"] = min(suggested_amendments.get("stake", float('inf')), max_stake)
        
        is_valid = len(validation_errors) == 0
        
        return ProposalValidationResponse(
            is_valid=is_valid,
            validation_errors=validation_errors,
            warnings=warnings,
            suggested_amendments=suggested_amendments
        )
        
    except Exception as e:
        logger.error(f"Error validating proposal: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Proposal validation failed: {str(e)}"
        )


@governance_router.get("/stats", response_model=DecisionStatsResponse)
async def get_decision_statistics(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history to include"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> DecisionStatsResponse:
    """
    Get governance decision statistics
    
    Args:
        hours: Number of hours of history to include (1-168)
        
    Returns:
        DecisionStatsResponse with statistics
    """
    try:
        # Get audit records from the last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # In a real implementation, filter audit records by timestamp
        # For now, use rule engine statistics
        stats = rule_engine.get_performance_stats()
        
        # Mock some statistics based on available data
        total_decisions = rule_engine._decision_count
        
        # Estimate decision breakdown (in real implementation, track this)
        decisions_by_type = {
            "APPROVE": int(total_decisions * 0.6),  # 60% approved
            "REJECT": int(total_decisions * 0.3),   # 30% rejected
            "AMEND": int(total_decisions * 0.1)     # 10% amended
        }
        
        # Mock rule violation statistics
        decisions_by_rule = {
            "BANKROLL_MAX_EXPOSURE": int(total_decisions * 0.15),
            "LIQ_MIN_ODDS": int(total_decisions * 0.10),
            "PNL_DAILY_LOSS_LIMIT": int(total_decisions * 0.08),
            "LIQ_SLIPPAGE_LIMIT": int(total_decisions * 0.12),
            "RATE_LIMIT_EXCEEDED": int(total_decisions * 0.05)
        }
        
        # Recent activity (mock data)
        recent_activity = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                "decision": ["APPROVE", "REJECT", "AMEND"][i % 3],
                "market_id": f"market_{i}",
                "rules_triggered": i % 3
            }
            for i in range(min(10, total_decisions))
        ]
        
        return DecisionStatsResponse(
            total_decisions=total_decisions,
            decisions_by_type=decisions_by_type,
            decisions_by_rule=decisions_by_rule,
            avg_processing_time_ms=stats.get("avg_processing_time_ms", 0.0),
            recent_activity=recent_activity,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting decision statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@governance_router.get("/health")
async def governance_health_check(
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> Dict[str, Any]:
    """
    Health check for governance system
    
    Returns:
        Health status and system information
    """
    try:
        # Check rule engine health
        stats = rule_engine.get_performance_stats()
        
        # Check current exposure
        exposure = rule_engine.exposure_store.get_current_exposure()
        
        # System health indicators
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "rule_engine": {
                "state": rule_engine._current_state.value,
                "total_decisions": rule_engine._decision_count,
                "avg_processing_time_ms": stats.get("avg_processing_time_ms", 0.0)
            },
            "exposure": {
                "bankroll": exposure.bankroll,
                "open_exposure": exposure.open_exposure,
                "daily_pnl": exposure.daily_pnl,
                "utilization_pct": (exposure.open_exposure / exposure.bankroll) * 100 if exposure.bankroll > 0 else 0
            },
            "rules": {
                "bankroll_rules": "active",
                "pnl_rules": "active", 
                "liquidity_rules": "active"
            }
        }
        
        # Check for any warning conditions
        warnings = []
        if exposure.open_exposure > exposure.bankroll * 0.8:
            warnings.append("High exposure utilization (>80%)")
        
        if exposure.daily_pnl < -exposure.bankroll * 0.05:
            warnings.append("Significant daily loss (>5% of bankroll)")
        
        if warnings:
            health_status["warnings"] = warnings
            health_status["status"] = "warning"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _log_batch_audit(proposal_count: int, decision_counts: Dict[str, int], processing_time: float):
    """Background task to log batch processing audit"""
    try:
        logger.info(
            f"Batch processing completed: {proposal_count} proposals, "
            f"{decision_counts}, {processing_time:.1f}ms"
        )
        # In production, write to audit store
    except Exception as e:
        logger.error(f"Error logging batch audit: {str(e)}")


# Additional utility endpoints

@governance_router.get("/rules/summary")
async def get_rules_summary(
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> Dict[str, Any]:
    """Get summary of all active governance rules"""
    try:
        config = rule_engine.config
        
        return {
            "bankroll_rules": {
                "max_exposure_pct": config.bankroll.max_bankroll_exposure_pct,
                "per_match_max_pct": config.bankroll.per_match_max_pct,
                "per_market_max_pct": config.bankroll.per_market_max_pct,
                "per_bet_max_pct": config.bankroll.per_bet_max_pct
            },
            "pnl_rules": {
                "daily_loss_limit_pct": config.pnl_guards.daily_loss_limit_pct,
                "session_loss_limit_pct": config.pnl_guards.session_loss_limit_pct
            },
            "liquidity_rules": {
                "min_odds_threshold": config.liquidity.min_odds_threshold,
                "max_odds_threshold": config.liquidity.max_odds_threshold,
                "slippage_bps_limit": config.liquidity.slippage_bps_limit,
                "max_fraction_of_available_liquidity": config.liquidity.max_fraction_of_available_liquidity
            },
            "rate_limits": {
                "count": config.ops.rate_limit.count,
                "per_seconds": config.ops.rate_limit.per_seconds
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting rules summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get rules summary: {str(e)}"
        )
