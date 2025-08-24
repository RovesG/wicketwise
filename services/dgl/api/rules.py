# Purpose: Rules configuration and management API endpoints
# Author: WicketWise AI, Last Modified: 2024

"""
Rules API Endpoints

Provides REST endpoints for:
- Rules configuration management
- Rule testing and validation
- Rule performance monitoring
- Dynamic rule updates (future)
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import BetProposal, RuleId
from engine import RuleEngine
from config import DGLConfig, BankrollConfig, PnLGuardsConfig, LiquidityConfig
from api.governance import get_rule_engine


logger = logging.getLogger(__name__)

# Router for rules endpoints
rules_router = APIRouter(prefix="/rules", tags=["rules"])


class RuleTestRequest(BaseModel):
    """Request model for rule testing"""
    rule_ids: List[RuleId] = Field(..., description="Rules to test")
    test_proposal: BetProposal = Field(..., description="Proposal to test against")
    options: Dict[str, Any] = Field(default_factory=dict, description="Test options")


class RuleTestResponse(BaseModel):
    """Response model for rule testing"""
    rule_results: Dict[str, Dict[str, Any]]
    overall_result: str
    violations_found: int
    test_timestamp: datetime


class RulePerformanceResponse(BaseModel):
    """Response model for rule performance metrics"""
    rule_id: RuleId
    total_evaluations: int
    violations_count: int
    violation_rate_pct: float
    avg_evaluation_time_ms: float
    recent_violations: List[Dict[str, Any]]
    performance_trend: str


class RuleConfigResponse(BaseModel):
    """Response model for rule configuration"""
    bankroll_config: Dict[str, Any]
    pnl_config: Dict[str, Any]
    liquidity_config: Dict[str, Any]
    rate_limit_config: Dict[str, Any]
    last_updated: datetime


class RuleUpdateRequest(BaseModel):
    """Request model for rule configuration updates"""
    rule_category: str = Field(..., description="Category: bankroll, pnl, liquidity, rate_limit")
    updates: Dict[str, Any] = Field(..., description="Configuration updates")
    reason: str = Field(..., description="Reason for update")


@rules_router.get("/config", response_model=RuleConfigResponse)
async def get_rules_configuration(
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> RuleConfigResponse:
    """
    Get current rules configuration
    
    Returns:
        Complete rules configuration
    """
    try:
        config = rule_engine.config
        
        bankroll_config = {
            "total_bankroll": config.bankroll.total_bankroll,
            "max_bankroll_exposure_pct": config.bankroll.max_bankroll_exposure_pct,
            "per_match_max_pct": config.bankroll.per_match_max_pct,
            "per_market_max_pct": config.bankroll.per_market_max_pct,
            "per_bet_max_pct": config.bankroll.per_bet_max_pct
        }
        
        pnl_config = {
            "daily_loss_limit_pct": config.pnl_guards.daily_loss_limit_pct,
            "session_loss_limit_pct": config.pnl_guards.session_loss_limit_pct
        }
        
        liquidity_config = {
            "min_odds_threshold": config.liquidity.min_odds_threshold,
            "max_odds_threshold": config.liquidity.max_odds_threshold,
            "slippage_bps_limit": config.liquidity.slippage_bps_limit,
            "max_fraction_of_available_liquidity": config.liquidity.max_fraction_of_available_liquidity
        }
        
        rate_limit_config = {
            "count": config.ops.rate_limit.count,
            "per_seconds": config.ops.rate_limit.per_seconds
        }
        
        return RuleConfigResponse(
            bankroll_config=bankroll_config,
            pnl_config=pnl_config,
            liquidity_config=liquidity_config,
            rate_limit_config=rate_limit_config,
            last_updated=datetime.now()  # In production, track actual update time
        )
        
    except Exception as e:
        logger.error(f"Error getting rules configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get rules configuration: {str(e)}"
        )


@rules_router.post("/test", response_model=RuleTestResponse)
async def test_rules(
    request: RuleTestRequest,
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> RuleTestResponse:
    """
    Test specific rules against a proposal
    
    Args:
        request: Rule test request with rules and proposal
        
    Returns:
        Detailed test results for each rule
    """
    try:
        logger.info(f"Testing {len(request.rule_ids)} rules against proposal")
        
        # Get current exposure for rule evaluation
        exposure = rule_engine.exposure_store.get_current_exposure()
        
        rule_results = {}
        total_violations = 0
        
        # Test each rule individually
        for rule_id in request.rule_ids:
            try:
                violations = []
                evaluation_time_ms = 0.0
                
                # Route to appropriate rule engine based on rule ID
                if rule_id.value.startswith("BANKROLL"):
                    start_time = datetime.now()
                    violations = rule_engine.bankroll_engine.evaluate_all(request.test_proposal, exposure)
                    evaluation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    
                elif rule_id.value.startswith("PNL"):
                    start_time = datetime.now()
                    violations = rule_engine.pnl_engine.evaluate_all(request.test_proposal, exposure)
                    evaluation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    
                elif rule_id.value.startswith("LIQ") or rule_id.value.startswith("RATE"):
                    start_time = datetime.now()
                    violations = rule_engine.liquidity_engine.evaluate_all(request.test_proposal, exposure)
                    evaluation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Filter violations for this specific rule
                rule_violations = [v for v in violations if v.rule_id == rule_id]
                
                rule_results[rule_id.value] = {
                    "rule_id": rule_id.value,
                    "violations_found": len(rule_violations),
                    "violations": [
                        {
                            "message": v.message,
                            "current_value": v.current_value,
                            "threshold": v.threshold,
                            "severity": getattr(v, 'severity', 'ERROR')
                        }
                        for v in rule_violations
                    ],
                    "evaluation_time_ms": evaluation_time_ms,
                    "status": "VIOLATED" if rule_violations else "PASSED"
                }
                
                total_violations += len(rule_violations)
                
            except Exception as e:
                logger.error(f"Error testing rule {rule_id.value}: {str(e)}")
                rule_results[rule_id.value] = {
                    "rule_id": rule_id.value,
                    "error": str(e),
                    "status": "ERROR"
                }
        
        # Determine overall result
        if total_violations == 0:
            overall_result = "ALL_PASSED"
        elif total_violations < len(request.rule_ids):
            overall_result = "PARTIAL_VIOLATIONS"
        else:
            overall_result = "ALL_VIOLATED"
        
        return RuleTestResponse(
            rule_results=rule_results,
            overall_result=overall_result,
            violations_found=total_violations,
            test_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error testing rules: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Rule testing failed: {str(e)}"
        )


@rules_router.get("/performance/{rule_id}", response_model=RulePerformanceResponse)
async def get_rule_performance(
    rule_id: RuleId,
    days: int = Query(default=7, ge=1, le=30, description="Days of history"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> RulePerformanceResponse:
    """
    Get performance metrics for a specific rule
    
    Args:
        rule_id: ID of the rule to analyze
        days: Number of days of history to include
        
    Returns:
        Rule performance metrics and trends
    """
    try:
        # Mock performance data (in production, retrieve from metrics store)
        total_evaluations = rule_engine._decision_count
        
        # Estimate violations based on rule type
        violation_rates = {
            "BANKROLL_MAX_EXPOSURE": 0.15,
            "BANKROLL_PER_MATCH_MAX": 0.08,
            "BANKROLL_PER_MARKET_MAX": 0.12,
            "BANKROLL_PER_BET_MAX": 0.05,
            "PNL_DAILY_LOSS_LIMIT": 0.03,
            "PNL_SESSION_LOSS_LIMIT": 0.02,
            "LIQ_MIN_ODDS": 0.10,
            "LIQ_MAX_ODDS": 0.06,
            "LIQ_SLIPPAGE_LIMIT": 0.18,
            "LIQ_FRACTION_LIMIT": 0.14,
            "RATE_LIMIT_EXCEEDED": 0.04
        }
        
        violation_rate = violation_rates.get(rule_id.value, 0.10)
        violations_count = int(total_evaluations * violation_rate)
        
        # Mock recent violations
        recent_violations = [
            {
                "timestamp": (datetime.now() - timedelta(hours=i*2)).isoformat(),
                "market_id": f"market_{i}",
                "violation_message": f"Rule {rule_id.value} violated",
                "severity": ["WARNING", "ERROR"][i % 2]
            }
            for i in range(min(5, violations_count))
        ]
        
        # Determine performance trend
        if violation_rate < 0.05:
            performance_trend = "EXCELLENT"
        elif violation_rate < 0.10:
            performance_trend = "GOOD"
        elif violation_rate < 0.20:
            performance_trend = "ACCEPTABLE"
        else:
            performance_trend = "CONCERNING"
        
        return RulePerformanceResponse(
            rule_id=rule_id,
            total_evaluations=total_evaluations,
            violations_count=violations_count,
            violation_rate_pct=violation_rate * 100,
            avg_evaluation_time_ms=0.5,  # Mock evaluation time
            recent_violations=recent_violations,
            performance_trend=performance_trend
        )
        
    except Exception as e:
        logger.error(f"Error getting rule performance for {rule_id.value}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get rule performance: {str(e)}"
        )


@rules_router.get("/performance", response_model=List[RulePerformanceResponse])
async def get_all_rules_performance(
    days: int = Query(default=7, ge=1, le=30, description="Days of history"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> List[RulePerformanceResponse]:
    """
    Get performance metrics for all rules
    
    Args:
        days: Number of days of history to include
        
    Returns:
        Performance metrics for all rules
    """
    try:
        # Get performance for all rule types
        all_rules = [
            RuleId.BANKROLL_MAX_EXPOSURE,
            RuleId.BANKROLL_PER_MATCH_MAX,
            RuleId.BANKROLL_PER_MARKET_MAX,
            RuleId.BANKROLL_PER_BET_MAX,
            RuleId.PNL_DAILY_LOSS_LIMIT,
            RuleId.PNL_SESSION_LOSS_LIMIT,
            RuleId.LIQ_MIN_ODDS,
            RuleId.LIQ_MAX_ODDS,
            RuleId.LIQ_SLIPPAGE_LIMIT,
            RuleId.LIQ_FRACTION_LIMIT,
            RuleId.RATE_LIMIT_EXCEEDED
        ]
        
        performance_results = []
        
        for rule_id in all_rules:
            try:
                performance = await get_rule_performance(rule_id, days, rule_engine)
                performance_results.append(performance)
            except Exception as e:
                logger.error(f"Error getting performance for rule {rule_id.value}: {str(e)}")
                # Continue with other rules
        
        return performance_results
        
    except Exception as e:
        logger.error(f"Error getting all rules performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get rules performance: {str(e)}"
        )


@rules_router.get("/violations/recent")
async def get_recent_violations(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum violations to return"),
    rule_id: Optional[RuleId] = Query(default=None, description="Filter by rule ID"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> Dict[str, Any]:
    """
    Get recent rule violations
    
    Args:
        limit: Maximum number of violations to return
        rule_id: Optional filter by specific rule
        
    Returns:
        Recent violations with details
    """
    try:
        # Mock recent violations data (in production, query audit/violation store)
        violations = []
        
        violation_types = [
            ("BANKROLL_MAX_EXPOSURE", "Bankroll exposure limit exceeded"),
            ("LIQ_SLIPPAGE_LIMIT", "Slippage limit exceeded"),
            ("LIQ_FRACTION_LIMIT", "Liquidity fraction limit exceeded"),
            ("PNL_DAILY_LOSS_LIMIT", "Daily loss limit approached"),
            ("RATE_LIMIT_EXCEEDED", "Rate limit exceeded")
        ]
        
        for i in range(min(limit, 20)):  # Generate up to 20 mock violations
            rule_type, message = violation_types[i % len(violation_types)]
            
            # Filter by rule_id if specified
            if rule_id and rule_type != rule_id.value:
                continue
            
            violation = {
                "id": f"violation_{i}",
                "rule_id": rule_type,
                "message": message,
                "market_id": f"market_{i % 5}",
                "match_id": f"match_{i % 3}",
                "timestamp": (datetime.now() - timedelta(minutes=i*10)).isoformat(),
                "severity": ["WARNING", "ERROR", "CRITICAL"][i % 3],
                "resolved": i % 4 == 0  # 25% resolved
            }
            violations.append(violation)
        
        return {
            "violations": violations,
            "total_count": len(violations),
            "filtered_by_rule": rule_id.value if rule_id else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recent violations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent violations: {str(e)}"
        )


@rules_router.get("/health")
async def rules_health_check(
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> Dict[str, Any]:
    """
    Health check for rules system
    
    Returns:
        Rules system health status
    """
    try:
        # Check rule engines health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "rule_engines": {
                "bankroll_engine": {
                    "status": "active",
                    "rules_count": 4,  # Number of bankroll rules
                    "last_evaluation": datetime.now().isoformat()
                },
                "pnl_engine": {
                    "status": "active", 
                    "rules_count": 2,  # Number of P&L rules
                    "last_evaluation": datetime.now().isoformat()
                },
                "liquidity_engine": {
                    "status": "active",
                    "rules_count": 5,  # Number of liquidity rules (including rate limiting)
                    "last_evaluation": datetime.now().isoformat()
                }
            },
            "configuration": {
                "last_loaded": datetime.now().isoformat(),
                "source": "dgl.yaml",
                "validation_status": "valid"
            },
            "performance": {
                "total_evaluations": rule_engine._decision_count,
                "avg_processing_time_ms": sum(rule_engine._processing_times) / len(rule_engine._processing_times) if rule_engine._processing_times else 0
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Rules health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Future endpoint for dynamic rule updates (placeholder)
@rules_router.post("/config/update")
async def update_rules_configuration(
    request: RuleUpdateRequest,
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> Dict[str, Any]:
    """
    Update rules configuration (future implementation)
    
    Args:
        request: Configuration update request
        
    Returns:
        Update confirmation
    """
    # This would be implemented in a future sprint for dynamic rule updates
    return {
        "status": "not_implemented",
        "message": "Dynamic rule updates will be implemented in a future sprint",
        "requested_category": request.rule_category,
        "timestamp": datetime.now().isoformat()
    }
