# Purpose: Exposure monitoring and reporting API endpoints
# Author: WicketWise AI, Last Modified: 2024

"""
Exposure API Endpoints

Provides REST endpoints for:
- Current exposure monitoring
- Historical exposure tracking
- Risk metrics and alerts
- Position management
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import ExposureSnapshot, BetSide
from engine import RuleEngine
from api.governance import get_rule_engine


logger = logging.getLogger(__name__)

# Router for exposure endpoints
exposure_router = APIRouter(prefix="/exposure", tags=["exposure"])


class ExposureBreakdown(BaseModel):
    """Detailed exposure breakdown"""
    total_exposure: float
    by_market: Dict[str, float]
    by_match: Dict[str, float]
    by_side: Dict[str, float]
    largest_positions: List[Dict[str, Any]]
    risk_metrics: Dict[str, float]


class ExposureAlert(BaseModel):
    """Exposure alert model"""
    alert_id: str
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    acknowledged: bool = False


class ExposureHistoryResponse(BaseModel):
    """Historical exposure data"""
    snapshots: List[ExposureSnapshot]
    summary: Dict[str, Any]
    period_start: datetime
    period_end: datetime


class RiskMetricsResponse(BaseModel):
    """Risk metrics response"""
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    exposure_utilization: float
    concentration_risk: float
    liquidity_risk: float
    timestamp: datetime


@exposure_router.get("/current", response_model=ExposureSnapshot)
async def get_current_exposure(
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> ExposureSnapshot:
    """
    Get current exposure snapshot
    
    Returns:
        Current exposure including bankroll, open positions, and P&L
    """
    try:
        exposure = rule_engine.exposure_store.get_current_exposure()
        logger.info(f"Retrieved current exposure: {exposure.open_exposure:.2f}")
        return exposure
        
    except Exception as e:
        logger.error(f"Error getting current exposure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current exposure: {str(e)}"
        )


@exposure_router.get("/breakdown", response_model=ExposureBreakdown)
async def get_exposure_breakdown(
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> ExposureBreakdown:
    """
    Get detailed exposure breakdown by market, match, and side
    
    Returns:
        Comprehensive exposure analysis
    """
    try:
        exposure = rule_engine.exposure_store.get_current_exposure()
        
        # Mock detailed breakdown (in production, calculate from actual positions)
        by_market = {
            "match_winner": exposure.open_exposure * 0.4,
            "total_runs": exposure.open_exposure * 0.3,
            "top_batsman": exposure.open_exposure * 0.2,
            "method_of_dismissal": exposure.open_exposure * 0.1
        }
        
        by_match = {
            "match_001": exposure.open_exposure * 0.6,
            "match_002": exposure.open_exposure * 0.4
        }
        
        by_side = {
            "BACK": exposure.open_exposure * 0.7,
            "LAY": exposure.open_exposure * 0.3
        }
        
        # Largest positions
        largest_positions = [
            {
                "market_id": "match_001_winner",
                "selection": "Team A",
                "side": "BACK",
                "exposure": exposure.open_exposure * 0.25,
                "odds": 2.5,
                "stake": (exposure.open_exposure * 0.25) / 1.5
            },
            {
                "market_id": "match_001_total_runs", 
                "selection": "Over 300.5",
                "side": "BACK",
                "exposure": exposure.open_exposure * 0.20,
                "odds": 1.9,
                "stake": (exposure.open_exposure * 0.20) / 0.9
            }
        ]
        
        # Risk metrics
        bankroll_utilization = (exposure.open_exposure / exposure.bankroll) * 100 if exposure.bankroll > 0 else 0
        risk_metrics = {
            "bankroll_utilization_pct": bankroll_utilization,
            "daily_pnl_pct": (exposure.daily_pnl / exposure.bankroll) * 100 if exposure.bankroll > 0 else 0,
            "session_pnl_pct": (exposure.session_pnl / exposure.bankroll) * 100 if exposure.bankroll > 0 else 0,
            "largest_position_pct": 25.0,  # Largest single position as % of bankroll
            "market_concentration": 40.0,  # Largest market as % of total exposure
            "volatility_estimate": 15.0    # Estimated daily volatility %
        }
        
        return ExposureBreakdown(
            total_exposure=exposure.open_exposure,
            by_market=by_market,
            by_match=by_match,
            by_side=by_side,
            largest_positions=largest_positions,
            risk_metrics=risk_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting exposure breakdown: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get exposure breakdown: {str(e)}"
        )


@exposure_router.get("/history", response_model=ExposureHistoryResponse)
async def get_exposure_history(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history"),
    interval_minutes: int = Query(default=60, ge=5, le=1440, description="Snapshot interval"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> ExposureHistoryResponse:
    """
    Get historical exposure data
    
    Args:
        hours: Number of hours of history (1-168)
        interval_minutes: Interval between snapshots (5-1440)
        
    Returns:
        Historical exposure snapshots and summary
    """
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Mock historical data (in production, retrieve from time series store)
        current_exposure = rule_engine.exposure_store.get_current_exposure()
        
        # Generate mock snapshots
        snapshots = []
        num_snapshots = min(100, (hours * 60) // interval_minutes)  # Limit to 100 snapshots
        
        for i in range(num_snapshots):
            snapshot_time = start_time + timedelta(minutes=i * interval_minutes)
            
            # Simulate exposure changes over time
            time_factor = i / num_snapshots
            exposure_variation = 1.0 + (time_factor - 0.5) * 0.2  # ±10% variation
            pnl_variation = (time_factor - 0.5) * current_exposure.bankroll * 0.02  # ±2% PnL swing
            
            snapshot = ExposureSnapshot(
                bankroll=current_exposure.bankroll,
                open_exposure=current_exposure.open_exposure * exposure_variation,
                daily_pnl=pnl_variation,
                session_pnl=pnl_variation * 0.8,
                timestamp=snapshot_time
            )
            snapshots.append(snapshot)
        
        # Calculate summary statistics
        if snapshots:
            exposures = [s.open_exposure for s in snapshots]
            pnls = [s.daily_pnl for s in snapshots]
            
            summary = {
                "avg_exposure": sum(exposures) / len(exposures),
                "max_exposure": max(exposures),
                "min_exposure": min(exposures),
                "exposure_volatility": _calculate_volatility(exposures),
                "total_pnl_change": pnls[-1] - pnls[0] if len(pnls) >= 2 else 0,
                "max_pnl": max(pnls),
                "min_pnl": min(pnls),
                "num_snapshots": len(snapshots)
            }
        else:
            summary = {}
        
        return ExposureHistoryResponse(
            snapshots=snapshots,
            summary=summary,
            period_start=start_time,
            period_end=end_time
        )
        
    except Exception as e:
        logger.error(f"Error getting exposure history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get exposure history: {str(e)}"
        )


@exposure_router.get("/alerts", response_model=List[ExposureAlert])
async def get_exposure_alerts(
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    acknowledged: Optional[bool] = Query(default=None, description="Filter by acknowledgment"),
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> List[ExposureAlert]:
    """
    Get current exposure alerts and warnings
    
    Args:
        severity: Filter by alert severity (INFO, WARNING, CRITICAL)
        acknowledged: Filter by acknowledgment status
        
    Returns:
        List of active exposure alerts
    """
    try:
        exposure = rule_engine.exposure_store.get_current_exposure()
        config = rule_engine.config
        
        alerts = []
        
        # Check bankroll utilization
        utilization = (exposure.open_exposure / exposure.bankroll) * 100 if exposure.bankroll > 0 else 0
        
        if utilization > 80:
            alerts.append(ExposureAlert(
                alert_id="EXP_001",
                severity="CRITICAL",
                message=f"Very high bankroll utilization: {utilization:.1f}%",
                threshold=80.0,
                current_value=utilization,
                timestamp=datetime.now()
            ))
        elif utilization > 60:
            alerts.append(ExposureAlert(
                alert_id="EXP_002", 
                severity="WARNING",
                message=f"High bankroll utilization: {utilization:.1f}%",
                threshold=60.0,
                current_value=utilization,
                timestamp=datetime.now()
            ))
        
        # Check daily P&L
        daily_pnl_pct = (exposure.daily_pnl / exposure.bankroll) * 100 if exposure.bankroll > 0 else 0
        
        if daily_pnl_pct < -config.pnl_guards.daily_loss_limit_pct * 0.8:  # 80% of limit
            alerts.append(ExposureAlert(
                alert_id="PNL_001",
                severity="WARNING",
                message=f"Approaching daily loss limit: {daily_pnl_pct:.1f}%",
                threshold=-config.pnl_guards.daily_loss_limit_pct * 0.8,
                current_value=daily_pnl_pct,
                timestamp=datetime.now()
            ))
        
        # Check session P&L
        session_pnl_pct = (exposure.session_pnl / exposure.bankroll) * 100 if exposure.bankroll > 0 else 0
        
        if session_pnl_pct < -config.pnl_guards.session_loss_limit_pct * 0.8:  # 80% of limit
            alerts.append(ExposureAlert(
                alert_id="PNL_002",
                severity="WARNING", 
                message=f"Approaching session loss limit: {session_pnl_pct:.1f}%",
                threshold=-config.pnl_guards.session_loss_limit_pct * 0.8,
                current_value=session_pnl_pct,
                timestamp=datetime.now()
            ))
        
        # Add informational alerts for positive performance
        if daily_pnl_pct > 5.0:
            alerts.append(ExposureAlert(
                alert_id="PNL_003",
                severity="INFO",
                message=f"Strong daily performance: +{daily_pnl_pct:.1f}%",
                threshold=5.0,
                current_value=daily_pnl_pct,
                timestamp=datetime.now()
            ))
        
        # Filter alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity.upper()]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting exposure alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get exposure alerts: {str(e)}"
        )


@exposure_router.get("/risk-metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics(
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> RiskMetricsResponse:
    """
    Get comprehensive risk metrics
    
    Returns:
        Risk metrics including VaR, expected shortfall, and other measures
    """
    try:
        exposure = rule_engine.exposure_store.get_current_exposure()
        
        # Mock risk calculations (in production, use historical data and proper models)
        bankroll = exposure.bankroll
        open_exposure = exposure.open_exposure
        
        # Value at Risk (95% confidence)
        var_95 = bankroll * 0.05  # 5% of bankroll at 95% confidence
        
        # Expected Shortfall (average loss beyond VaR)
        expected_shortfall = var_95 * 1.3  # Typically 1.2-1.4x VaR
        
        # Maximum Drawdown (mock calculation)
        max_drawdown = bankroll * 0.08  # 8% maximum historical drawdown
        
        # Sharpe Ratio (mock calculation)
        sharpe_ratio = 1.2  # Risk-adjusted return ratio
        
        # Exposure utilization
        exposure_utilization = (open_exposure / bankroll) * 100 if bankroll > 0 else 0
        
        # Concentration risk (largest position as % of bankroll)
        concentration_risk = 25.0  # Mock: 25% in largest position
        
        # Liquidity risk (time to unwind positions)
        liquidity_risk = 15.0  # Mock: 15 minutes average unwind time
        
        return RiskMetricsResponse(
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            exposure_utilization=exposure_utilization,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate risk metrics: {str(e)}"
        )


@exposure_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    rule_engine: RuleEngine = Depends(get_rule_engine)
) -> Dict[str, Any]:
    """
    Acknowledge an exposure alert
    
    Args:
        alert_id: ID of the alert to acknowledge
        
    Returns:
        Acknowledgment confirmation
    """
    try:
        # In production, update alert status in database
        logger.info(f"Alert {alert_id} acknowledged")
        
        return {
            "alert_id": alert_id,
            "acknowledged": True,
            "acknowledged_at": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to acknowledge alert: {str(e)}"
        )


def _calculate_volatility(values: List[float]) -> float:
    """Calculate volatility (standard deviation) of a series"""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
