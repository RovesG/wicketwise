# Purpose: Pydantic schemas for DGL data structures
# Author: WicketWise AI, Last Modified: 2024

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from decimal import Decimal
from enum import Enum
import uuid


class BetSide(str, Enum):
    """Betting side enumeration"""
    BACK = "BACK"
    LAY = "LAY"


class DecisionType(str, Enum):
    """Governance decision types"""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    AMEND = "AMEND"


class GovernanceState(str, Enum):
    """DGL governance states"""
    READY = "READY"
    SHADOW = "SHADOW"
    LIVE = "LIVE"
    KILLED = "KILLED"


class RuleId(str, Enum):
    """Rule identifier enumeration"""
    # Bankroll and exposure rules
    BANKROLL_MAX_EXPOSURE = "BANKROLL.MAX_EXPOSURE"
    EXPO_PER_MATCH_MAX = "EXPO.PER_MATCH_MAX"
    EXPO_PER_MARKET_MAX = "EXPO.PER_MARKET_MAX"
    EXPO_PER_BET_MAX = "EXPO.PER_BET_MAX"
    
    # P&L guard rules
    PNL_DAILY_LOSS_LIMIT = "PNL.DAILY_LOSS_LIMIT"
    PNL_SESSION_LOSS_LIMIT = "PNL.SESSION_LOSS_LIMIT"
    
    # Liquidity rules
    LIQ_FRACTION_LIMIT = "LIQ.FRACTION_LIMIT"
    LIQ_MIN_ODDS = "LIQ.MIN_ODDS"
    LIQ_MAX_ODDS = "LIQ.MAX_ODDS"
    LIQ_SLIPPAGE_LIMIT = "LIQ.SLIPPAGE_LIMIT"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE.LIMIT_EXCEEDED"
    
    # Concentration rules
    CONC_MAX_MARKETS_PER_MATCH = "CONC.MAX_MARKETS_PER_MATCH"
    CONC_CORRELATION_GROUP = "CONC.CORRELATION_GROUP"
    
    # Compliance rules
    COMP_JURISDICTION = "COMP.JURISDICTION"
    COMP_CURRENCY = "COMP.CURRENCY"
    COMP_BLOCKED_MARKET = "COMP.BLOCKED_MARKET"
    COMP_DUAL_APPROVAL = "COMP.DUAL_APPROVAL"
    
    # Operational rules
    OPS_KILL_SWITCH = "OPS.KILL_SWITCH"
    OPS_SHADOW_MODE = "OPS.SHADOW_MODE"


class MarketDepth(BaseModel):
    """Market depth information"""
    odds: float = Field(..., gt=1.0, description="Odds level")
    size: float = Field(..., gt=0, description="Available size at this odds level")


class LiquidityInfo(BaseModel):
    """Liquidity information for a market"""
    available: float = Field(..., ge=0, description="Total available liquidity")
    market_depth: List[MarketDepth] = Field(default_factory=list, description="Market depth ladder")


class BetProposal(BaseModel):
    """Bet proposal input to DGL"""
    
    # Identifiers
    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique proposal ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Proposal timestamp")
    
    # Market information
    market_id: str = Field(..., description="Market identifier (e.g., betfair:1.234567)")
    match_id: str = Field(..., description="Match identifier")
    
    # Bet details
    side: BetSide = Field(..., description="BACK or LAY")
    selection: str = Field(..., description="Selection name")
    odds: float = Field(..., gt=1.0, description="Requested odds")
    stake: float = Field(..., gt=0, description="Requested stake amount")
    currency: str = Field(default="GBP", description="Currency code")
    
    # Model information
    model_confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    fair_odds: Optional[float] = Field(default=None, gt=1.0, description="Fair odds estimate")
    expected_edge_pct: float = Field(..., description="Expected edge percentage")
    
    # Context
    features: Dict[str, Any] = Field(default_factory=dict, description="Model features")
    liquidity: Optional[LiquidityInfo] = Field(default=None, description="Market liquidity info")
    correlation_group: Optional[str] = Field(default=None, description="Correlation group identifier")
    explain: Optional[str] = Field(default=None, description="Human-readable explanation")
    
    @validator('currency')
    def validate_currency(cls, v):
        """Validate currency code format"""
        if len(v) != 3 or not v.isupper():
            raise ValueError("Currency must be a 3-letter uppercase code")
        return v
    
    @validator('odds', 'fair_odds')
    def validate_odds(cls, v):
        """Validate odds are reasonable"""
        if v < 1.01 or v > 1000.0:
            raise ValueError("Odds must be between 1.01 and 1000.0")
        return v


class BetAmendment(BaseModel):
    """Proposed amendments to a bet"""
    stake: Optional[float] = Field(default=None, gt=0, description="Amended stake")
    odds: Optional[float] = Field(default=None, gt=1.0, description="Amended odds")
    
    @validator('odds')
    def validate_odds(cls, v):
        """Validate amended odds"""
        if v is not None and (v < 1.01 or v > 1000.0):
            raise ValueError("Amended odds must be between 1.01 and 1000.0")
        return v


class GovernanceDecision(BaseModel):
    """DGL governance decision output"""
    
    # Core decision
    proposal_id: str = Field(..., description="Original proposal ID")
    decision: DecisionType = Field(..., description="APPROVE, REJECT, or AMEND")
    
    # Amendment details (if applicable)
    amendment: Optional[BetAmendment] = Field(default=None, description="Proposed amendments")
    
    # Rule information
    rule_ids_triggered: List[RuleId] = Field(default_factory=list, description="Rules that triggered")
    human_message: str = Field(..., description="Human-readable decision explanation")
    
    # State and timing
    state: GovernanceState = Field(..., description="Current DGL state")
    ttl_seconds: int = Field(default=5, ge=1, le=300, description="Decision validity TTL")
    
    # Audit and security
    signature: Optional[str] = Field(default=None, description="Decision signature (ed25519_hex)")
    audit_ref: str = Field(..., description="Audit record reference")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Decision timestamp")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")


class ExposureSnapshot(BaseModel):
    """Current exposure snapshot"""
    bankroll: float = Field(..., ge=0, description="Current bankroll")
    open_exposure: float = Field(..., ge=0, description="Total open exposure")
    daily_pnl: float = Field(..., description="Daily P&L")
    session_pnl: float = Field(..., description="Session P&L")
    
    # Exposure breakdowns
    per_match_exposure: Dict[str, float] = Field(default_factory=dict, description="Exposure per match")
    per_market_exposure: Dict[str, float] = Field(default_factory=dict, description="Exposure per market")
    per_correlation_group: Dict[str, float] = Field(default_factory=dict, description="Exposure per correlation group")
    
    # Timestamps
    snapshot_time: datetime = Field(default_factory=datetime.utcnow, description="Snapshot timestamp")
    session_start: datetime = Field(default_factory=datetime.utcnow, description="Session start time")


class AuditRecord(BaseModel):
    """Immutable audit log record with hash chaining"""
    
    # Core audit information
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique audit ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Audit timestamp")
    entity: str = Field(default="DGL", description="Entity that created the record")
    
    # Decision context
    proposal_id: str = Field(..., description="Related proposal ID")
    decision: DecisionType = Field(..., description="Decision made")
    rule_ids: List[RuleId] = Field(default_factory=list, description="Rules that were evaluated")
    
    # State snapshot
    snapshot: ExposureSnapshot = Field(..., description="Exposure snapshot at decision time")
    
    # Hash chaining for integrity
    hash_prev: Optional[str] = Field(default=None, description="Hash of previous audit record")
    hash_curr: Optional[str] = Field(default=None, description="Hash of current record")
    
    # Additional metadata
    user_id: Optional[str] = Field(default=None, description="User ID if applicable")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., ge=0, description="Service uptime in seconds")
    
    # Component health
    components: Dict[str, str] = Field(default_factory=dict, description="Component health status")
    
    # Performance metrics
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")


class VersionResponse(BaseModel):
    """Version information response"""
    service: str = Field(default="DGL", description="Service name")
    version: str = Field(..., description="Service version")
    build_time: Optional[str] = Field(default=None, description="Build timestamp")
    git_commit: Optional[str] = Field(default=None, description="Git commit hash")
    config_version: str = Field(..., description="Configuration version")


class RuleStatus(BaseModel):
    """Individual rule status"""
    rule_id: RuleId = Field(..., description="Rule identifier")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    threshold: Optional[float] = Field(default=None, description="Rule threshold value")
    current_value: Optional[float] = Field(default=None, description="Current measured value")
    last_triggered: Optional[datetime] = Field(default=None, description="Last time rule was triggered")
    trigger_count: int = Field(default=0, ge=0, description="Number of times rule has triggered")


class GovernanceStats(BaseModel):
    """Governance statistics"""
    
    # Decision counts
    total_decisions: int = Field(default=0, ge=0, description="Total decisions made")
    approvals: int = Field(default=0, ge=0, description="Number of approvals")
    rejections: int = Field(default=0, ge=0, description="Number of rejections")
    amendments: int = Field(default=0, ge=0, description="Number of amendments")
    
    # Performance metrics
    avg_processing_time_ms: float = Field(default=0.0, ge=0, description="Average processing time")
    p99_processing_time_ms: float = Field(default=0.0, ge=0, description="99th percentile processing time")
    
    # Rule statistics
    rule_trigger_counts: Dict[RuleId, int] = Field(default_factory=dict, description="Rule trigger counts")
    
    # Time window
    stats_period_start: datetime = Field(default_factory=datetime.utcnow, description="Statistics period start")
    stats_period_end: datetime = Field(default_factory=datetime.utcnow, description="Statistics period end")
