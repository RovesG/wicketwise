# Purpose: DGL configuration management using pydantic-settings
# Author: WicketWise AI, Last Modified: 2024

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml
from pathlib import Path


class BankrollConfig(BaseModel):
    """Bankroll and exposure limit configuration"""
    total_bankroll: float = Field(default=100000.0, gt=0.0, description="Total available bankroll")
    max_bankroll_exposure_pct: float = Field(default=5.0, ge=0.1, le=50.0)
    per_match_max_pct: float = Field(default=2.0, ge=0.1, le=20.0)
    per_market_max_pct: float = Field(default=1.0, ge=0.1, le=10.0)
    per_bet_max_pct: float = Field(default=0.5, ge=0.01, le=5.0)


class PnLGuardsConfig(BaseModel):
    """P&L protection configuration"""
    daily_loss_limit_pct: float = Field(default=3.0, ge=0.1, le=20.0)
    session_loss_limit_pct: float = Field(default=2.0, ge=0.1, le=15.0)


class LiquidityConfig(BaseModel):
    """Liquidity and execution constraints"""
    max_fraction_of_available_liquidity: float = Field(default=10.0, ge=1.0, le=50.0)
    min_odds_threshold: float = Field(default=1.25, ge=1.01, le=2.0)
    max_odds_threshold: float = Field(default=10.0, ge=2.0, le=1000.0)
    slippage_bps_limit: int = Field(default=50, ge=1, le=1000)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    count: int = Field(default=5, ge=1, le=1000)
    per_seconds: int = Field(default=120, ge=1, le=3600)


class OpsConfig(BaseModel):
    """Operational controls configuration"""
    require_dual_approval_threshold_gbp: float = Field(default=2000.0, ge=100.0)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    kill_switch_enabled: bool = Field(default=False)
    shadow_mode_log_only: bool = Field(default=True)


class ConcentrationConfig(BaseModel):
    """Concentration and correlation limits"""
    max_concurrent_markets_per_match: int = Field(default=3, ge=1, le=20)
    max_correlation_group_exposure_pct: float = Field(default=1.5, ge=0.1, le=10.0)


class ComplianceConfig(BaseModel):
    """Compliance and jurisdiction configuration"""
    jurisdiction: str = Field(default="UK")
    allowed_currencies: List[str] = Field(default=["GBP", "USD", "EUR"])
    blocked_markets: List[str] = Field(default_factory=list)
    require_mfa_for_state_changes: bool = Field(default=True)


class AuditConfig(BaseModel):
    """Audit and logging configuration"""
    retention_days: int = Field(default=2555, ge=1)  # 7 years
    hash_algorithm: str = Field(default="sha256")
    enable_hash_chaining: bool = Field(default=True)
    log_level: str = Field(default="INFO")


class ServiceConfig(BaseModel):
    """Service configuration"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    log_level: str = Field(default="INFO")
    enable_metrics: bool = Field(default=True)


class DGLConfig(BaseSettings):
    """Main DGL configuration"""
    
    # Governance mode
    mode: str = Field(default="SHADOW", pattern="^(READY|SHADOW|LIVE|KILLED)$")
    
    # Configuration sections
    bankroll: BankrollConfig = Field(default_factory=BankrollConfig)
    pnl_guards: PnLGuardsConfig = Field(default_factory=PnLGuardsConfig)
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
    ops: OpsConfig = Field(default_factory=OpsConfig)
    concentration: ConcentrationConfig = Field(default_factory=ConcentrationConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    
    # Runtime state
    config_version: str = Field(default="1.0.0")
    last_updated: Optional[str] = Field(default=None)
    
    class Config:
        env_prefix = "DGL_"
        case_sensitive = False

    @classmethod
    def load_from_yaml(cls, config_path: str) -> "DGLConfig":
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        return cls(**yaml_data)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return self.model_dump()
    
    def validate_constraints(self) -> List[str]:
        """Validate configuration constraints and return any violations"""
        violations = []
        
        # Bankroll constraint validations
        if self.bankroll.per_match_max_pct > self.bankroll.max_bankroll_exposure_pct:
            violations.append("per_match_max_pct cannot exceed max_bankroll_exposure_pct")
        
        if self.bankroll.per_market_max_pct > self.bankroll.per_match_max_pct:
            violations.append("per_market_max_pct cannot exceed per_match_max_pct")
        
        if self.bankroll.per_bet_max_pct > self.bankroll.per_market_max_pct:
            violations.append("per_bet_max_pct cannot exceed per_market_max_pct")
        
        # P&L constraint validations
        if self.pnl_guards.session_loss_limit_pct > self.pnl_guards.daily_loss_limit_pct:
            violations.append("session_loss_limit_pct cannot exceed daily_loss_limit_pct")
        
        # Odds constraint validations
        if self.liquidity.min_odds_threshold >= self.liquidity.max_odds_threshold:
            violations.append("min_odds_threshold must be less than max_odds_threshold")
        
        return violations


# Global configuration instance
_config: Optional[DGLConfig] = None


def get_config() -> DGLConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call load_config() first.")
    return _config


def load_config(config_path: str = "configs/dgl.yaml") -> DGLConfig:
    """Load and initialize the global configuration"""
    global _config
    _config = DGLConfig.load_from_yaml(config_path)
    
    # Validate constraints
    violations = _config.validate_constraints()
    if violations:
        raise ValueError(f"Configuration validation failed: {'; '.join(violations)}")
    
    return _config


def reload_config(config_path: str = "configs/dgl.yaml") -> DGLConfig:
    """Reload configuration from file"""
    global _config
    _config = None
    return load_config(config_path)
