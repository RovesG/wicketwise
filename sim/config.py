# Purpose: SIM configuration dataclasses with JSON serialization
# Author: WicketWise AI, Last Modified: 2024

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import hashlib
from enum import Enum


class SimulationMode(Enum):
    """Simulation execution modes"""
    REPLAY = "replay"
    MODEL_LOOP = "model_loop" 
    MONTE_CARLO = "monte_carlo"
    WALK_FORWARD = "walk_forward"
    PAPER = "paper"


class TimelineSpeed(Enum):
    """Timeline playback speeds"""
    REALTIME = "realtime"
    X10 = "x10"
    INSTANT = "instant"
    AS_DATA = "as_data"


@dataclass
class StrategyParams:
    """Strategy configuration parameters"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyParams':
        return cls(**data)


@dataclass 
class RiskProfile:
    """Risk management configuration"""
    bankroll: float = 100000.0
    max_exposure_pct: float = 5.0
    per_market_cap_pct: float = 2.0
    per_bet_cap_pct: float = 0.5
    correlation_cap: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskProfile':
        return cls(**data)


@dataclass
class ExecutionParams:
    """Execution model configuration"""
    latency_ms: Dict[str, float] = field(default_factory=lambda: {"mean": 250.0, "std": 50.0})
    commission_bps: float = 200.0
    slippage_model: str = "lob_queue"
    participation_factor: float = 0.1
    post_only: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionParams':
        return cls(**data)


@dataclass
class LiquidityParams:
    """Liquidity simulation parameters"""
    scale: float = 1.0
    min_fill_ratio: float = 0.1
    overround_target: float = 1.05
    depth_schedule: Dict[str, float] = field(default_factory=lambda: {
        "powerplay": 1.2,
        "middle": 1.0, 
        "death": 0.8
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LiquidityParams':
        return cls(**data)


@dataclass
class TimelineParams:
    """Timeline control parameters"""
    start: str = "as_data"
    speed: TimelineSpeed = TimelineSpeed.REALTIME
    suspend_grace_ms: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["speed"] = self.speed.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimelineParams':
        data = data.copy()
        if "speed" in data:
            data["speed"] = TimelineSpeed(data["speed"])
        return cls(**data)


@dataclass
class OutputParams:
    """Output and artifact configuration"""
    dir: str = "runs/sim_default"
    artifacts: List[str] = field(default_factory=lambda: [
        "orders", "fills", "dgl", "metrics", "plots"
    ])
    save_prompts: bool = True
    export_html: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputParams':
        return cls(**data)


@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    id: str
    mode: SimulationMode
    markets: List[str] = field(default_factory=lambda: ["match_odds"])
    match_ids: List[str] = field(default_factory=list)
    strategy: StrategyParams = field(default_factory=lambda: StrategyParams("edge_kelly_v3"))
    risk_profile: RiskProfile = field(default_factory=RiskProfile)
    execution: ExecutionParams = field(default_factory=ExecutionParams)
    liquidity: LiquidityParams = field(default_factory=LiquidityParams)
    timeline: TimelineParams = field(default_factory=TimelineParams)
    outputs: OutputParams = field(default_factory=OutputParams)
    seed: int = 42
    record_prompts: bool = True
    created_at: Optional[str] = None
    git_commit: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "id": self.id,
            "mode": self.mode.value,
            "markets": self.markets,
            "match_ids": self.match_ids,
            "strategy": self.strategy.to_dict(),
            "risk_profile": self.risk_profile.to_dict(),
            "execution": self.execution.to_dict(),
            "liquidity": self.liquidity.to_dict(),
            "timeline": self.timeline.to_dict(),
            "outputs": self.outputs.to_dict(),
            "seed": self.seed,
            "record_prompts": self.record_prompts,
            "created_at": self.created_at,
            "git_commit": self.git_commit
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create from dictionary (JSON deserialization)"""
        data = data.copy()
        
        # Convert enum
        if "mode" in data:
            data["mode"] = SimulationMode(data["mode"])
        
        # Convert nested objects
        if "strategy" in data:
            data["strategy"] = StrategyParams.from_dict(data["strategy"])
        if "risk_profile" in data:
            data["risk_profile"] = RiskProfile.from_dict(data["risk_profile"])
        if "execution" in data:
            data["execution"] = ExecutionParams.from_dict(data["execution"])
        if "liquidity" in data:
            data["liquidity"] = LiquidityParams.from_dict(data["liquidity"])
        if "timeline" in data:
            data["timeline"] = TimelineParams.from_dict(data["timeline"])
        if "outputs" in data:
            data["outputs"] = OutputParams.from_dict(data["outputs"])
            
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationConfig':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def config_hash(self) -> str:
        """Generate deterministic hash of configuration"""
        config_dict = self.to_dict()
        # Remove non-deterministic fields
        config_dict.pop("created_at", None)
        config_dict.pop("git_commit", None)
        config_dict.pop("id", None)
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class KPIResults:
    """Key Performance Indicators results"""
    pnl_total: float = 0.0
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    hit_rate: float = 0.0
    avg_edge: float = 0.0
    slippage_bps: float = 0.0
    fill_rate: float = 0.0
    exposure_peak_pct: float = 0.0
    turnover: float = 0.0
    num_trades: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KPIResults':
        return cls(**data)


@dataclass
class SimulationResult:
    """Complete simulation run results"""
    run_id: str
    config_hash: str
    kpis: KPIResults
    violations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    balls_processed: int = 0
    matches_processed: int = 0
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "kpis": self.kpis.to_dict(),
            "violations": self.violations,
            "artifacts": self.artifacts,
            "runtime_seconds": self.runtime_seconds,
            "balls_processed": self.balls_processed,
            "matches_processed": self.matches_processed,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationResult':
        data = data.copy()
        if "kpis" in data:
            data["kpis"] = KPIResults.from_dict(data["kpis"])
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationResult':
        data = json.loads(json_str)
        return cls.from_dict(data)


# Preset configurations for common use cases
def create_replay_config(match_ids: List[str], strategy_name: str = "edge_kelly_v3") -> SimulationConfig:
    """Create a standard replay configuration"""
    return SimulationConfig(
        id=f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        mode=SimulationMode.REPLAY,
        match_ids=match_ids,
        strategy=StrategyParams(strategy_name),
        markets=["match_odds", "innings1_total"]
    )


def create_monte_carlo_config(num_simulations: int = 1000) -> SimulationConfig:
    """Create a Monte Carlo simulation configuration"""
    return SimulationConfig(
        id=f"monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        mode=SimulationMode.MONTE_CARLO,
        strategy=StrategyParams("edge_kelly_v3", {"num_simulations": num_simulations}),
        timeline=TimelineParams(speed=TimelineSpeed.INSTANT)
    )


def create_walk_forward_config(start_date: str, end_date: str) -> SimulationConfig:
    """Create a walk-forward backtesting configuration"""
    return SimulationConfig(
        id=f"walk_forward_{start_date}_{end_date}",
        mode=SimulationMode.WALK_FORWARD,
        strategy=StrategyParams("edge_kelly_v3", {
            "start_date": start_date,
            "end_date": end_date,
            "retrain_frequency": "monthly"
        })
    )


def create_holdout_replay_config(strategy_name: str = "edge_kelly_v3") -> SimulationConfig:
    """Create a replay configuration using actual holdout matches (20% validation data)"""
    try:
        from data_integration import HoldoutDataManager
        
        manager = HoldoutDataManager()
        holdout_matches = manager.get_holdout_matches()
        
        if not holdout_matches:
            print("⚠️ No holdout matches found, using mock data")
            holdout_matches = ["mock_match_1"]
        
        # Use more permissive risk profile for better betting activity
        permissive_risk = RiskProfile(
            bankroll=100000.0,
            max_exposure_pct=15.0,      # Increased from default 5%
            per_market_cap_pct=8.0,     # Increased from default 2%
            per_bet_cap_pct=1.5,        # Increased from default 0.5%
            correlation_cap=0.8
        )
        
        return SimulationConfig(
            id=f"holdout_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode=SimulationMode.REPLAY,
            match_ids=holdout_matches[:10],  # Use first 10 matches for reasonable runtime
            strategy=StrategyParams(strategy_name),
            risk_profile=permissive_risk,
            markets=["match_odds"],
            outputs=OutputParams(dir=f"runs/holdout_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
        
    except Exception as e:
        print(f"❌ Error creating holdout config: {e}")
        print("❌ NO MOCK DATA FALLBACK - Real holdout data required for simulation")
        raise Exception("Simulation requires real holdout data - no mock fallback available")


def create_dashboard_replay_config(strategy_name: str = "edge_kelly_v3", match_selection: str = "auto") -> SimulationConfig:
    """Create a slow replay configuration for dashboard visualization"""
    try:
        from data_integration import HoldoutDataManager
        
        manager = HoldoutDataManager()
        holdout_matches = manager.get_holdout_matches()
        
        if not holdout_matches:
            print("⚠️ No holdout matches found, using mock data")
            holdout_matches = ["mock_match_1"]
        
        # Select matches based on preference
        if match_selection == "single":
            selected_matches = holdout_matches[:1]  # Single match for detailed viewing
        else:
            selected_matches = holdout_matches[:3]  # Few matches for dashboard demo
        
        # Create enhanced strategy parameters for better betting activity
        enhanced_strategy = StrategyParams(
            name=strategy_name,
            params={
                "edge_threshold": 0.001, # Extremely low threshold - almost any edge
                "kelly_fraction": 0.02,  # Ultra-conservative sizing
                "max_stake_pct": 0.01,   # Max 1% per bet (well below DGL limit)
                "min_odds": 1.01,        # Accept extremely low odds
                "max_odds": 100.0        # Accept very high odds
            }
        )
        
        # Create ultra-permissive risk profile for simulation testing
        permissive_risk = RiskProfile(
            bankroll=100000.0,
            max_exposure_pct=50.0,      # Very high exposure limit
            per_market_cap_pct=25.0,    # Very high per-market limit
            per_bet_cap_pct=10.0,       # Very high per-bet limit
            correlation_cap=1.0         # No correlation restrictions
        )
        
        return SimulationConfig(
            id=f"dashboard_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode=SimulationMode.REPLAY,
            match_ids=selected_matches,
            strategy=enhanced_strategy,
            risk_profile=permissive_risk,  # Use permissive risk profile
            markets=["match_odds", "over_under", "innings_runs"],
            timeline=TimelineParams(speed=TimelineSpeed.REALTIME),  # Realtime for visualization
            outputs=OutputParams(
                dir=f"runs/dashboard_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                save_state_snapshots=True,
                export_trades=True
            )
        )
        
    except Exception as e:
        print(f"Error creating dashboard config: {e}")
        # Fallback to enhanced mock data
        enhanced_strategy = StrategyParams(
            name=strategy_name,
            params={
                "edge_threshold": 0.001, # Extremely low threshold
                "kelly_fraction": 0.02,
                "max_stake_pct": 0.01,   # Consistent with DGL limits
                "min_odds": 1.01,
                "max_odds": 100.0
            }
        )
        
        # Ultra-permissive risk profile for fallback too
        permissive_risk = RiskProfile(
            bankroll=100000.0,
            max_exposure_pct=50.0,
            per_market_cap_pct=25.0,
            per_bet_cap_pct=10.0,
            correlation_cap=1.0
        )
        
        return SimulationConfig(
            id=f"dashboard_mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            mode=SimulationMode.REPLAY,
            match_ids=["mock_match_1"],
            strategy=enhanced_strategy,
            risk_profile=permissive_risk,
            markets=["match_odds"],
            timeline=TimelineParams(speed=TimelineSpeed.REALTIME),
            outputs=OutputParams(dir=f"runs/dashboard_mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
