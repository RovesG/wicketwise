# Purpose: SIM orchestrator for run management, seeds, and telemetry
# Author: WicketWise AI, Last Modified: 2024

import random
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
import json

try:
    from .config import SimulationConfig, SimulationResult, KPIResults, SimulationMode
    from .state import MatchState, MarketState
    from .strategy import Strategy, StrategyAction, FillEvent, AccountState, create_strategy
    from .adapters import EnvironmentAdapter, ReplayAdapter, SyntheticAdapter, WalkForwardAdapter
    from .matching import MatchingEngine
    from .dgl_adapter import SimDGLAdapter, DGLDecision
    from .metrics import SimMetrics
except ImportError:
    # Fallback for standalone execution
    from config import SimulationConfig, SimulationResult, KPIResults, SimulationMode
    from state import MatchState, MarketState
    from strategy import Strategy, StrategyAction, FillEvent, AccountState, create_strategy
    from adapters import EnvironmentAdapter, ReplayAdapter, SyntheticAdapter, WalkForwardAdapter
    from matching import MatchingEngine
    from dgl_adapter import SimDGLAdapter, DGLDecision
    from metrics import SimMetrics


class SimOrchestrator:
    """
    Main orchestrator for simulation runs
    
    Manages the complete simulation lifecycle:
    - Environment setup and seeding
    - Strategy execution loop
    - DGL enforcement
    - Metrics collection
    - Artifact generation
    """
    
    def __init__(self):
        self.config: Optional[SimulationConfig] = None
        self.environment: Optional[EnvironmentAdapter] = None
        self.strategy: Optional[Strategy] = None
        self.matching_engine: Optional[MatchingEngine] = None
        self.dgl_adapter: Optional[SimDGLAdapter] = None
        self.metrics: Optional[SimMetrics] = None
        self.account_state: Optional[AccountState] = None
        
        # Run state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.events_processed = 0
        self.total_events = 0
        
        # Artifacts
        self.order_log: List[Dict[str, Any]] = []
        self.fill_log: List[Dict[str, Any]] = []
        self.state_log: List[Dict[str, Any]] = []
        
    def initialize(self, config: SimulationConfig) -> bool:
        """
        Initialize simulation with configuration
        
        Args:
            config: Simulation configuration
            
        Returns:
            True if initialization successful
        """
        try:
            self.config = config
            
            # Set random seed for reproducibility
            random.seed(config.seed)
            
            # Initialize environment adapter
            self.environment = self._create_environment_adapter(config.mode)
            if not self.environment.initialize(config):
                return False
            
            # Initialize strategy
            self.strategy = create_strategy(config.strategy.name, config.strategy.params)
            
            # Initialize matching engine
            try:
                from .matching import LatencyModel, SlippageModel, CommissionModel
            except ImportError:
                from matching import LatencyModel, SlippageModel, CommissionModel
            
            latency_model = LatencyModel(
                mean_ms=config.execution.latency_ms["mean"],
                std_ms=config.execution.latency_ms["std"]
            )
            
            slippage_model = SlippageModel(
                model_type=config.execution.slippage_model
            )
            
            commission_model = CommissionModel(
                commission_bps=config.execution.commission_bps
            )
            
            self.matching_engine = MatchingEngine(
                latency_model=latency_model,
                slippage_model=slippage_model,
                commission_model=commission_model,
                participation_factor=config.execution.participation_factor
            )
            
            # Initialize DGL adapter
            self.dgl_adapter = SimDGLAdapter(config.risk_profile)
            
            # Initialize account state
            self.account_state = AccountState(cash=config.risk_profile.bankroll)
            
            # Initialize metrics
            self.metrics = SimMetrics()
            
            # Get total events for progress tracking
            self.total_events = self.environment.get_total_events()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize simulation: {e}")
            return False
    
    def run(self) -> SimulationResult:
        """
        Execute the complete simulation
        
        Returns:
            Simulation results with KPIs and artifacts
        """
        if not self.config:
            raise ValueError("Simulation not initialized")
        
        self.is_running = True
        self.start_time = datetime.now()
        self.events_processed = 0
        
        # Clear logs
        self.order_log.clear()
        self.fill_log.clear()
        self.state_log.clear()
        
        try:
            # Main simulation loop
            for match_state, market_state in self.environment.get_events():
                if not self.is_running:
                    break
                
                self._process_tick(match_state, market_state)
                self.events_processed += 1
                
                # Update progress periodically
                if self.events_processed % 100 == 0:
                    self._log_progress()
            
            # Finalize simulation
            return self._finalize_simulation()
            
        except Exception as e:
            print(f"Simulation error: {e}")
            return self._create_error_result(str(e))
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the simulation"""
        self.is_running = False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current simulation progress"""
        if not self.start_time:
            return {"progress": 0.0, "status": "not_started"}
        
        progress = self.events_processed / self.total_events if self.total_events > 0 else 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "progress": progress,
            "events_processed": self.events_processed,
            "total_events": self.total_events,
            "elapsed_seconds": elapsed,
            "status": "running" if self.is_running else "completed",
            "current_pnl": self.account_state.total_balance() - self.config.risk_profile.bankroll if self.account_state else 0.0
        }
    
    def _create_environment_adapter(self, mode: SimulationMode) -> EnvironmentAdapter:
        """Create appropriate environment adapter"""
        if mode == SimulationMode.REPLAY:
            return ReplayAdapter()
        elif mode == SimulationMode.MONTE_CARLO:
            return SyntheticAdapter()
        elif mode == SimulationMode.WALK_FORWARD:
            return WalkForwardAdapter()
        else:
            # Default to replay
            return ReplayAdapter()
    
    def _process_tick(self, match_state: MatchState, market_state: MarketState):
        """Process a single simulation tick"""
        
        # Update matching engine with market state
        for market_id, snapshot in market_state.market_snapshots.items():
            self.matching_engine.update_market_state(snapshot)
        
        # Get strategy actions
        strategy_actions = self.strategy.on_tick(match_state, market_state, self.account_state)
        
        # Process each action through DGL and matching engine
        for action in strategy_actions:
            self._process_action(action, match_state, market_state)
        
        # Log current state
        self._log_state(match_state, market_state)
        
        # Update metrics
        self.metrics.update_tick(match_state, market_state, self.account_state)
    
    def _process_action(self, action: StrategyAction, match_state: MatchState, market_state: MarketState):
        """Process a single strategy action"""
        
        # Log the action
        self.order_log.append({
            "timestamp": action.ts,
            "action": action.to_dict(),
            "match_state": {
                "over": match_state.over,
                "ball": match_state.ball,
                "score": match_state.score,
                "wickets": match_state.wickets
            }
        })
        
        # Check with DGL
        dgl_response = self.dgl_adapter.evaluate_action(action, self.account_state)
        
        if dgl_response.decision == DGLDecision.REJECT:
            # Action rejected by DGL
            self.fill_log.append({
                "timestamp": action.ts,
                "client_order_id": action.client_order_id,
                "fill_qty": 0.0,
                "reason": f"DGL_REJECT: {dgl_response.reason}",
                "dgl_response": dgl_response.to_dict()
            })
            return
        
        # Amend size if required by DGL
        if dgl_response.decision == DGLDecision.AMEND and dgl_response.amended_size:
            action.size = dgl_response.amended_size
        
        # Submit to matching engine
        fill_events = self.matching_engine.handle_action(action)
        
        # Process fills
        for fill in fill_events:
            self._process_fill(fill, action, dgl_response)
    
    def _process_fill(self, fill: FillEvent, action: StrategyAction, dgl_response):
        """Process a fill event"""
        
        # Update account state
        self.account_state.update_from_fill(fill, action)
        
        # Update DGL exposures
        if fill.fill_qty > 0:
            self.dgl_adapter.update_exposures(action, fill.fill_qty)
        
        # Log the fill
        self.fill_log.append({
            "timestamp": fill.ts,
            "fill": fill.to_dict(),
            "action": action.to_dict(),
            "dgl_response": dgl_response.to_dict(),
            "account_balance": self.account_state.total_balance()
        })
        
        # Update metrics
        self.metrics.update_fill(fill, action, self.account_state)
    
    def _log_state(self, match_state: MatchState, market_state: MarketState):
        """Log current simulation state"""
        
        # Log every 10th tick to avoid excessive logging
        if self.events_processed % 10 == 0:
            self.state_log.append({
                "timestamp": datetime.now().isoformat(),
                "events_processed": self.events_processed,
                "match_state": match_state.to_dict(),
                "account_state": self.account_state.to_dict(),
                "dgl_exposures": self.dgl_adapter.get_current_exposures()
            })
    
    def _log_progress(self):
        """Log simulation progress"""
        progress = self.get_progress()
        print(f"Simulation progress: {progress['progress']:.1%} "
              f"({progress['events_processed']}/{progress['total_events']}) "
              f"- PnL: £{progress['current_pnl']:.2f}")
    
    def _finalize_simulation(self) -> SimulationResult:
        """Finalize simulation and generate results"""
        
        end_time = datetime.now()
        runtime_seconds = (end_time - self.start_time).total_seconds()
        
        # Calculate final KPIs
        kpis = self.metrics.calculate_final_kpis(self.account_state, self.config.risk_profile.bankroll)
        
        # Create output directory
        output_dir = Path(self.config.outputs.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save artifacts
        artifacts = self._save_artifacts(output_dir)
        
        # Get DGL violations
        violations = []
        dgl_stats = self.dgl_adapter.get_stats()
        for rule_id, count in dgl_stats["violations_by_rule"].items():
            violations.append(f"{rule_id}: {count} violations")
        
        # Create result
        result = SimulationResult(
            run_id=self.config.id,
            config_hash=self.config.config_hash(),
            kpis=kpis,
            violations=violations,
            artifacts=artifacts,
            runtime_seconds=runtime_seconds,
            balls_processed=self.events_processed,
            matches_processed=len(self.config.match_ids) if self.config.match_ids else 1,
            completed_at=end_time.isoformat()
        )
        
        # Save result
        result_file = output_dir / "simulation_result.json"
        with open(result_file, 'w') as f:
            f.write(result.to_json())
        
        return result
    
    def _save_artifacts(self, output_dir: Path) -> List[str]:
        """Save simulation artifacts"""
        artifacts = []
        
        if "orders" in self.config.outputs.artifacts:
            orders_file = output_dir / "orders.jsonl"
            with open(orders_file, 'w') as f:
                for order in self.order_log:
                    f.write(json.dumps(order) + '\n')
            artifacts.append(str(orders_file))
        
        if "fills" in self.config.outputs.artifacts:
            fills_file = output_dir / "fills.jsonl"
            with open(fills_file, 'w') as f:
                for fill in self.fill_log:
                    f.write(json.dumps(fill) + '\n')
            artifacts.append(str(fills_file))
        
        if "dgl" in self.config.outputs.artifacts:
            dgl_file = output_dir / "dgl_decisions.jsonl"
            self.dgl_adapter.export_audit_log(str(dgl_file))
            artifacts.append(str(dgl_file))
        
        if "metrics" in self.config.outputs.artifacts:
            metrics_file = output_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    "dgl_stats": self.dgl_adapter.get_stats(),
                    "matching_stats": self.matching_engine.get_stats(),
                    "strategy_params": self.strategy.get_params()
                }, f, indent=2)
            artifacts.append(str(metrics_file))
        
        # Save configuration
        config_file = output_dir / "config.json"
        with open(config_file, 'w') as f:
            f.write(self.config.to_json())
        artifacts.append(str(config_file))
        
        return artifacts
    
    def _create_error_result(self, error_message: str) -> SimulationResult:
        """Create error result"""
        return SimulationResult(
            run_id=self.config.id if self.config else "error_run",
            config_hash="",
            kpis=KPIResults(),
            violations=[f"SIMULATION_ERROR: {error_message}"],
            artifacts=[],
            runtime_seconds=0.0,
            balls_processed=self.events_processed,
            matches_processed=0
        )
