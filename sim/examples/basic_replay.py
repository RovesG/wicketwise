#!/usr/bin/env python3
# Purpose: Basic replay simulation example
# Author: WicketWise AI, Last Modified: 2024

"""
Basic Replay Simulation Example

Demonstrates how to set up and run a basic historical replay simulation
using the WicketWise SIM system.
"""

import sys
from pathlib import Path

# Add sim directory to path
sim_dir = Path(__file__).parent.parent
sys.path.insert(0, str(sim_dir))

from config import SimulationConfig, SimulationMode, StrategyParams, RiskProfile
from orchestrator import SimOrchestrator
import json

def create_basic_replay_config():
    """Create a basic replay configuration"""
    
    config = SimulationConfig(
        id="basic_replay_example",
        mode=SimulationMode.REPLAY,
        markets=["match_odds"],
        match_ids=["example_match_1"],
        strategy=StrategyParams(
            name="edge_kelly_v3",
            params={
                "edge_threshold": 0.02,
                "kelly_fraction": 0.25,
                "max_stake_pct": 0.05,
                "min_odds": 1.1,
                "max_odds": 10.0
            }
        ),
        risk_profile=RiskProfile(
            bankroll=100000.0,
            max_exposure_pct=5.0,
            per_market_cap_pct=2.0,
            per_bet_cap_pct=0.5
        ),
        seed=42
    )
    
    return config

def run_basic_replay():
    """Run basic replay simulation"""
    
    print("🏏 WicketWise SIM - Basic Replay Example")
    print("=" * 50)
    
    # Create configuration
    config = create_basic_replay_config()
    
    print(f"📋 Configuration:")
    print(f"  - Mode: {config.mode.value}")
    print(f"  - Strategy: {config.strategy.name}")
    print(f"  - Bankroll: £{config.risk_profile.bankroll:,.0f}")
    print(f"  - Max Exposure: {config.risk_profile.max_exposure_pct}%")
    print(f"  - Seed: {config.seed}")
    
    # Initialize orchestrator
    orchestrator = SimOrchestrator()
    
    print("\n🚀 Initializing simulation...")
    if not orchestrator.initialize(config):
        print("❌ Failed to initialize simulation")
        return
    
    print("✅ Simulation initialized successfully")
    
    # Run simulation
    print("\n▶️ Running simulation...")
    result = orchestrator.run()
    
    if result:
        print("✅ Simulation completed successfully!")
        
        # Display results
        print(f"\n📊 Results Summary:")
        print(f"  - Run ID: {result.run_id}")
        print(f"  - Runtime: {result.runtime_seconds:.1f}s")
        print(f"  - Balls Processed: {result.balls_processed:,}")
        print(f"  - Matches: {result.matches_processed}")
        
        print(f"\n💰 Performance Metrics:")
        kpis = result.kpis
        print(f"  - Total P&L: £{kpis.pnl_total:.2f}")
        print(f"  - Realized P&L: £{kpis.pnl_realized:.2f}")
        print(f"  - Unrealized P&L: £{kpis.pnl_unrealized:.2f}")
        print(f"  - Sharpe Ratio: {kpis.sharpe:.3f}")
        print(f"  - Max Drawdown: {kpis.max_drawdown:.1f}%")
        print(f"  - Hit Rate: {kpis.hit_rate:.1%}")
        print(f"  - Avg Edge: {kpis.avg_edge:.1%}")
        print(f"  - Fill Rate: {kpis.fill_rate:.1%}")
        print(f"  - Slippage: {kpis.slippage_bps:.0f}bps")
        print(f"  - Total Trades: {kpis.num_trades}")
        
        if result.violations:
            print(f"\n⚠️ Violations:")
            for violation in result.violations:
                print(f"  - {violation}")
        else:
            print(f"\n✅ No governance violations")
        
        print(f"\n📁 Artifacts:")
        for artifact in result.artifacts:
            print(f"  - {artifact}")
        
        # Save result to file
        result_file = Path("basic_replay_result.json")
        with open(result_file, 'w') as f:
            f.write(result.to_json())
        
        print(f"\n💾 Result saved to: {result_file}")
        
    else:
        print("❌ Simulation failed")

if __name__ == "__main__":
    run_basic_replay()
