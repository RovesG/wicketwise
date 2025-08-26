#!/usr/bin/env python3
# Purpose: Strategy comparison example
# Author: WicketWise AI, Last Modified: 2024

"""
Strategy Comparison Example

Demonstrates how to compare multiple trading strategies using
the WicketWise SIM system with identical market conditions.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add sim directory to path
sim_dir = Path(__file__).parent.parent
sys.path.insert(0, str(sim_dir))

from config import SimulationConfig, SimulationMode, StrategyParams, RiskProfile
from orchestrator import SimOrchestrator

def create_strategy_configs():
    """Create configurations for different strategies"""
    
    base_config = {
        "mode": SimulationMode.REPLAY,
        "markets": ["match_odds"],
        "match_ids": ["comparison_match_1"],
        "risk_profile": RiskProfile(
            bankroll=100000.0,
            max_exposure_pct=5.0,
            per_market_cap_pct=2.0,
            per_bet_cap_pct=0.5
        ),
        "seed": 42  # Same seed for fair comparison
    }
    
    strategies = [
        {
            "name": "edge_kelly_v3",
            "display_name": "Edge Kelly",
            "params": {
                "edge_threshold": 0.02,
                "kelly_fraction": 0.25,
                "max_stake_pct": 0.05
            }
        },
        {
            "name": "mean_revert_lob",
            "display_name": "Mean Revert",
            "params": {
                "revert_threshold": 0.02,
                "max_position": 1000.0,
                "hold_time_seconds": 30.0
            }
        },
        {
            "name": "momentum_follow",
            "display_name": "Momentum Follow",
            "params": {
                "momentum_threshold": 0.01,
                "event_multiplier": 1.5,
                "max_stake": 1000.0
            }
        }
    ]
    
    configs = []
    for strategy in strategies:
        config = SimulationConfig(
            id=f"comparison_{strategy['name']}",
            strategy=StrategyParams(strategy["name"], strategy["params"]),
            **base_config
        )
        configs.append((strategy["display_name"], config))
    
    return configs

def run_strategy_comparison():
    """Run strategy comparison simulation"""
    
    print("🏏 WicketWise SIM - Strategy Comparison")
    print("=" * 50)
    
    # Create strategy configurations
    strategy_configs = create_strategy_configs()
    
    print(f"📋 Comparing {len(strategy_configs)} strategies:")
    for display_name, config in strategy_configs:
        print(f"  - {display_name} ({config.strategy.name})")
    
    results = []
    
    # Run each strategy
    for display_name, config in strategy_configs:
        print(f"\n🚀 Running {display_name}...")
        
        orchestrator = SimOrchestrator()
        
        if orchestrator.initialize(config):
            result = orchestrator.run()
            
            if result:
                print(f"✅ {display_name} completed")
                results.append((display_name, config, result))
            else:
                print(f"❌ {display_name} failed")
        else:
            print(f"❌ {display_name} initialization failed")
    
    if not results:
        print("❌ No strategies completed successfully")
        return
    
    # Analyze and compare results
    print(f"\n📊 Strategy Comparison Results")
    print("=" * 50)
    
    comparison_data = []
    
    for display_name, config, result in results:
        kpis = result.kpis
        
        comparison_data.append({
            "Strategy": display_name,
            "Total P&L (£)": kpis.pnl_total,
            "Sharpe Ratio": kpis.sharpe,
            "Max Drawdown (%)": kpis.max_drawdown,
            "Hit Rate (%)": kpis.hit_rate * 100,
            "Avg Edge (%)": kpis.avg_edge * 100,
            "Fill Rate (%)": kpis.fill_rate * 100,
            "Slippage (bps)": kpis.slippage_bps,
            "Total Trades": kpis.num_trades,
            "Runtime (s)": result.runtime_seconds,
            "Violations": len(result.violations)
        })
    
    # Create comparison table
    df = pd.DataFrame(comparison_data)
    
    print("\n📈 Performance Summary:")
    print(df.to_string(index=False, float_format="%.2f"))
    
    # Identify best performers
    print(f"\n🏆 Best Performers:")
    
    best_pnl = df.loc[df["Total P&L (£)"].idxmax()]
    print(f"  - Highest P&L: {best_pnl['Strategy']} (£{best_pnl['Total P&L (£)']:.2f})")
    
    best_sharpe = df.loc[df["Sharpe Ratio"].idxmax()]
    print(f"  - Best Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.3f})")
    
    best_hit_rate = df.loc[df["Hit Rate (%)"].idxmax()]
    print(f"  - Best Hit Rate: {best_hit_rate['Strategy']} ({best_hit_rate['Hit Rate (%)']:.1f}%)")
    
    lowest_drawdown = df.loc[df["Max Drawdown (%)"].idxmin()]
    print(f"  - Lowest Drawdown: {lowest_drawdown['Strategy']} ({lowest_drawdown['Max Drawdown (%)']:.1f}%)")
    
    # Risk-adjusted ranking
    print(f"\n⚖️ Risk-Adjusted Ranking:")
    df["Risk Score"] = df["Sharpe Ratio"] - (df["Max Drawdown (%)"] / 100)
    df_ranked = df.sort_values("Risk Score", ascending=False)
    
    for i, row in df_ranked.iterrows():
        print(f"  {df_ranked.index.get_loc(i) + 1}. {row['Strategy']} (Score: {row['Risk Score']:.3f})")
    
    # Save detailed results
    comparison_file = Path("strategy_comparison_results.json")
    detailed_results = {
        "comparison_summary": comparison_data,
        "detailed_results": [
            {
                "strategy": display_name,
                "config": config.to_dict(),
                "result": result.to_dict()
            }
            for display_name, config, result in results
        ]
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {comparison_file}")
    
    # Save CSV for further analysis
    csv_file = Path("strategy_comparison_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"📊 Summary CSV saved to: {csv_file}")

if __name__ == "__main__":
    run_strategy_comparison()
