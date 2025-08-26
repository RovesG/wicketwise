# Purpose: Streamlit UI for WicketWise Simulator
# Author: WicketWise AI, Last Modified: 2024

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from .config import (
    SimulationConfig, SimulationMode, TimelineSpeed, 
    create_replay_config, create_monte_carlo_config, create_walk_forward_config
)
from .orchestrator import SimOrchestrator


def render_simulator_tab():
    """Render the main simulator tab"""
    
    st.title("🏏 WicketWise Simulator & Market Replay")
    st.markdown("---")
    
    # Initialize session state
    if 'sim_orchestrator' not in st.session_state:
        st.session_state.sim_orchestrator = SimOrchestrator()
    if 'sim_config' not in st.session_state:
        st.session_state.sim_config = None
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
    if 'sim_result' not in st.session_state:
        st.session_state.sim_result = None
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_config_panel()
    
    with col2:
        render_simulation_panel()
    
    # Bottom section for results
    if st.session_state.sim_result:
        st.markdown("---")
        render_results_panel()


def render_config_panel():
    """Render simulation configuration panel"""
    
    st.subheader("📋 Configuration")
    
    # Simulation mode
    mode = st.selectbox(
        "Simulation Mode",
        options=["replay", "monte_carlo", "walk_forward", "paper"],
        help="Choose simulation mode"
    )
    
    # Quick presets
    st.markdown("**Quick Presets:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Replay", help="Historical replay"):
            st.session_state.sim_config = create_replay_config(["test_match_1"], "edge_kelly_v3")
    
    with col2:
        if st.button("🎲 Monte Carlo", help="Synthetic generation"):
            st.session_state.sim_config = create_monte_carlo_config(1000)
    
    with col3:
        if st.button("📈 Walk Forward", help="Backtesting"):
            st.session_state.sim_config = create_walk_forward_config("2023-01-01", "2024-01-01")
    
    st.markdown("---")
    
    # Detailed configuration
    with st.expander("🔧 Detailed Configuration", expanded=False):
        render_detailed_config(mode)
    
    # Load/Save presets
    with st.expander("💾 Presets", expanded=False):
        render_preset_management()


def render_detailed_config(mode: str):
    """Render detailed configuration options"""
    
    # Basic settings
    st.markdown("**Basic Settings**")
    
    sim_id = st.text_input("Simulation ID", value=f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Markets
    markets = st.multiselect(
        "Markets",
        options=["match_odds", "innings1_total", "innings2_win", "top_batsman", "top_bowler"],
        default=["match_odds"],
        help="Select markets to simulate"
    )
    
    # Match selection
    if mode == "replay":
        match_ids = st.text_area(
            "Match IDs (one per line)",
            value="test_match_1\ntest_match_2",
            help="Enter match IDs to replay"
        ).strip().split('\n')
        match_ids = [mid.strip() for mid in match_ids if mid.strip()]
    else:
        match_ids = []
    
    # Strategy configuration
    st.markdown("**Strategy Configuration**")
    
    strategy_name = st.selectbox(
        "Strategy",
        options=["edge_kelly_v3", "mean_revert_lob", "momentum_follow"],
        help="Select trading strategy"
    )
    
    # Strategy parameters
    strategy_params = {}
    if strategy_name == "edge_kelly_v3":
        col1, col2 = st.columns(2)
        with col1:
            strategy_params["edge_threshold"] = st.number_input("Edge Threshold", value=0.02, step=0.001, format="%.3f")
            strategy_params["kelly_fraction"] = st.number_input("Kelly Fraction", value=0.25, step=0.05, format="%.2f")
        with col2:
            strategy_params["max_stake_pct"] = st.number_input("Max Stake %", value=5.0, step=0.5, format="%.1f")
            strategy_params["min_odds"] = st.number_input("Min Odds", value=1.1, step=0.1, format="%.1f")
    
    # Risk profile
    st.markdown("**Risk Profile**")
    
    col1, col2 = st.columns(2)
    with col1:
        bankroll = st.number_input("Bankroll (£)", value=100000.0, step=1000.0, format="%.0f")
        max_exposure_pct = st.number_input("Max Exposure %", value=5.0, step=0.5, format="%.1f")
    with col2:
        per_market_cap_pct = st.number_input("Per Market Cap %", value=2.0, step=0.1, format="%.1f")
        per_bet_cap_pct = st.number_input("Per Bet Cap %", value=0.5, step=0.1, format="%.1f")
    
    # Execution parameters
    st.markdown("**Execution Parameters**")
    
    col1, col2 = st.columns(2)
    with col1:
        latency_mean = st.number_input("Latency Mean (ms)", value=250.0, step=10.0, format="%.0f")
        commission_bps = st.number_input("Commission (bps)", value=200.0, step=10.0, format="%.0f")
    with col2:
        participation_factor = st.number_input("Participation Factor", value=0.1, step=0.01, format="%.2f")
        slippage_model = st.selectbox("Slippage Model", options=["lob_queue", "linear", "none"])
    
    # Timeline
    st.markdown("**Timeline**")
    
    col1, col2 = st.columns(2)
    with col1:
        speed = st.selectbox("Speed", options=["realtime", "x10", "instant"], index=2)
    with col2:
        seed = st.number_input("Random Seed", value=42, step=1)
    
    # Create configuration
    if st.button("📝 Create Configuration", type="primary"):
        from .config import (
            StrategyParams, RiskProfile, ExecutionParams, 
            TimelineParams, OutputParams, SimulationMode
        )
        
        config = SimulationConfig(
            id=sim_id,
            mode=SimulationMode(mode),
            markets=markets,
            match_ids=match_ids,
            strategy=StrategyParams(strategy_name, strategy_params),
            risk_profile=RiskProfile(
                bankroll=bankroll,
                max_exposure_pct=max_exposure_pct,
                per_market_cap_pct=per_market_cap_pct,
                per_bet_cap_pct=per_bet_cap_pct
            ),
            execution=ExecutionParams(
                latency_ms={"mean": latency_mean, "std": latency_mean * 0.2},
                commission_bps=commission_bps,
                participation_factor=participation_factor,
                slippage_model=slippage_model
            ),
            timeline=TimelineParams(speed=TimelineSpeed(speed)),
            seed=int(seed),
            outputs=OutputParams(dir=f"runs/{sim_id}")
        )
        
        st.session_state.sim_config = config
        st.success("✅ Configuration created successfully!")


def render_preset_management():
    """Render preset load/save functionality"""
    
    # Save current config
    if st.session_state.sim_config:
        preset_name = st.text_input("Preset Name", value="my_preset")
        
        if st.button("💾 Save Preset"):
            presets_dir = Path("sim/presets")
            presets_dir.mkdir(exist_ok=True)
            
            preset_file = presets_dir / f"{preset_name}.json"
            with open(preset_file, 'w') as f:
                f.write(st.session_state.sim_config.to_json())
            
            st.success(f"✅ Preset saved as {preset_name}")
    
    # Load existing presets
    presets_dir = Path("sim/presets")
    if presets_dir.exists():
        preset_files = list(presets_dir.glob("*.json"))
        
        if preset_files:
            preset_names = [f.stem for f in preset_files]
            selected_preset = st.selectbox("Load Preset", options=[""] + preset_names)
            
            if selected_preset and st.button("📂 Load Preset"):
                preset_file = presets_dir / f"{selected_preset}.json"
                with open(preset_file, 'r') as f:
                    config_data = json.load(f)
                
                st.session_state.sim_config = SimulationConfig.from_dict(config_data)
                st.success(f"✅ Loaded preset: {selected_preset}")


def render_simulation_panel():
    """Render simulation control and monitoring panel"""
    
    st.subheader("🚀 Simulation Control")
    
    # Configuration status
    if st.session_state.sim_config:
        st.success(f"✅ Configuration ready: {st.session_state.sim_config.id}")
        
        # Configuration summary
        with st.expander("📊 Configuration Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mode", st.session_state.sim_config.mode.value.title())
                st.metric("Strategy", st.session_state.sim_config.strategy.name)
                st.metric("Markets", len(st.session_state.sim_config.markets))
            
            with col2:
                st.metric("Bankroll", f"£{st.session_state.sim_config.risk_profile.bankroll:,.0f}")
                st.metric("Max Exposure", f"{st.session_state.sim_config.risk_profile.max_exposure_pct}%")
                st.metric("Seed", st.session_state.sim_config.seed)
            
            with col3:
                st.metric("Latency", f"{st.session_state.sim_config.execution.latency_ms['mean']:.0f}ms")
                st.metric("Commission", f"{st.session_state.sim_config.execution.commission_bps:.0f}bps")
                st.metric("Speed", st.session_state.sim_config.timeline.speed.value)
    
    else:
        st.warning("⚠️ No configuration loaded. Please create or load a configuration.")
        return
    
    # Simulation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Simulation", disabled=st.session_state.sim_running, type="primary"):
            start_simulation()
    
    with col2:
        if st.button("⏹️ Stop Simulation", disabled=not st.session_state.sim_running):
            stop_simulation()
    
    with col3:
        if st.button("🔄 Reset", disabled=st.session_state.sim_running):
            reset_simulation()
    
    # Progress monitoring
    if st.session_state.sim_running:
        render_progress_monitor()
    
    # Real-time charts
    if st.session_state.sim_running or st.session_state.sim_result:
        render_realtime_charts()


def start_simulation():
    """Start the simulation"""
    
    try:
        # Initialize orchestrator
        if st.session_state.sim_orchestrator.initialize(st.session_state.sim_config):
            st.session_state.sim_running = True
            st.success("🚀 Simulation started!")
            
            # Run simulation in background (simplified for demo)
            # In production, this would use threading or async
            with st.spinner("Running simulation..."):
                result = st.session_state.sim_orchestrator.run()
                st.session_state.sim_result = result
                st.session_state.sim_running = False
            
            st.success("✅ Simulation completed!")
            st.rerun()
        
        else:
            st.error("❌ Failed to initialize simulation")
    
    except Exception as e:
        st.error(f"❌ Simulation error: {str(e)}")
        st.session_state.sim_running = False


def stop_simulation():
    """Stop the running simulation"""
    st.session_state.sim_orchestrator.stop()
    st.session_state.sim_running = False
    st.warning("⏹️ Simulation stopped")


def reset_simulation():
    """Reset simulation state"""
    st.session_state.sim_result = None
    st.session_state.sim_running = False
    st.info("🔄 Simulation reset")


def render_progress_monitor():
    """Render real-time progress monitoring"""
    
    progress_placeholder = st.empty()
    
    # Get progress (simplified - would be real-time in production)
    progress = st.session_state.sim_orchestrator.get_progress()
    
    with progress_placeholder.container():
        st.markdown("**📊 Progress**")
        
        # Progress bar
        st.progress(progress.get("progress", 0.0))
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Events", f"{progress.get('events_processed', 0):,}")
        
        with col2:
            st.metric("Elapsed", f"{progress.get('elapsed_seconds', 0):.0f}s")
        
        with col3:
            st.metric("Current P&L", f"£{progress.get('current_pnl', 0):.2f}")
        
        with col4:
            st.metric("Status", progress.get("status", "unknown").title())


def render_realtime_charts():
    """Render real-time simulation charts"""
    
    st.markdown("**📈 Live Charts**")
    
    # Create mock data for demo (would be real data in production)
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=2), end=datetime.now(), freq='1min')
    
    # P&L chart
    pnl_data = pd.DataFrame({
        'timestamp': timestamps,
        'pnl': [100 * i + 50 * (i % 10) for i in range(len(timestamps))],
        'balance': [100000 + 100 * i + 50 * (i % 10) for i in range(len(timestamps))]
    })
    
    fig_pnl = px.line(pnl_data, x='timestamp', y='pnl', title='P&L Over Time')
    fig_pnl.update_layout(height=300)
    st.plotly_chart(fig_pnl, use_container_width=True)
    
    # Exposure and other metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Exposure chart
        exposure_data = pd.DataFrame({
            'timestamp': timestamps,
            'exposure': [5000 + 1000 * (i % 5) for i in range(len(timestamps))]
        })
        
        fig_exposure = px.line(exposure_data, x='timestamp', y='exposure', title='Exposure Over Time')
        fig_exposure.update_layout(height=250)
        st.plotly_chart(fig_exposure, use_container_width=True)
    
    with col2:
        # Drawdown chart
        drawdown_data = pd.DataFrame({
            'timestamp': timestamps,
            'drawdown': [max(0, -2 + 0.1 * (i % 20)) for i in range(len(timestamps))]
        })
        
        fig_dd = px.line(drawdown_data, x='timestamp', y='drawdown', title='Drawdown %')
        fig_dd.update_layout(height=250)
        st.plotly_chart(fig_dd, use_container_width=True)


def render_results_panel():
    """Render simulation results panel"""
    
    st.subheader("📊 Simulation Results")
    
    result = st.session_state.sim_result
    
    # KPI Summary
    st.markdown("**🎯 Key Performance Indicators**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total P&L", f"£{result.kpis.pnl_total:.2f}")
        st.metric("Sharpe Ratio", f"{result.kpis.sharpe:.2f}")
    
    with col2:
        st.metric("Max Drawdown", f"{result.kpis.max_drawdown:.1f}%")
        st.metric("Hit Rate", f"{result.kpis.hit_rate:.1%}")
    
    with col3:
        st.metric("Avg Edge", f"{result.kpis.avg_edge:.1%}")
        st.metric("Fill Rate", f"{result.kpis.fill_rate:.1%}")
    
    with col4:
        st.metric("Slippage", f"{result.kpis.slippage_bps:.0f}bps")
        st.metric("Trades", f"{result.kpis.num_trades:,}")
    
    # Detailed results
    col1, col2 = st.columns(2)
    
    with col1:
        # Violations
        if result.violations:
            st.markdown("**⚠️ Violations**")
            for violation in result.violations:
                st.warning(violation)
        else:
            st.success("✅ No violations detected")
    
    with col2:
        # Runtime info
        st.markdown("**⏱️ Runtime Information**")
        st.info(f"Runtime: {result.runtime_seconds:.1f}s")
        st.info(f"Balls processed: {result.balls_processed:,}")
        st.info(f"Matches: {result.matches_processed}")
    
    # Export options
    st.markdown("**📤 Export Options**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export CSV"):
            export_csv_results(result)
    
    with col2:
        if st.button("📋 Export Report"):
            export_html_report(result)
    
    with col3:
        if st.button("💾 Save Results"):
            save_results(result)


def export_csv_results(result):
    """Export results to CSV"""
    # Create CSV data (simplified)
    csv_data = f"metric,value\nTotal P&L,{result.kpis.pnl_total}\nSharpe Ratio,{result.kpis.sharpe}\n"
    
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"{result.run_id}_results.csv",
        mime="text/csv"
    )


def export_html_report(result):
    """Export HTML report"""
    html_content = f"""
    <html>
    <head><title>Simulation Report: {result.run_id}</title></head>
    <body>
    <h1>WicketWise Simulation Report</h1>
    <h2>Run ID: {result.run_id}</h2>
    <p>Total P&L: £{result.kpis.pnl_total:.2f}</p>
    <p>Sharpe Ratio: {result.kpis.sharpe:.2f}</p>
    </body>
    </html>
    """
    
    st.download_button(
        label="Download Report",
        data=html_content,
        file_name=f"{result.run_id}_report.html",
        mime="text/html"
    )


def save_results(result):
    """Save results to file"""
    results_dir = Path("sim/results")
    results_dir.mkdir(exist_ok=True)
    
    result_file = results_dir / f"{result.run_id}_result.json"
    with open(result_file, 'w') as f:
        f.write(result.to_json())
    
    st.success(f"✅ Results saved to {result_file}")


# Main entry point for integration with existing Streamlit app
def add_simulator_tab():
    """Add simulator tab to existing Streamlit app"""
    
    # This would be called from the main Streamlit app
    # Example integration:
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "DGL", "Simulator", "Settings"])
    
    with tab3:
        render_simulator_tab()


if __name__ == "__main__":
    # Standalone mode for testing
    st.set_page_config(
        page_title="WicketWise Simulator",
        page_icon="🏏",
        layout="wide"
    )
    
    render_simulator_tab()
