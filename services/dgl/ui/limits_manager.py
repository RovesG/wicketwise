# Purpose: Limits management interface for DGL
# Author: WicketWise AI, Last Modified: 2024

"""
Limits Manager

Specialized interface for managing DGL limits and rules:
- Interactive rule configuration
- Impact analysis and simulation
- Approval workflow integration
- Historical limits tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.dgl_client import DGLClient, DGLClientConfig
from schemas import RuleId


logger = logging.getLogger(__name__)


class LimitsManager:
    """
    Specialized limits management interface for DGL
    
    Provides comprehensive interface for:
    - Rule configuration and updates
    - Impact analysis and simulation
    - Approval workflow management
    - Historical limits tracking
    """
    
    def __init__(self, dgl_base_url: str = "http://localhost:8001"):
        """
        Initialize limits manager
        
        Args:
            dgl_base_url: DGL service base URL
        """
        self.dgl_base_url = dgl_base_url
        self.client_config = DGLClientConfig(base_url=dgl_base_url)
        
        # Initialize session state for limits management
        if 'limits_data' not in st.session_state:
            st.session_state.limits_data = {}
        
        if 'pending_changes' not in st.session_state:
            st.session_state.pending_changes = {}
        
        if 'approval_requests' not in st.session_state:
            st.session_state.approval_requests = []
    
    def render(self):
        """Render the limits management interface"""
        
        st.markdown("# 🔧 Limits & Rules Management")
        st.markdown("Configure and manage DGL governance rules with impact analysis and approval workflows.")
        
        # Alert for pending changes
        if st.session_state.pending_changes:
            st.warning(f"⚠️ {len(st.session_state.pending_changes)} pending rule changes require approval")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💰 Bankroll Rules", "📊 P&L Guards", "💧 Liquidity Rules", "🚦 Rate Limits", "📋 Approvals"
        ])
        
        with tab1:
            self._render_bankroll_limits()
        
        with tab2:
            self._render_pnl_limits()
        
        with tab3:
            self._render_liquidity_limits()
        
        with tab4:
            self._render_rate_limits()
        
        with tab5:
            self._render_approvals_workflow()
    
    def _render_bankroll_limits(self):
        """Render bankroll limits configuration"""
        st.markdown("## 💰 Bankroll & Exposure Limits")
        
        # Current vs Proposed comparison
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Current Limits")
            
            current_limits = self._get_current_bankroll_limits()
            
            # Display current limits as metrics
            st.metric("Total Bankroll", f"£{current_limits['total_bankroll']:,.0f}")
            st.metric("Max Exposure", f"{current_limits['max_exposure_pct']:.1f}%")
            st.metric("Per Match Limit", f"{current_limits['per_match_pct']:.1f}%")
            st.metric("Per Market Limit", f"{current_limits['per_market_pct']:.1f}%")
            st.metric("Per Bet Limit", f"{current_limits['per_bet_pct']:.1f}%")
            
            # Current utilization
            st.markdown("#### 📈 Current Utilization")
            utilization_data = self._get_bankroll_utilization()
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = utilization_data['current_utilization_pct'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Bankroll Utilization %"},
                delta = {'reference': utilization_data['target_utilization_pct']},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#c8712d"},
                    'steps': [
                        {'range': [0, 50], 'color': "#d4edda"},
                        {'range': [50, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ Proposed Changes")
            
            # Editable limits
            with st.form("bankroll_limits_form"):
                st.markdown("#### Adjust Limits")
                
                new_max_exposure = st.slider(
                    "Max Bankroll Exposure (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=current_limits['max_exposure_pct'],
                    step=0.1,
                    help="Maximum percentage of total bankroll that can be at risk"
                )
                
                new_per_match = st.slider(
                    "Per Match Limit (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=current_limits['per_match_pct'],
                    step=0.1,
                    help="Maximum exposure per single match"
                )
                
                new_per_market = st.slider(
                    "Per Market Limit (%)",
                    min_value=0.1,
                    max_value=5.0,
                    value=current_limits['per_market_pct'],
                    step=0.1,
                    help="Maximum exposure per market type"
                )
                
                new_per_bet = st.slider(
                    "Per Bet Limit (%)",
                    min_value=0.1,
                    max_value=2.0,
                    value=current_limits['per_bet_pct'],
                    step=0.05,
                    help="Maximum single bet size as % of bankroll"
                )
                
                # Change justification
                change_reason = st.text_area(
                    "Justification for Changes",
                    placeholder="Explain why these changes are necessary...",
                    help="Required for audit trail and approval process"
                )
                
                # Impact analysis
                st.markdown("#### 📊 Impact Analysis")
                
                # Calculate impacts
                impact_analysis = self._calculate_bankroll_impact(
                    current_limits, {
                        'max_exposure_pct': new_max_exposure,
                        'per_match_pct': new_per_match,
                        'per_market_pct': new_per_market,
                        'per_bet_pct': new_per_bet
                    }
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Exposure Change", f"{impact_analysis['exposure_change']:+.1f}%")
                with col2:
                    st.metric("Risk Level", impact_analysis['risk_level'])
                with col3:
                    st.metric("Affected Bets", f"{impact_analysis['affected_bets']}")
                
                # Submit changes
                submitted = st.form_submit_button("📤 Submit for Approval", type="primary")
                
                if submitted:
                    if change_reason.strip():
                        self._submit_bankroll_changes({
                            'max_exposure_pct': new_max_exposure,
                            'per_match_pct': new_per_match,
                            'per_market_pct': new_per_market,
                            'per_bet_pct': new_per_bet,
                            'reason': change_reason,
                            'impact': impact_analysis
                        })
                        st.success("✅ Bankroll limit changes submitted for approval!")
                    else:
                        st.error("❌ Please provide justification for the changes")
        
        # Historical limits chart
        st.markdown("### 📈 Historical Limits")
        
        historical_data = self._get_bankroll_history()
        if historical_data is not None and not historical_data.empty:
            fig = px.line(
                historical_data,
                x='date',
                y=['max_exposure_pct', 'per_match_pct', 'per_market_pct'],
                title='Bankroll Limits History (Last 30 Days)',
                labels={'value': 'Limit (%)', 'date': 'Date'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_pnl_limits(self):
        """Render P&L limits configuration"""
        st.markdown("## 📊 P&L Protection Guards")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Current P&L Status")
            
            pnl_status = self._get_current_pnl_status()
            
            # P&L metrics
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Daily P&L", f"£{pnl_status['daily_pnl']:,.0f}", f"{pnl_status['daily_pnl_pct']:+.2f}%")
                st.metric("Session P&L", f"£{pnl_status['session_pnl']:,.0f}", f"{pnl_status['session_pnl_pct']:+.2f}%")
            
            with col1b:
                st.metric("Weekly P&L", f"£{pnl_status['weekly_pnl']:,.0f}", f"{pnl_status['weekly_pnl_pct']:+.2f}%")
                st.metric("Monthly P&L", f"£{pnl_status['monthly_pnl']:,.0f}", f"{pnl_status['monthly_pnl_pct']:+.2f}%")
            
            # P&L trend chart
            pnl_trend = self._get_pnl_trend()
            fig = px.line(
                pnl_trend,
                x='timestamp',
                y='cumulative_pnl',
                title='P&L Trend (Last 7 Days)',
                color_discrete_sequence=['#c8712d']
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ P&L Guard Configuration")
            
            current_pnl_limits = self._get_current_pnl_limits()
            
            with st.form("pnl_limits_form"):
                st.markdown("#### Loss Limit Settings")
                
                new_daily_loss = st.slider(
                    "Daily Loss Limit (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=current_pnl_limits['daily_loss_limit_pct'],
                    step=0.1,
                    help="Maximum daily loss as % of bankroll"
                )
                
                new_session_loss = st.slider(
                    "Session Loss Limit (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=current_pnl_limits['session_loss_limit_pct'],
                    step=0.1,
                    help="Maximum session loss as % of bankroll"
                )
                
                # Advanced settings
                st.markdown("#### Advanced P&L Settings")
                
                enable_profit_taking = st.checkbox(
                    "Enable Profit Taking Rules",
                    value=current_pnl_limits.get('profit_taking_enabled', False),
                    help="Automatically reduce position sizes after significant profits"
                )
                
                if enable_profit_taking:
                    profit_threshold = st.slider(
                        "Profit Taking Threshold (%)",
                        min_value=5.0,
                        max_value=50.0,
                        value=current_pnl_limits.get('profit_threshold_pct', 20.0),
                        step=1.0
                    )
                
                enable_drawdown_protection = st.checkbox(
                    "Enable Drawdown Protection",
                    value=current_pnl_limits.get('drawdown_protection_enabled', False),
                    help="Reduce limits after significant drawdowns"
                )
                
                # Change justification
                pnl_change_reason = st.text_area(
                    "Justification for P&L Changes",
                    placeholder="Explain the rationale for these P&L limit changes...",
                    help="Required for compliance and approval"
                )
                
                # Impact preview
                st.markdown("#### 📊 Impact Preview")
                
                current_daily_limit = pnl_status['bankroll'] * (current_pnl_limits['daily_loss_limit_pct'] / 100)
                new_daily_limit = pnl_status['bankroll'] * (new_daily_loss / 100)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Daily Limit", f"£{current_daily_limit:,.0f}")
                with col2:
                    st.metric("New Daily Limit", f"£{new_daily_limit:,.0f}", f"£{new_daily_limit - current_daily_limit:+,.0f}")
                
                # Submit P&L changes
                submitted_pnl = st.form_submit_button("📤 Submit P&L Changes", type="primary")
                
                if submitted_pnl:
                    if pnl_change_reason.strip():
                        self._submit_pnl_changes({
                            'daily_loss_limit_pct': new_daily_loss,
                            'session_loss_limit_pct': new_session_loss,
                            'profit_taking_enabled': enable_profit_taking,
                            'profit_threshold_pct': profit_threshold if enable_profit_taking else None,
                            'drawdown_protection_enabled': enable_drawdown_protection,
                            'reason': pnl_change_reason
                        })
                        st.success("✅ P&L limit changes submitted for approval!")
                    else:
                        st.error("❌ Please provide justification for the changes")
        
        # P&L limit utilization
        st.markdown("### 📊 Limit Utilization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            daily_utilization = abs(pnl_status['daily_pnl_pct']) / current_pnl_limits['daily_loss_limit_pct'] * 100
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = daily_utilization,
                title = {'text': "Daily Loss Limit Usage %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#dc3545" if daily_utilization > 80 else "#ffc107" if daily_utilization > 60 else "#28a745"},
                    'steps': [
                        {'range': [0, 60], 'color': "#d4edda"},
                        {'range': [60, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            session_utilization = abs(pnl_status['session_pnl_pct']) / current_pnl_limits['session_loss_limit_pct'] * 100
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = session_utilization,
                title = {'text': "Session Loss Limit Usage %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#dc3545" if session_utilization > 80 else "#ffc107" if session_utilization > 60 else "#28a745"},
                    'steps': [
                        {'range': [0, 60], 'color': "#d4edda"},
                        {'range': [60, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # P&L statistics
            st.markdown("#### 📈 P&L Statistics")
            st.metric("Win Rate", "64.2%")
            st.metric("Avg Win", "£245")
            st.metric("Avg Loss", "£-189")
            st.metric("Profit Factor", "1.34")
            st.metric("Sharpe Ratio", "1.18")
    
    def _render_liquidity_limits(self):
        """Render liquidity limits configuration"""
        st.markdown("## 💧 Liquidity & Execution Limits")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Current Market Conditions")
            
            market_conditions = self._get_market_conditions()
            
            # Market liquidity overview
            st.metric("Average Market Liquidity", f"£{market_conditions['avg_liquidity']:,.0f}")
            st.metric("Liquidity Volatility", f"{market_conditions['liquidity_volatility']:.1f}%")
            st.metric("Average Spread", f"{market_conditions['avg_spread_bps']:.0f} bps")
            
            # Liquidity by market type
            liquidity_by_market = market_conditions['liquidity_by_market']
            fig = px.bar(
                x=list(liquidity_by_market.keys()),
                y=list(liquidity_by_market.values()),
                title="Average Liquidity by Market Type",
                labels={'x': 'Market Type', 'y': 'Liquidity (£)'},
                color_discrete_sequence=['#c8712d']
            )
            fig.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ Liquidity Rule Configuration")
            
            current_liquidity_limits = self._get_current_liquidity_limits()
            
            with st.form("liquidity_limits_form"):
                st.markdown("#### Odds Constraints")
                
                new_min_odds = st.number_input(
                    "Minimum Odds",
                    min_value=1.01,
                    max_value=2.0,
                    value=current_liquidity_limits['min_odds_threshold'],
                    step=0.01,
                    help="Minimum acceptable odds for any bet"
                )
                
                new_max_odds = st.number_input(
                    "Maximum Odds",
                    min_value=5.0,
                    max_value=100.0,
                    value=current_liquidity_limits['max_odds_threshold'],
                    step=0.5,
                    help="Maximum acceptable odds for any bet"
                )
                
                st.markdown("#### Execution Constraints")
                
                new_slippage_limit = st.number_input(
                    "Slippage Limit (bps)",
                    min_value=10,
                    max_value=500,
                    value=current_liquidity_limits['slippage_bps_limit'],
                    step=5,
                    help="Maximum acceptable slippage in basis points"
                )
                
                new_liquidity_fraction = st.slider(
                    "Max Liquidity Fraction (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=current_liquidity_limits['max_fraction_of_available_liquidity'],
                    step=1.0,
                    help="Maximum fraction of available liquidity to consume"
                )
                
                # Advanced liquidity settings
                st.markdown("#### Advanced Settings")
                
                enable_dynamic_limits = st.checkbox(
                    "Enable Dynamic Liquidity Limits",
                    value=current_liquidity_limits.get('dynamic_limits_enabled', False),
                    help="Adjust limits based on market conditions"
                )
                
                if enable_dynamic_limits:
                    volatility_adjustment = st.slider(
                        "Volatility Adjustment Factor",
                        min_value=0.5,
                        max_value=2.0,
                        value=current_liquidity_limits.get('volatility_factor', 1.0),
                        step=0.1
                    )
                
                # Change justification
                liquidity_change_reason = st.text_area(
                    "Justification for Liquidity Changes",
                    placeholder="Explain the rationale for these liquidity limit changes...",
                    help="Required for compliance and approval"
                )
                
                # Submit liquidity changes
                submitted_liquidity = st.form_submit_button("📤 Submit Liquidity Changes", type="primary")
                
                if submitted_liquidity:
                    if liquidity_change_reason.strip():
                        self._submit_liquidity_changes({
                            'min_odds_threshold': new_min_odds,
                            'max_odds_threshold': new_max_odds,
                            'slippage_bps_limit': new_slippage_limit,
                            'max_fraction_of_available_liquidity': new_liquidity_fraction,
                            'dynamic_limits_enabled': enable_dynamic_limits,
                            'volatility_factor': volatility_adjustment if enable_dynamic_limits else None,
                            'reason': liquidity_change_reason
                        })
                        st.success("✅ Liquidity limit changes submitted for approval!")
                    else:
                        st.error("❌ Please provide justification for the changes")
        
        # Liquidity impact analysis
        st.markdown("### 📊 Liquidity Impact Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Recent Slippage Events")
            slippage_events = self._get_recent_slippage_events()
            
            if slippage_events:
                df = pd.DataFrame(slippage_events)
                fig = px.histogram(
                    df,
                    x='slippage_bps',
                    title='Slippage Distribution (Last 24h)',
                    nbins=20,
                    color_discrete_sequence=['#ffc107']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Market Impact Analysis")
            impact_data = self._get_market_impact_data()
            
            fig = px.scatter(
                impact_data,
                x='bet_size_pct',
                y='market_impact_bps',
                title='Market Impact vs Bet Size',
                labels={'bet_size_pct': 'Bet Size (% of liquidity)', 'market_impact_bps': 'Market Impact (bps)'},
                color_discrete_sequence=['#002466']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("#### Execution Quality Metrics")
            execution_metrics = self._get_execution_metrics()
            
            st.metric("Fill Rate", f"{execution_metrics['fill_rate_pct']:.1f}%")
            st.metric("Avg Slippage", f"{execution_metrics['avg_slippage_bps']:.1f} bps")
            st.metric("Execution Speed", f"{execution_metrics['avg_execution_ms']:.0f} ms")
            st.metric("Reject Rate", f"{execution_metrics['reject_rate_pct']:.2f}%")
    
    def _render_rate_limits(self):
        """Render rate limiting configuration"""
        st.markdown("## 🚦 Rate Limiting Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Current Rate Limit Status")
            
            rate_status = self._get_rate_limit_status()
            
            # Current rate limit metrics
            st.metric("Requests Today", f"{rate_status['requests_today']:,}")
            st.metric("Rate Limited Today", f"{rate_status['rate_limited_today']:,}")
            st.metric("Current Rate Limit %", f"{rate_status['rate_limit_pct']:.2f}%")
            
            # Rate limit timeline
            rate_timeline = self._get_rate_limit_timeline()
            fig = px.line(
                rate_timeline,
                x='hour',
                y=['requests', 'rate_limited'],
                title='Requests vs Rate Limited (Last 24h)',
                labels={'value': 'Count', 'hour': 'Hour of Day'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ Rate Limit Configuration")
            
            current_rate_limits = self._get_current_rate_limits()
            
            with st.form("rate_limits_form"):
                st.markdown("#### Global Rate Limits")
                
                new_global_rate = st.number_input(
                    "Global Rate (requests/second)",
                    min_value=1.0,
                    max_value=100.0,
                    value=current_rate_limits['global_rate'],
                    step=0.5,
                    help="Maximum requests per second across all markets"
                )
                
                new_global_burst = st.number_input(
                    "Global Burst Capacity",
                    min_value=5,
                    max_value=500,
                    value=current_rate_limits['global_burst'],
                    step=5,
                    help="Maximum burst capacity for token bucket"
                )
                
                st.markdown("#### Per-Market Rate Limits")
                
                new_market_requests = st.number_input(
                    "Market Requests",
                    min_value=1,
                    max_value=50,
                    value=current_rate_limits['market_requests'],
                    step=1,
                    help="Maximum requests per market in time window"
                )
                
                new_market_window = st.number_input(
                    "Market Window (seconds)",
                    min_value=30,
                    max_value=600,
                    value=current_rate_limits['market_window_seconds'],
                    step=10,
                    help="Time window for per-market rate limiting"
                )
                
                # Advanced rate limiting
                st.markdown("#### Advanced Settings")
                
                enable_adaptive_limits = st.checkbox(
                    "Enable Adaptive Rate Limits",
                    value=current_rate_limits.get('adaptive_enabled', False),
                    help="Automatically adjust limits based on system load"
                )
                
                if enable_adaptive_limits:
                    load_threshold = st.slider(
                        "Load Threshold (%)",
                        min_value=50,
                        max_value=95,
                        value=current_rate_limits.get('load_threshold_pct', 80),
                        step=5
                    )
                
                enable_priority_queuing = st.checkbox(
                    "Enable Priority Queuing",
                    value=current_rate_limits.get('priority_queuing_enabled', False),
                    help="Prioritize requests based on stake size and confidence"
                )
                
                # Change justification
                rate_change_reason = st.text_area(
                    "Justification for Rate Limit Changes",
                    placeholder="Explain the rationale for these rate limit changes...",
                    help="Required for system stability and approval"
                )
                
                # Submit rate limit changes
                submitted_rate = st.form_submit_button("📤 Submit Rate Limit Changes", type="primary")
                
                if submitted_rate:
                    if rate_change_reason.strip():
                        self._submit_rate_limit_changes({
                            'global_rate': new_global_rate,
                            'global_burst': new_global_burst,
                            'market_requests': new_market_requests,
                            'market_window_seconds': new_market_window,
                            'adaptive_enabled': enable_adaptive_limits,
                            'load_threshold_pct': load_threshold if enable_adaptive_limits else None,
                            'priority_queuing_enabled': enable_priority_queuing,
                            'reason': rate_change_reason
                        })
                        st.success("✅ Rate limit changes submitted for approval!")
                    else:
                        st.error("❌ Please provide justification for the changes")
        
        # Rate limit performance analysis
        st.markdown("### 📊 Rate Limit Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rate limit effectiveness
            effectiveness_data = self._get_rate_limit_effectiveness()
            
            fig = px.pie(
                values=list(effectiveness_data.values()),
                names=list(effectiveness_data.keys()),
                title="Rate Limit Effectiveness",
                color_discrete_map={
                    'Allowed': '#28a745',
                    'Rate Limited': '#dc3545',
                    'Queued': '#ffc107'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # System load correlation
            load_correlation = self._get_load_correlation_data()
            
            fig = px.scatter(
                load_correlation,
                x='system_load_pct',
                y='rate_limit_pct',
                title='Rate Limiting vs System Load',
                labels={'system_load_pct': 'System Load (%)', 'rate_limit_pct': 'Rate Limited (%)'},
                color_discrete_sequence=['#c8712d']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_approvals_workflow(self):
        """Render approvals workflow interface"""
        st.markdown("## 📋 Approvals Workflow")
        
        # Pending approvals summary
        pending_approvals = st.session_state.approval_requests
        
        if pending_approvals:
            st.warning(f"⚠️ {len(pending_approvals)} rule changes pending approval")
            
            # Approval queue
            st.markdown("### 📋 Pending Approvals")
            
            for i, approval in enumerate(pending_approvals):
                with st.expander(f"#{i+1}: {approval['type']} - {approval['submitted_at']}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Change Type:** {approval['type']}")
                        st.markdown(f"**Submitted By:** {approval['submitted_by']}")
                        st.markdown(f"**Submitted At:** {approval['submitted_at']}")
                        st.markdown(f"**Reason:** {approval['reason']}")
                        
                        # Show proposed changes
                        st.markdown("**Proposed Changes:**")
                        for key, value in approval['changes'].items():
                            if key != 'reason':
                                st.markdown(f"- {key}: {value}")
                        
                        # Impact analysis
                        if 'impact' in approval:
                            st.markdown("**Impact Analysis:**")
                            impact = approval['impact']
                            for key, value in impact.items():
                                st.markdown(f"- {key}: {value}")
                    
                    with col2:
                        st.markdown("**Actions:**")
                        
                        col2a, col2b = st.columns(2)
                        
                        with col2a:
                            if st.button(f"✅ Approve", key=f"approve_{i}"):
                                self._approve_change(i)
                                st.success("Change approved!")
                                st.rerun()
                        
                        with col2b:
                            if st.button(f"❌ Reject", key=f"reject_{i}"):
                                self._reject_change(i)
                                st.error("Change rejected!")
                                st.rerun()
                        
                        # Require additional approval for high-risk changes
                        if approval.get('requires_dual_approval', False):
                            st.warning("🔐 Requires dual approval")
                            
                            if st.button(f"🔐 Second Approval", key=f"second_approve_{i}"):
                                st.info("Second approval recorded")
        else:
            st.success("✅ No pending approvals")
        
        # Approval history
        st.markdown("### 📊 Approval History")
        
        approval_history = self._get_approval_history()
        if approval_history:
            df = pd.DataFrame(approval_history)
            
            # Filter controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.selectbox("Status", ["All", "Approved", "Rejected"])
            
            with col2:
                type_filter = st.selectbox("Type", ["All", "Bankroll", "P&L", "Liquidity", "Rate Limit"])
            
            with col3:
                days_filter = st.selectbox("Period", ["7 days", "30 days", "90 days"])
            
            # Apply filters
            filtered_df = df.copy()
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
            if type_filter != "All":
                filtered_df = filtered_df[filtered_df['type'] == type_filter]
            
            # Display filtered history
            st.dataframe(filtered_df, use_container_width=True)
            
            # Approval statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Requests", len(df))
            
            with col2:
                approved_count = len(df[df['status'] == 'Approved'])
                st.metric("Approved", approved_count)
            
            with col3:
                rejected_count = len(df[df['status'] == 'Rejected'])
                st.metric("Rejected", rejected_count)
            
            with col4:
                approval_rate = (approved_count / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # Approval workflow settings
        st.markdown("### ⚙️ Approval Workflow Settings")
        
        with st.expander("Configure Approval Workflow", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Approval Requirements")
                
                require_dual_approval = st.checkbox(
                    "Require Dual Approval for High-Risk Changes",
                    value=True,
                    help="Changes above certain thresholds require two approvals"
                )
                
                dual_approval_threshold = st.slider(
                    "Dual Approval Threshold (% change)",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Percentage change that triggers dual approval requirement"
                )
                
                auto_approve_minor = st.checkbox(
                    "Auto-approve Minor Changes",
                    value=False,
                    help="Automatically approve changes below minor threshold"
                )
            
            with col2:
                st.markdown("#### Notification Settings")
                
                email_notifications = st.checkbox(
                    "Email Notifications",
                    value=True,
                    help="Send email notifications for approval requests"
                )
                
                slack_notifications = st.checkbox(
                    "Slack Notifications",
                    value=True,
                    help="Send Slack notifications for approval requests"
                )
                
                escalation_hours = st.number_input(
                    "Escalation Time (hours)",
                    min_value=1,
                    max_value=72,
                    value=24,
                    help="Hours before escalating unapproved requests"
                )
            
            if st.button("💾 Save Workflow Settings"):
                st.success("Workflow settings saved!")
    
    # Helper methods for data retrieval and processing
    
    def _get_current_bankroll_limits(self) -> Dict[str, float]:
        """Get current bankroll limits (mock)"""
        return {
            'total_bankroll': 100000.0,
            'max_exposure_pct': 5.0,
            'per_match_pct': 2.0,
            'per_market_pct': 1.0,
            'per_bet_pct': 0.5
        }
    
    def _get_bankroll_utilization(self) -> Dict[str, float]:
        """Get bankroll utilization data (mock)"""
        return {
            'current_utilization_pct': 67.3,
            'target_utilization_pct': 70.0,
            'max_utilization_pct': 85.0
        }
    
    def _calculate_bankroll_impact(self, current: Dict, proposed: Dict) -> Dict[str, Any]:
        """Calculate impact of bankroll changes (mock)"""
        exposure_change = proposed['max_exposure_pct'] - current['max_exposure_pct']
        
        if abs(exposure_change) > 2.0:
            risk_level = "HIGH"
        elif abs(exposure_change) > 1.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'exposure_change': exposure_change,
            'risk_level': risk_level,
            'affected_bets': 156 if abs(exposure_change) > 1.0 else 23
        }
    
    def _get_bankroll_history(self) -> pd.DataFrame:
        """Get bankroll limits history (mock)"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'max_exposure_pct': [5.0 + (i % 10) * 0.1 for i in range(len(dates))],
            'per_match_pct': [2.0 + (i % 8) * 0.05 for i in range(len(dates))],
            'per_market_pct': [1.0 + (i % 6) * 0.02 for i in range(len(dates))]
        })
    
    def _submit_bankroll_changes(self, changes: Dict[str, Any]):
        """Submit bankroll changes for approval"""
        approval_request = {
            'type': 'Bankroll Limits',
            'changes': changes,
            'submitted_by': 'current_user',
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'Pending',
            'requires_dual_approval': abs(changes.get('impact', {}).get('exposure_change', 0)) > 1.0,
            'reason': changes['reason'],
            'impact': changes.get('impact', {})
        }
        
        st.session_state.approval_requests.append(approval_request)
    
    def _get_current_pnl_status(self) -> Dict[str, float]:
        """Get current P&L status (mock)"""
        return {
            'daily_pnl': -1250.0,
            'daily_pnl_pct': -1.25,
            'session_pnl': -800.0,
            'session_pnl_pct': -0.80,
            'weekly_pnl': 2340.0,
            'weekly_pnl_pct': 2.34,
            'monthly_pnl': 8750.0,
            'monthly_pnl_pct': 8.75,
            'bankroll': 100000.0
        }
    
    def _get_current_pnl_limits(self) -> Dict[str, float]:
        """Get current P&L limits (mock)"""
        return {
            'daily_loss_limit_pct': 3.0,
            'session_loss_limit_pct': 2.0,
            'profit_taking_enabled': False,
            'profit_threshold_pct': 20.0,
            'drawdown_protection_enabled': False
        }
    
    def _get_pnl_trend(self) -> pd.DataFrame:
        """Get P&L trend data (mock)"""
        import numpy as np
        
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
        cumulative_pnl = np.cumsum(np.random.normal(10, 50, len(timestamps)))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'cumulative_pnl': cumulative_pnl
        })
    
    def _submit_pnl_changes(self, changes: Dict[str, Any]):
        """Submit P&L changes for approval"""
        approval_request = {
            'type': 'P&L Guards',
            'changes': changes,
            'submitted_by': 'current_user',
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'Pending',
            'requires_dual_approval': False,
            'reason': changes['reason']
        }
        
        st.session_state.approval_requests.append(approval_request)
    
    def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions (mock)"""
        return {
            'avg_liquidity': 25000.0,
            'liquidity_volatility': 15.3,
            'avg_spread_bps': 12.5,
            'liquidity_by_market': {
                'Match Winner': 45000,
                'Total Runs': 32000,
                'Top Batsman': 18000,
                'Method of Dismissal': 12000,
                'Innings Runs': 28000
            }
        }
    
    def _get_current_liquidity_limits(self) -> Dict[str, Any]:
        """Get current liquidity limits (mock)"""
        return {
            'min_odds_threshold': 1.25,
            'max_odds_threshold': 10.0,
            'slippage_bps_limit': 50,
            'max_fraction_of_available_liquidity': 10.0,
            'dynamic_limits_enabled': False,
            'volatility_factor': 1.0
        }
    
    def _submit_liquidity_changes(self, changes: Dict[str, Any]):
        """Submit liquidity changes for approval"""
        approval_request = {
            'type': 'Liquidity Rules',
            'changes': changes,
            'submitted_by': 'current_user',
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'Pending',
            'requires_dual_approval': False,
            'reason': changes['reason']
        }
        
        st.session_state.approval_requests.append(approval_request)
    
    def _get_recent_slippage_events(self) -> List[Dict[str, Any]]:
        """Get recent slippage events (mock)"""
        import numpy as np
        
        events = []
        for i in range(100):
            events.append({
                'timestamp': datetime.now() - timedelta(minutes=i*15),
                'slippage_bps': max(0, np.random.normal(25, 15))
            })
        
        return events
    
    def _get_market_impact_data(self) -> pd.DataFrame:
        """Get market impact data (mock)"""
        import numpy as np
        
        bet_sizes = np.random.uniform(1, 30, 100)
        market_impacts = bet_sizes * 0.5 + np.random.normal(0, 2, 100)
        
        return pd.DataFrame({
            'bet_size_pct': bet_sizes,
            'market_impact_bps': np.maximum(0, market_impacts)
        })
    
    def _get_execution_metrics(self) -> Dict[str, float]:
        """Get execution quality metrics (mock)"""
        return {
            'fill_rate_pct': 98.7,
            'avg_slippage_bps': 23.5,
            'avg_execution_ms': 145.0,
            'reject_rate_pct': 1.3
        }
    
    def _get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limit status (mock)"""
        return {
            'requests_today': 15234,
            'rate_limited_today': 187,
            'rate_limit_pct': 1.23
        }
    
    def _get_rate_limit_timeline(self) -> pd.DataFrame:
        """Get rate limit timeline (mock)"""
        hours = list(range(24))
        requests = [500 + (h % 8) * 100 + (h % 3) * 50 for h in hours]
        rate_limited = [max(0, r * 0.02 + (h % 5) - 2) for h, r in zip(hours, requests)]
        
        return pd.DataFrame({
            'hour': hours,
            'requests': requests,
            'rate_limited': rate_limited
        })
    
    def _get_current_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limits (mock)"""
        return {
            'global_rate': 10.0,
            'global_burst': 50,
            'market_requests': 5,
            'market_window_seconds': 120,
            'adaptive_enabled': False,
            'load_threshold_pct': 80,
            'priority_queuing_enabled': False
        }
    
    def _submit_rate_limit_changes(self, changes: Dict[str, Any]):
        """Submit rate limit changes for approval"""
        approval_request = {
            'type': 'Rate Limits',
            'changes': changes,
            'submitted_by': 'current_user',
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'Pending',
            'requires_dual_approval': False,
            'reason': changes['reason']
        }
        
        st.session_state.approval_requests.append(approval_request)
    
    def _get_rate_limit_effectiveness(self) -> Dict[str, int]:
        """Get rate limit effectiveness data (mock)"""
        return {
            'Allowed': 14850,
            'Rate Limited': 187,
            'Queued': 197
        }
    
    def _get_load_correlation_data(self) -> pd.DataFrame:
        """Get load correlation data (mock)"""
        import numpy as np
        
        system_loads = np.random.uniform(20, 95, 100)
        rate_limits = np.maximum(0, (system_loads - 60) * 0.1 + np.random.normal(0, 0.5, 100))
        
        return pd.DataFrame({
            'system_load_pct': system_loads,
            'rate_limit_pct': rate_limits
        })
    
    def _approve_change(self, index: int):
        """Approve a pending change"""
        if 0 <= index < len(st.session_state.approval_requests):
            st.session_state.approval_requests[index]['status'] = 'Approved'
            st.session_state.approval_requests[index]['approved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.approval_requests[index]['approved_by'] = 'current_approver'
            
            # Move to history (in real implementation, would save to database)
            approved_request = st.session_state.approval_requests.pop(index)
    
    def _reject_change(self, index: int):
        """Reject a pending change"""
        if 0 <= index < len(st.session_state.approval_requests):
            st.session_state.approval_requests[index]['status'] = 'Rejected'
            st.session_state.approval_requests[index]['rejected_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.approval_requests[index]['rejected_by'] = 'current_approver'
            
            # Move to history (in real implementation, would save to database)
            rejected_request = st.session_state.approval_requests.pop(index)
    
    def _get_approval_history(self) -> List[Dict[str, Any]]:
        """Get approval history (mock)"""
        history = []
        
        for i in range(20):
            history.append({
                'timestamp': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'type': ['Bankroll', 'P&L', 'Liquidity', 'Rate Limit'][i % 4],
                'status': ['Approved', 'Rejected'][i % 7 == 0],  # Mostly approved
                'submitted_by': f'user_{i % 5}',
                'approved_by': f'approver_{i % 3}' if i % 7 != 0 else None,
                'description': f'Rule change #{i+1}'
            })
        
        return history


def main():
    """Main entry point for limits manager"""
    limits_manager = LimitsManager()
    limits_manager.render()


if __name__ == "__main__":
    main()
