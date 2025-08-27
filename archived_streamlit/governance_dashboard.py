# Purpose: Main governance dashboard for DGL Streamlit interface
# Author: WicketWise AI, Last Modified: 2024

"""
Governance Dashboard

Main Streamlit dashboard for DGL governance and limits management:
- Real-time system status and metrics
- Governance decision monitoring
- Rule configuration interface
- Performance analytics
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
from schemas import DecisionType, RuleId


logger = logging.getLogger(__name__)


class GovernanceDashboard:
    """
    Main governance dashboard for DGL system monitoring and control
    
    Provides comprehensive interface for:
    - System health monitoring
    - Decision analytics
    - Rule management
    - Performance metrics
    """
    
    def __init__(self, dgl_base_url: str = "http://localhost:8001"):
        """
        Initialize governance dashboard
        
        Args:
            dgl_base_url: DGL service base URL
        """
        self.dgl_base_url = dgl_base_url
        self.client_config = DGLClientConfig(base_url=dgl_base_url)
        
        # Initialize session state
        if 'dashboard_data' not in st.session_state:
            st.session_state.dashboard_data = {}
        
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
    
    def render(self):
        """Render the main governance dashboard"""
        st.set_page_config(
            page_title="WicketWise DGL - Governance Dashboard",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for cricket theme
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #c8712d 0%, #002466 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #c8712d;
            margin-bottom: 1rem;
        }
        .status-healthy { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .cricket-accent { color: #c8712d; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown("""
        <div class="main-header">
            <h1>🛡️ WicketWise Deterministic Governance Layer</h1>
            <p>AI-Independent Safety Engine for Cricket Betting Governance</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "⚖️ Decisions", "🔧 Rules", "📈 Analytics", "🔍 Audit"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_decisions_tab()
        
        with tab3:
            self._render_rules_tab()
        
        with tab4:
            self._render_analytics_tab()
        
        with tab5:
            self._render_audit_tab()
    
    def _render_sidebar(self):
        """Render sidebar with controls and status"""
        st.sidebar.markdown("## 🎛️ Control Panel")
        
        # System status
        st.sidebar.markdown("### System Status")
        
        try:
            # Get system health (mock for now)
            health_status = self._get_system_health()
            
            if health_status.get("status") == "healthy":
                st.sidebar.success("🟢 System Healthy")
            elif health_status.get("status") == "degraded":
                st.sidebar.warning("🟡 System Degraded")
            else:
                st.sidebar.error("🔴 System Unhealthy")
            
            # Key metrics in sidebar
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Uptime", health_status.get("uptime", "Unknown"))
            with col2:
                st.metric("Mode", health_status.get("mode", "SHADOW"))
            
        except Exception as e:
            st.sidebar.error(f"❌ Connection Error: {str(e)}")
        
        # Refresh controls
        st.sidebar.markdown("### 🔄 Refresh")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh Now"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        if auto_refresh:
            st.sidebar.info("⏱️ Auto-refresh enabled (30s)")
            # Note: In production, implement proper auto-refresh
        
        # Quick actions
        st.sidebar.markdown("### ⚡ Quick Actions")
        
        if st.sidebar.button("🚨 Emergency Stop"):
            st.sidebar.error("Emergency stop would be triggered!")
        
        if st.sidebar.button("📊 Export Report"):
            st.sidebar.success("Report export initiated!")
        
        # Connection settings
        st.sidebar.markdown("### 🔗 Connection")
        new_url = st.sidebar.text_input("DGL Service URL", value=self.dgl_base_url)
        if new_url != self.dgl_base_url:
            self.dgl_base_url = new_url
            self.client_config = DGLClientConfig(base_url=new_url)
    
    def _render_overview_tab(self):
        """Render system overview tab"""
        st.markdown("## 📊 System Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            # Get current metrics (mock data for now)
            metrics = self._get_overview_metrics()
            
            with col1:
                st.metric(
                    "Total Decisions Today",
                    metrics.get("total_decisions", 0),
                    delta=metrics.get("decisions_delta", 0)
                )
            
            with col2:
                approval_rate = metrics.get("approval_rate_pct", 0)
                st.metric(
                    "Approval Rate",
                    f"{approval_rate:.1f}%",
                    delta=f"{metrics.get('approval_delta', 0):.1f}%"
                )
            
            with col3:
                avg_time = metrics.get("avg_processing_time_ms", 0)
                st.metric(
                    "Avg Response Time",
                    f"{avg_time:.1f}ms",
                    delta=f"{metrics.get('time_delta', 0):.1f}ms"
                )
            
            with col4:
                error_rate = metrics.get("error_rate_pct", 0)
                st.metric(
                    "Error Rate",
                    f"{error_rate:.2f}%",
                    delta=f"{metrics.get('error_delta', 0):.2f}%"
                )
        
        except Exception as e:
            st.error(f"Failed to load metrics: {str(e)}")
        
        # System health details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🏥 System Health")
            
            # Health status chart
            health_data = self._get_health_timeline()
            if health_data is not None and not health_data.empty:
                fig = px.line(
                    health_data,
                    x='timestamp',
                    y='response_time_ms',
                    title='Response Time Trend (Last 24h)',
                    color_discrete_sequence=['#c8712d']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Current Limits")
            
            # Current rule limits
            limits = self._get_current_limits()
            
            st.markdown(f"""
            **Bankroll Limits:**
            - Max Exposure: {limits.get('max_exposure_pct', 5.0)}%
            - Per Match: {limits.get('per_match_pct', 2.0)}%
            - Per Market: {limits.get('per_market_pct', 1.0)}%
            
            **P&L Guards:**
            - Daily Loss: {limits.get('daily_loss_pct', 3.0)}%
            - Session Loss: {limits.get('session_loss_pct', 2.0)}%
            
            **Liquidity:**
            - Min Odds: {limits.get('min_odds', 1.25)}
            - Max Odds: {limits.get('max_odds', 10.0)}
            - Slippage: {limits.get('slippage_bps', 50)}bps
            """)
        
        # Recent activity
        st.markdown("### 📋 Recent Activity")
        
        recent_decisions = self._get_recent_decisions(limit=10)
        if recent_decisions:
            df = pd.DataFrame(recent_decisions)
            
            # Style the dataframe
            def style_decision(val):
                if val == "APPROVE":
                    return "background-color: #d4edda; color: #155724"
                elif val == "REJECT":
                    return "background-color: #f8d7da; color: #721c24"
                else:
                    return "background-color: #fff3cd; color: #856404"
            
            styled_df = df.style.applymap(style_decision, subset=['decision'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No recent decisions available")
    
    def _render_decisions_tab(self):
        """Render decisions monitoring tab"""
        st.markdown("## ⚖️ Decision Monitoring")
        
        # Decision filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            time_filter = st.selectbox(
                "Time Period",
                ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
                index=2
            )
        
        with col2:
            decision_filter = st.selectbox(
                "Decision Type",
                ["All", "APPROVE", "REJECT", "AMEND"]
            )
        
        with col3:
            market_filter = st.text_input("Market ID Filter", placeholder="e.g., match_001")
        
        with col4:
            if st.button("🔍 Apply Filters"):
                st.success("Filters applied!")
        
        # Decision analytics
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📊 Decision Distribution")
            
            # Decision pie chart
            decision_data = self._get_decision_distribution()
            if decision_data:
                fig = px.pie(
                    values=list(decision_data.values()),
                    names=list(decision_data.keys()),
                    title="Decision Distribution",
                    color_discrete_map={
                        'APPROVE': '#28a745',
                        'REJECT': '#dc3545', 
                        'AMEND': '#ffc107'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🚨 Rule Violations")
            
            violations = self._get_rule_violations()
            for rule, count in violations.items():
                st.markdown(f"**{rule}:** {count}")
        
        # Decision timeline
        st.markdown("### ⏰ Decision Timeline")
        
        timeline_data = self._get_decision_timeline()
        if timeline_data is not None and not timeline_data.empty:
            fig = px.bar(
                timeline_data,
                x='hour',
                y='count',
                color='decision',
                title='Decisions by Hour',
                color_discrete_map={
                    'APPROVE': '#28a745',
                    'REJECT': '#dc3545',
                    'AMEND': '#ffc107'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed decision log
        st.markdown("### 📝 Decision Log")
        
        with st.expander("View Detailed Decisions", expanded=False):
            detailed_decisions = self._get_detailed_decisions(limit=50)
            if detailed_decisions:
                df = pd.DataFrame(detailed_decisions)
                st.dataframe(df, use_container_width=True)
    
    def _render_rules_tab(self):
        """Render rules configuration tab"""
        st.markdown("## 🔧 Rules Configuration")
        
        # Warning about rule changes
        st.warning("⚠️ Rule changes require appropriate authorization and may affect live trading.")
        
        # Rule categories
        rule_tab1, rule_tab2, rule_tab3, rule_tab4 = st.tabs([
            "💰 Bankroll", "📊 P&L Guards", "💧 Liquidity", "🚦 Rate Limits"
        ])
        
        with rule_tab1:
            self._render_bankroll_rules()
        
        with rule_tab2:
            self._render_pnl_rules()
        
        with rule_tab3:
            self._render_liquidity_rules()
        
        with rule_tab4:
            self._render_rate_limit_rules()
    
    def _render_bankroll_rules(self):
        """Render bankroll rules configuration"""
        st.markdown("### 💰 Bankroll & Exposure Rules")
        
        try:
            current_config = self._get_current_rules_config()
            bankroll_config = current_config.get("bankroll_config", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Current Settings")
                
                total_bankroll = st.number_input(
                    "Total Bankroll (£)",
                    value=bankroll_config.get("total_bankroll", 100000.0),
                    min_value=1000.0,
                    max_value=10000000.0,
                    step=1000.0,
                    disabled=True  # Read-only for safety
                )
                
                max_exposure = st.slider(
                    "Max Bankroll Exposure (%)",
                    min_value=0.1,
                    max_value=20.0,
                    value=bankroll_config.get("max_bankroll_exposure_pct", 5.0),
                    step=0.1,
                    disabled=True
                )
                
                per_match = st.slider(
                    "Per Match Max (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=bankroll_config.get("per_match_max_pct", 2.0),
                    step=0.1,
                    disabled=True
                )
            
            with col2:
                st.markdown("#### Proposed Changes")
                
                new_max_exposure = st.slider(
                    "New Max Bankroll Exposure (%)",
                    min_value=0.1,
                    max_value=20.0,
                    value=bankroll_config.get("max_bankroll_exposure_pct", 5.0),
                    step=0.1,
                    key="new_max_exposure"
                )
                
                new_per_match = st.slider(
                    "New Per Match Max (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=bankroll_config.get("per_match_max_pct", 2.0),
                    step=0.1,
                    key="new_per_match"
                )
                
                new_per_market = st.slider(
                    "New Per Market Max (%)",
                    min_value=0.1,
                    max_value=5.0,
                    value=bankroll_config.get("per_market_max_pct", 1.0),
                    step=0.1,
                    key="new_per_market"
                )
                
                if st.button("💾 Save Bankroll Rules", type="primary"):
                    st.success("Bankroll rules update requested (requires approval)")
            
            # Impact analysis
            st.markdown("#### 📊 Impact Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_exposure = 15000.0  # Mock current exposure
                current_utilization = (current_exposure / total_bankroll) * 100
                st.metric("Current Utilization", f"{current_utilization:.1f}%")
            
            with col2:
                new_max_amount = total_bankroll * (new_max_exposure / 100)
                st.metric("New Max Exposure Amount", f"£{new_max_amount:,.0f}")
            
            with col3:
                impact = "Tighter" if new_max_exposure < max_exposure else "Looser"
                st.metric("Rule Impact", impact)
        
        except Exception as e:
            st.error(f"Failed to load bankroll rules: {str(e)}")
    
    def _render_pnl_rules(self):
        """Render P&L rules configuration"""
        st.markdown("### 📊 P&L Protection Guards")
        
        try:
            current_config = self._get_current_rules_config()
            pnl_config = current_config.get("pnl_config", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Daily Loss Limits")
                
                current_daily = pnl_config.get("daily_loss_limit_pct", 3.0)
                new_daily = st.slider(
                    "Daily Loss Limit (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=current_daily,
                    step=0.1
                )
                
                st.markdown("#### Session Loss Limits")
                
                current_session = pnl_config.get("session_loss_limit_pct", 2.0)
                new_session = st.slider(
                    "Session Loss Limit (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=current_session,
                    step=0.1
                )
            
            with col2:
                st.markdown("#### Current P&L Status")
                
                # Mock P&L data
                daily_pnl = -1250.0
                session_pnl = -800.0
                bankroll = 100000.0
                
                daily_pct = (daily_pnl / bankroll) * 100
                session_pct = (session_pnl / bankroll) * 100
                
                st.metric("Daily P&L", f"£{daily_pnl:,.0f}", f"{daily_pct:.2f}%")
                st.metric("Session P&L", f"£{session_pnl:,.0f}", f"{session_pct:.2f}%")
                
                # Progress bars for limits
                daily_progress = abs(daily_pct) / current_daily
                session_progress = abs(session_pct) / current_session
                
                st.progress(daily_progress, text=f"Daily Limit Usage: {daily_progress*100:.1f}%")
                st.progress(session_progress, text=f"Session Limit Usage: {session_progress*100:.1f}%")
            
            if st.button("💾 Save P&L Rules", type="primary"):
                st.success("P&L rules update requested (requires approval)")
        
        except Exception as e:
            st.error(f"Failed to load P&L rules: {str(e)}")
    
    def _render_liquidity_rules(self):
        """Render liquidity rules configuration"""
        st.markdown("### 💧 Liquidity & Execution Rules")
        
        try:
            current_config = self._get_current_rules_config()
            liquidity_config = current_config.get("liquidity_config", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Odds Constraints")
                
                min_odds = st.number_input(
                    "Minimum Odds",
                    min_value=1.01,
                    max_value=2.0,
                    value=liquidity_config.get("min_odds_threshold", 1.25),
                    step=0.01
                )
                
                max_odds = st.number_input(
                    "Maximum Odds",
                    min_value=5.0,
                    max_value=100.0,
                    value=liquidity_config.get("max_odds_threshold", 10.0),
                    step=0.5
                )
                
                st.markdown("#### Execution Constraints")
                
                slippage_limit = st.number_input(
                    "Slippage Limit (bps)",
                    min_value=10,
                    max_value=500,
                    value=liquidity_config.get("slippage_bps_limit", 50),
                    step=5
                )
            
            with col2:
                st.markdown("#### Liquidity Constraints")
                
                max_fraction = st.slider(
                    "Max Liquidity Fraction (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=liquidity_config.get("max_fraction_of_available_liquidity", 10.0),
                    step=1.0
                )
                
                st.markdown("#### Market Impact Analysis")
                
                # Mock liquidity data
                avg_liquidity = 25000.0
                typical_bet = 1000.0
                impact = (typical_bet / avg_liquidity) * 100
                
                st.metric("Average Market Liquidity", f"£{avg_liquidity:,.0f}")
                st.metric("Typical Bet Size", f"£{typical_bet:,.0f}")
                st.metric("Market Impact", f"{impact:.1f}%")
            
            if st.button("💾 Save Liquidity Rules", type="primary"):
                st.success("Liquidity rules update requested (requires approval)")
        
        except Exception as e:
            st.error(f"Failed to load liquidity rules: {str(e)}")
    
    def _render_rate_limit_rules(self):
        """Render rate limiting rules configuration"""
        st.markdown("### 🚦 Rate Limiting Rules")
        
        try:
            current_config = self._get_current_rules_config()
            rate_config = current_config.get("rate_limit_config", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Current Limits")
                
                current_count = rate_config.get("count", 5)
                current_window = rate_config.get("per_seconds", 120)
                
                st.metric("Requests Allowed", current_count)
                st.metric("Time Window (seconds)", current_window)
                st.metric("Rate", f"{current_count}/{current_window}s")
                
                st.markdown("#### Proposed Changes")
                
                new_count = st.number_input(
                    "New Request Count",
                    min_value=1,
                    max_value=100,
                    value=current_count,
                    step=1
                )
                
                new_window = st.number_input(
                    "New Time Window (seconds)",
                    min_value=10,
                    max_value=3600,
                    value=current_window,
                    step=10
                )
            
            with col2:
                st.markdown("#### Rate Limit Statistics")
                
                # Mock rate limit data
                requests_today = 1250
                rate_limited_today = 15
                rate_limit_pct = (rate_limited_today / requests_today) * 100
                
                st.metric("Requests Today", requests_today)
                st.metric("Rate Limited", rate_limited_today)
                st.metric("Rate Limit %", f"{rate_limit_pct:.2f}%")
                
                # Rate limit timeline
                hours = list(range(24))
                rate_limits = [max(0, 5 + (i % 8) - 3) for i in hours]
                
                fig = px.bar(
                    x=hours,
                    y=rate_limits,
                    title="Rate Limits by Hour",
                    labels={'x': 'Hour', 'y': 'Rate Limited Requests'},
                    color_discrete_sequence=['#dc3545']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("💾 Save Rate Limit Rules", type="primary"):
                st.success("Rate limit rules update requested (requires approval)")
        
        except Exception as e:
            st.error(f"Failed to load rate limit rules: {str(e)}")
    
    def _render_analytics_tab(self):
        """Render analytics tab"""
        st.markdown("## 📈 Performance Analytics")
        
        # Analytics time period
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            period = st.selectbox(
                "Analysis Period",
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days"]
            )
        
        with col2:
            metric_type = st.selectbox(
                "Metric Type",
                ["Response Time", "Decision Rate", "Error Rate", "Rule Violations"]
            )
        
        with col3:
            st.markdown("")  # Spacer
        
        # Performance metrics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Performance Trends")
            
            # Generate mock performance data
            performance_data = self._get_performance_data(period, metric_type)
            
            if performance_data is not None and not performance_data.empty:
                fig = px.line(
                    performance_data,
                    x='timestamp',
                    y='value',
                    title=f'{metric_type} Trend - {period}',
                    color_discrete_sequence=['#c8712d']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Key Metrics")
            
            # Key performance indicators
            kpis = self._get_key_performance_indicators()
            
            st.metric("Avg Response Time", f"{kpis.get('avg_response_ms', 0):.1f}ms")
            st.metric("P99 Response Time", f"{kpis.get('p99_response_ms', 0):.1f}ms")
            st.metric("Uptime", f"{kpis.get('uptime_pct', 0):.2f}%")
            st.metric("Throughput", f"{kpis.get('throughput_rps', 0):.1f} req/s")
        
        # Rule performance analysis
        st.markdown("### 🔍 Rule Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rule violation frequency
            rule_violations = self._get_rule_violation_stats()
            
            if rule_violations:
                fig = px.bar(
                    x=list(rule_violations.keys()),
                    y=list(rule_violations.values()),
                    title="Rule Violations by Type",
                    color_discrete_sequence=['#dc3545']
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Decision confidence distribution
            confidence_data = self._get_confidence_distribution()
            
            if confidence_data is not None and len(confidence_data) > 0:
                fig = px.histogram(
                    x=confidence_data,
                    title="Decision Confidence Distribution",
                    nbins=20,
                    color_discrete_sequence=['#28a745']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_audit_tab(self):
        """Render audit trail tab"""
        st.markdown("## 🔍 Audit Trail")
        
        # Audit search controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=7))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now().date())
        
        with col3:
            event_type = st.selectbox("Event Type", ["All", "Decision", "Rule Change", "System"])
        
        with col4:
            if st.button("🔍 Search Audit Trail"):
                st.success("Audit search initiated!")
        
        # Audit statistics
        col1, col2, col3, col4 = st.columns(4)
        
        audit_stats = self._get_audit_statistics()
        
        with col1:
            st.metric("Total Records", audit_stats.get("total_records", 0))
        
        with col2:
            st.metric("Integrity Score", f"{audit_stats.get('integrity_score', 100):.1f}%")
        
        with col3:
            st.metric("Hash Chain Status", audit_stats.get("hash_status", "INTACT"))
        
        with col4:
            st.metric("Last Verified", audit_stats.get("last_verified", "Just now"))
        
        # Recent audit records
        st.markdown("### 📋 Recent Audit Records")
        
        audit_records = self._get_recent_audit_records(limit=20)
        if audit_records:
            df = pd.DataFrame(audit_records)
            st.dataframe(df, use_container_width=True)
        
        # Compliance report
        st.markdown("### 📊 Compliance Report")
        
        with st.expander("Generate Compliance Report", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                report_period = st.selectbox("Report Period", ["Last 30 Days", "Last Quarter", "Last Year"])
                report_type = st.selectbox("Report Type", ["Summary", "Detailed", "Regulatory"])
            
            with col2:
                if st.button("📄 Generate Report"):
                    st.success("Compliance report generation started!")
                    
                    # Mock report data
                    report_data = {
                        "Period": report_period,
                        "Total Decisions": "15,234",
                        "Compliance Rate": "99.8%",
                        "Violations": "32",
                        "Audit Score": "98.5%"
                    }
                    
                    st.json(report_data)
    
    # Helper methods for data retrieval (mock implementations)
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status (mock)"""
        return {
            "status": "healthy",
            "uptime": "15d 4h 23m",
            "mode": "SHADOW",
            "version": "1.0.0"
        }
    
    def _get_overview_metrics(self) -> Dict[str, Any]:
        """Get overview metrics (mock)"""
        return {
            "total_decisions": 1247,
            "decisions_delta": 156,
            "approval_rate_pct": 67.3,
            "approval_delta": 2.1,
            "avg_processing_time_ms": 23.5,
            "time_delta": -1.2,
            "error_rate_pct": 0.08,
            "error_delta": -0.02
        }
    
    def _get_health_timeline(self) -> pd.DataFrame:
        """Get health timeline data (mock)"""
        import numpy as np
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='h'
        )
        
        response_times = 20 + 10 * np.random.random(len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'response_time_ms': response_times
        })
    
    def _get_current_limits(self) -> Dict[str, float]:
        """Get current rule limits (mock)"""
        return {
            "max_exposure_pct": 5.0,
            "per_match_pct": 2.0,
            "per_market_pct": 1.0,
            "daily_loss_pct": 3.0,
            "session_loss_pct": 2.0,
            "min_odds": 1.25,
            "max_odds": 10.0,
            "slippage_bps": 50
        }
    
    def _get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decisions (mock)"""
        decisions = []
        for i in range(limit):
            decisions.append({
                "timestamp": (datetime.now() - timedelta(minutes=i*5)).strftime("%H:%M:%S"),
                "market_id": f"market_{i % 3 + 1}",
                "decision": ["APPROVE", "REJECT", "AMEND"][i % 3],
                "stake": f"£{1000 + i*100:,}",
                "processing_time": f"{20 + i*2}ms"
            })
        return decisions
    
    def _get_decision_distribution(self) -> Dict[str, int]:
        """Get decision distribution (mock)"""
        return {
            "APPROVE": 842,
            "REJECT": 312,
            "AMEND": 93
        }
    
    def _get_rule_violations(self) -> Dict[str, int]:
        """Get rule violations (mock)"""
        return {
            "Bankroll Limit": 45,
            "Liquidity Limit": 23,
            "P&L Guard": 12,
            "Rate Limit": 8
        }
    
    def _get_decision_timeline(self) -> pd.DataFrame:
        """Get decision timeline (mock)"""
        hours = list(range(24))
        data = []
        
        for hour in hours:
            for decision in ["APPROVE", "REJECT", "AMEND"]:
                count = max(0, 50 + (hour % 8) * 10 + {"APPROVE": 30, "REJECT": -10, "AMEND": -20}[decision])
                data.append({"hour": hour, "decision": decision, "count": count})
        
        return pd.DataFrame(data)
    
    def _get_detailed_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get detailed decisions (mock)"""
        decisions = []
        for i in range(limit):
            decisions.append({
                "timestamp": (datetime.now() - timedelta(minutes=i*3)).isoformat(),
                "proposal_id": f"prop_{i:04d}",
                "market_id": f"market_{i % 5 + 1}",
                "decision": ["APPROVE", "REJECT", "AMEND"][i % 3],
                "confidence": f"{0.7 + (i % 30) * 0.01:.2f}",
                "processing_time_ms": 20 + (i % 50),
                "rules_triggered": i % 4
            })
        return decisions
    
    def _get_current_rules_config(self) -> Dict[str, Any]:
        """Get current rules configuration (mock)"""
        return {
            "bankroll_config": {
                "total_bankroll": 100000.0,
                "max_bankroll_exposure_pct": 5.0,
                "per_match_max_pct": 2.0,
                "per_market_max_pct": 1.0,
                "per_bet_max_pct": 0.5
            },
            "pnl_config": {
                "daily_loss_limit_pct": 3.0,
                "session_loss_limit_pct": 2.0
            },
            "liquidity_config": {
                "min_odds_threshold": 1.25,
                "max_odds_threshold": 10.0,
                "slippage_bps_limit": 50,
                "max_fraction_of_available_liquidity": 10.0
            },
            "rate_limit_config": {
                "count": 5,
                "per_seconds": 120
            }
        }
    
    def _get_performance_data(self, period: str, metric_type: str) -> pd.DataFrame:
        """Get performance data (mock)"""
        import numpy as np
        
        if period == "Last 24 Hours":
            timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
        elif period == "Last 7 Days":
            timestamps = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='6H')
        else:
            timestamps = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        if metric_type == "Response Time":
            values = 20 + 10 * np.random.random(len(timestamps))
        elif metric_type == "Decision Rate":
            values = 50 + 20 * np.random.random(len(timestamps))
        elif metric_type == "Error Rate":
            values = 0.1 + 0.2 * np.random.random(len(timestamps))
        else:
            values = 5 + 10 * np.random.random(len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
    
    def _get_key_performance_indicators(self) -> Dict[str, float]:
        """Get KPIs (mock)"""
        return {
            "avg_response_ms": 23.5,
            "p99_response_ms": 87.2,
            "uptime_pct": 99.97,
            "throughput_rps": 12.3
        }
    
    def _get_rule_violation_stats(self) -> Dict[str, int]:
        """Get rule violation statistics (mock)"""
        return {
            "BANKROLL_MAX": 45,
            "LIQ_SLIPPAGE": 32,
            "PNL_DAILY": 18,
            "RATE_LIMIT": 12,
            "LIQ_FRACTION": 8
        }
    
    def _get_confidence_distribution(self) -> List[float]:
        """Get confidence distribution (mock)"""
        import numpy as np
        return np.random.beta(8, 2, 1000).tolist()
    
    def _get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit statistics (mock)"""
        return {
            "total_records": 15234,
            "integrity_score": 99.8,
            "hash_status": "INTACT",
            "last_verified": "2 min ago"
        }
    
    def _get_recent_audit_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent audit records (mock)"""
        records = []
        for i in range(limit):
            records.append({
                "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                "event_type": ["Decision", "Rule Change", "System"][i % 3],
                "record_id": f"audit_{i:06d}",
                "user": "system" if i % 3 == 2 else f"user_{i % 5}",
                "description": f"Governance decision #{i} processed",
                "integrity": "✓"
            })
        return records


def main():
    """Main entry point for Streamlit app"""
    dashboard = GovernanceDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
