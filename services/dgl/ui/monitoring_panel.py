# Purpose: Real-time monitoring panel for DGL system health
# Author: WicketWise AI, Last Modified: 2024

"""
Monitoring Panel

Real-time system monitoring interface for DGL:
- Live system metrics and performance
- Alert management and notifications
- Resource utilization tracking
- Performance optimization insights
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
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.dgl_client import DGLClient, DGLClientConfig


logger = logging.getLogger(__name__)


class MonitoringPanel:
    """
    Real-time monitoring panel for DGL system health and performance
    
    Provides interface for:
    - Live system metrics and dashboards
    - Performance monitoring and alerts
    - Resource utilization tracking
    - Optimization recommendations
    """
    
    def __init__(self, dgl_base_url: str = "http://localhost:8001"):
        """
        Initialize monitoring panel
        
        Args:
            dgl_base_url: DGL service base URL
        """
        self.dgl_base_url = dgl_base_url
        self.client_config = DGLClientConfig(base_url=dgl_base_url)
        
        # Initialize session state for monitoring data
        if 'monitoring_data' not in st.session_state:
            st.session_state.monitoring_data = {}
        
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
    
    def render(self):
        """Render the monitoring panel interface"""
        
        st.markdown("# 📈 System Monitoring Panel")
        st.markdown("Real-time monitoring and performance analytics for the WicketWise DGL system.")
        
        # System status header
        self._render_system_status_header()
        
        # Main monitoring tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Metrics", "🚨 Alerts", "💻 Resources", "⚡ Performance", "🔧 Optimization"
        ])
        
        with tab1:
            self._render_live_metrics_tab()
        
        with tab2:
            self._render_alerts_tab()
        
        with tab3:
            self._render_resources_tab()
        
        with tab4:
            self._render_performance_tab()
        
        with tab5:
            self._render_optimization_tab()
    
    def _render_system_status_header(self):
        """Render system status header with key indicators"""
        
        system_status = self._get_system_status()
        
        # Status indicators
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            status_color = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}
            st.metric("System Status", f"{status_color.get(system_status['status'], '🔵')} {system_status['status'].upper()}")
        
        with col2:
            st.metric("Uptime", system_status['uptime'])
        
        with col3:
            st.metric("Requests/sec", f"{system_status['requests_per_sec']:.1f}")
        
        with col4:
            st.metric("Avg Response", f"{system_status['avg_response_ms']:.1f}ms")
        
        with col5:
            st.metric("Error Rate", f"{system_status['error_rate_pct']:.2f}%")
        
        with col6:
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
            if auto_refresh != st.session_state.auto_refresh:
                st.session_state.auto_refresh = auto_refresh
                if auto_refresh:
                    st.rerun()
        
        # Active alerts banner
        active_alerts = [alert for alert in st.session_state.alerts if alert['status'] == 'active']
        if active_alerts:
            critical_alerts = [a for a in active_alerts if a['severity'] == 'CRITICAL']
            warning_alerts = [a for a in active_alerts if a['severity'] == 'WARNING']
            
            if critical_alerts:
                st.error(f"🚨 {len(critical_alerts)} CRITICAL alerts active! Immediate attention required.")
            elif warning_alerts:
                st.warning(f"⚠️ {len(warning_alerts)} WARNING alerts active.")
        
        # Auto-refresh mechanism
        if st.session_state.auto_refresh:
            time.sleep(5)  # Refresh every 5 seconds
            st.rerun()
    
    def _render_live_metrics_tab(self):
        """Render live metrics dashboard"""
        st.markdown("## 📊 Live System Metrics")
        
        # Real-time metrics
        metrics_data = self._get_live_metrics()
        
        # Key performance indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 🎯 Throughput")
            
            # Requests per second gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = metrics_data['throughput']['current_rps'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Requests/sec"},
                delta = {'reference': metrics_data['throughput']['target_rps']},
                gauge = {
                    'axis': {'range': [None, metrics_data['throughput']['max_rps']]},
                    'bar': {'color': "#c8712d"},
                    'steps': [
                        {'range': [0, metrics_data['throughput']['target_rps'] * 0.7], 'color': "#d4edda"},
                        {'range': [metrics_data['throughput']['target_rps'] * 0.7, metrics_data['throughput']['target_rps']], 'color': "#fff3cd"},
                        {'range': [metrics_data['throughput']['target_rps'], metrics_data['throughput']['max_rps']], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics_data['throughput']['max_rps'] * 0.9
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⏱️ Response Time")
            
            # Response time gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = metrics_data['response_time']['current_ms'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Response Time (ms)"},
                delta = {'reference': metrics_data['response_time']['target_ms']},
                gauge = {
                    'axis': {'range': [0, metrics_data['response_time']['max_ms']]},
                    'bar': {'color': "#002466"},
                    'steps': [
                        {'range': [0, metrics_data['response_time']['target_ms']], 'color': "#d4edda"},
                        {'range': [metrics_data['response_time']['target_ms'], metrics_data['response_time']['target_ms'] * 2], 'color': "#fff3cd"},
                        {'range': [metrics_data['response_time']['target_ms'] * 2, metrics_data['response_time']['max_ms']], 'color': "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### 🎯 Success Rate")
            
            # Success rate gauge
            success_rate = 100 - metrics_data['error_rate']['current_pct']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = success_rate,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Success Rate (%)"},
                gauge = {
                    'axis': {'range': [90, 100]},
                    'bar': {'color': "#28a745"},
                    'steps': [
                        {'range': [90, 95], 'color': "#f8d7da"},
                        {'range': [95, 98], 'color': "#fff3cd"},
                        {'range': [98, 100], 'color': "#d4edda"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("### 💾 Memory Usage")
            
            # Memory usage gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = metrics_data['memory']['usage_pct'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Memory Usage (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#660003"},
                    'steps': [
                        {'range': [0, 70], 'color': "#d4edda"},
                        {'range': [70, 85], 'color': "#fff3cd"},
                        {'range': [85, 100], 'color': "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Request Volume Timeline")
            
            timeline_data = self._get_request_timeline()
            fig = px.line(
                timeline_data,
                x='timestamp',
                y='requests_per_minute',
                title='Requests per Minute (Last Hour)',
                color_discrete_sequence=['#c8712d']
            )
            fig.add_hline(y=metrics_data['throughput']['target_rps'] * 60, line_dash="dash", line_color="green", annotation_text="Target")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Response Time Distribution")
            
            response_dist = self._get_response_time_distribution()
            fig = px.histogram(
                response_dist,
                x='response_time_ms',
                title='Response Time Distribution (Last Hour)',
                nbins=30,
                color_discrete_sequence=['#002466']
            )
            fig.add_vline(x=metrics_data['response_time']['target_ms'], line_dash="dash", line_color="red", annotation_text="Target")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Decision processing metrics
        st.markdown("### ⚖️ Decision Processing Metrics")
        
        decision_metrics = metrics_data['decision_processing']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Decisions/min", f"{decision_metrics['decisions_per_minute']:.1f}")
        
        with col2:
            st.metric("Avg Processing Time", f"{decision_metrics['avg_processing_ms']:.1f}ms")
        
        with col3:
            approval_rate = (decision_metrics['approvals'] / decision_metrics['total_decisions']) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        with col4:
            st.metric("Queue Depth", decision_metrics['queue_depth'])
        
        # Decision outcomes over time
        decision_timeline = self._get_decision_timeline()
        fig = px.bar(
            decision_timeline,
            x='timestamp',
            y='count',
            color='decision_type',
            title='Decision Outcomes Timeline (Last 4 Hours)',
            color_discrete_map={
                'APPROVE': '#28a745',
                'REJECT': '#dc3545',
                'AMEND': '#ffc107'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts_tab(self):
        """Render alerts management tab"""
        st.markdown("## 🚨 Alert Management")
        
        # Alert summary
        alert_summary = self._get_alert_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Alerts", alert_summary['active'])
        
        with col2:
            st.metric("Critical", alert_summary['critical'], delta=alert_summary['critical_change'])
        
        with col3:
            st.metric("Warnings", alert_summary['warnings'], delta=alert_summary['warnings_change'])
        
        with col4:
            st.metric("Info", alert_summary['info'])
        
        # Alert configuration
        with st.expander("🔧 Alert Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Performance Thresholds")
                
                response_time_threshold = st.slider("Response Time Alert (ms)", 10, 1000, 100)
                error_rate_threshold = st.slider("Error Rate Alert (%)", 0.1, 10.0, 1.0, step=0.1)
                throughput_threshold = st.slider("Min Throughput (req/s)", 1.0, 100.0, 10.0)
            
            with col2:
                st.markdown("#### Resource Thresholds")
                
                cpu_threshold = st.slider("CPU Usage Alert (%)", 50, 95, 80)
                memory_threshold = st.slider("Memory Usage Alert (%)", 50, 95, 85)
                disk_threshold = st.slider("Disk Usage Alert (%)", 50, 95, 90)
            
            if st.button("💾 Save Alert Configuration"):
                st.success("Alert configuration saved!")
        
        # Active alerts
        st.markdown("### 🚨 Active Alerts")
        
        active_alerts = [alert for alert in st.session_state.alerts if alert['status'] == 'active']
        
        if active_alerts:
            for alert in active_alerts:
                severity_colors = {
                    'CRITICAL': '🔴',
                    'WARNING': '🟡', 
                    'INFO': '🔵'
                }
                
                with st.expander(f"{severity_colors[alert['severity']]} {alert['title']} - {alert['timestamp']}", expanded=alert['severity'] == 'CRITICAL'):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Severity:** {alert['severity']}")
                        st.markdown(f"**Component:** {alert['component']}")
                        st.markdown(f"**Description:** {alert['description']}")
                        st.markdown(f"**Current Value:** {alert['current_value']}")
                        st.markdown(f"**Threshold:** {alert['threshold']}")
                        
                        if alert.get('recommendations'):
                            st.markdown("**Recommendations:**")
                            for rec in alert['recommendations']:
                                st.markdown(f"- {rec}")
                    
                    with col2:
                        if st.button(f"✅ Acknowledge", key=f"ack_{alert['id']}"):
                            self._acknowledge_alert(alert['id'])
                            st.success("Alert acknowledged!")
                            st.rerun()
                        
                        if st.button(f"🔇 Silence 1h", key=f"silence_{alert['id']}"):
                            self._silence_alert(alert['id'], 3600)
                            st.info("Alert silenced for 1 hour")
                            st.rerun()
        else:
            st.success("✅ No active alerts")
        
        # Alert history
        st.markdown("### 📊 Alert History")
        
        alert_history = self._get_alert_history()
        
        if alert_history:
            # Alert trends
            col1, col2 = st.columns(2)
            
            with col1:
                # Alerts by severity over time
                severity_timeline = self._get_alert_severity_timeline()
                fig = px.bar(
                    severity_timeline,
                    x='date',
                    y='count',
                    color='severity',
                    title='Alerts by Severity (Last 7 Days)',
                    color_discrete_map={
                        'CRITICAL': '#dc3545',
                        'WARNING': '#ffc107',
                        'INFO': '#17a2b8'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Alerts by component
                component_alerts = self._get_alerts_by_component()
                fig = px.pie(
                    values=list(component_alerts.values()),
                    names=list(component_alerts.keys()),
                    title='Alerts by Component (Last 7 Days)'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Alert history table
            df = pd.DataFrame(alert_history)
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                severity_filter = st.selectbox("Severity", ["All"] + list(df['severity'].unique()))
            
            with col2:
                component_filter = st.selectbox("Component", ["All"] + list(df['component'].unique()))
            
            with col3:
                status_filter = st.selectbox("Status", ["All"] + list(df['status'].unique()))
            
            # Apply filters
            filtered_df = df.copy()
            if severity_filter != "All":
                filtered_df = filtered_df[filtered_df['severity'] == severity_filter]
            if component_filter != "All":
                filtered_df = filtered_df[filtered_df['component'] == component_filter]
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
            
            st.dataframe(filtered_df, use_container_width=True)
        
        # Alert notifications
        st.markdown("### 📧 Notification Settings")
        
        with st.expander("Configure Notifications", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Email Notifications")
                
                email_enabled = st.checkbox("Enable Email Alerts", value=True)
                
                if email_enabled:
                    email_addresses = st.text_area(
                        "Email Addresses (one per line)",
                        value="admin@wicketwise.com\nops@wicketwise.com"
                    )
                    
                    email_severity = st.multiselect(
                        "Email Severity Levels",
                        ["CRITICAL", "WARNING", "INFO"],
                        default=["CRITICAL", "WARNING"]
                    )
            
            with col2:
                st.markdown("#### Slack Notifications")
                
                slack_enabled = st.checkbox("Enable Slack Alerts", value=True)
                
                if slack_enabled:
                    slack_webhook = st.text_input(
                        "Slack Webhook URL",
                        type="password",
                        placeholder="https://hooks.slack.com/..."
                    )
                    
                    slack_channel = st.text_input("Slack Channel", value="#dgl-alerts")
                    
                    slack_severity = st.multiselect(
                        "Slack Severity Levels",
                        ["CRITICAL", "WARNING", "INFO"],
                        default=["CRITICAL"]
                    )
            
            if st.button("💾 Save Notification Settings"):
                st.success("Notification settings saved!")
    
    def _render_resources_tab(self):
        """Render resource monitoring tab"""
        st.markdown("## 💻 Resource Monitoring")
        
        # Resource overview
        resource_data = self._get_resource_metrics()
        
        # System resources
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### 🖥️ CPU Usage")
            
            cpu_usage = resource_data['cpu']['usage_pct']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cpu_usage,
                title = {'text': "CPU Usage (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#c8712d"},
                    'steps': [
                        {'range': [0, 70], 'color': "#d4edda"},
                        {'range': [70, 85], 'color': "#fff3cd"},
                        {'range': [85, 100], 'color': "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💾 Memory Usage")
            
            memory_usage = resource_data['memory']['usage_pct']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = memory_usage,
                title = {'text': "Memory Usage (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#002466"},
                    'steps': [
                        {'range': [0, 70], 'color': "#d4edda"},
                        {'range': [70, 85], 'color': "#fff3cd"},
                        {'range': [85, 100], 'color': "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("### 💽 Disk Usage")
            
            disk_usage = resource_data['disk']['usage_pct']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = disk_usage,
                title = {'text': "Disk Usage (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#660003"},
                    'steps': [
                        {'range': [0, 80], 'color': "#d4edda"},
                        {'range': [80, 90], 'color': "#fff3cd"},
                        {'range': [90, 100], 'color': "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("### 🌐 Network I/O")
            
            network_usage = resource_data['network']['usage_mbps']
            max_bandwidth = resource_data['network']['max_bandwidth_mbps']
            usage_pct = (network_usage / max_bandwidth) * 100
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = usage_pct,
                title = {'text': "Network Usage (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#28a745"},
                    'steps': [
                        {'range': [0, 60], 'color': "#d4edda"},
                        {'range': [60, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed resource metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Resource Utilization Timeline")
            
            resource_timeline = self._get_resource_timeline()
            fig = px.line(
                resource_timeline,
                x='timestamp',
                y=['cpu_pct', 'memory_pct', 'disk_pct'],
                title='Resource Utilization (Last 24 Hours)',
                labels={'value': 'Usage (%)', 'timestamp': 'Time'}
            )
            fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
            fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔄 Process Information")
            
            process_info = resource_data['processes']
            
            st.markdown("#### Top Processes by CPU")
            for proc in process_info['top_cpu'][:5]:
                col2a, col2b, col2c = st.columns([2, 1, 1])
                with col2a:
                    st.markdown(f"**{proc['name']}** (PID: {proc['pid']})")
                with col2b:
                    st.markdown(f"CPU: {proc['cpu_pct']:.1f}%")
                with col2c:
                    st.markdown(f"Mem: {proc['memory_mb']:.0f}MB")
            
            st.markdown("#### Top Processes by Memory")
            for proc in process_info['top_memory'][:5]:
                col2a, col2b, col2c = st.columns([2, 1, 1])
                with col2a:
                    st.markdown(f"**{proc['name']}** (PID: {proc['pid']})")
                with col2b:
                    st.markdown(f"CPU: {proc['cpu_pct']:.1f}%")
                with col2c:
                    st.markdown(f"Mem: {proc['memory_mb']:.0f}MB")
        
        # Database and cache metrics
        st.markdown("### 🗄️ Database & Cache Metrics")
        
        db_metrics = resource_data['database']
        cache_metrics = resource_data['cache']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("DB Connections", f"{db_metrics['active_connections']}/{db_metrics['max_connections']}")
        
        with col2:
            st.metric("DB Query Time", f"{db_metrics['avg_query_ms']:.1f}ms")
        
        with col3:
            st.metric("Cache Hit Rate", f"{cache_metrics['hit_rate_pct']:.1f}%")
        
        with col4:
            st.metric("Cache Memory", f"{cache_metrics['memory_usage_mb']:.0f}MB")
        
        # Storage breakdown
        st.markdown("### 💽 Storage Breakdown")
        
        storage_data = resource_data['storage']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Storage usage by type
            fig = px.pie(
                values=list(storage_data['by_type'].values()),
                names=list(storage_data['by_type'].keys()),
                title='Storage Usage by Type'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Storage growth trend
            growth_data = storage_data['growth_trend']
            fig = px.line(
                growth_data,
                x='date',
                y='size_gb',
                title='Storage Growth Trend (Last 30 Days)',
                color_discrete_sequence=['#c8712d']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_tab(self):
        """Render performance analysis tab"""
        st.markdown("## ⚡ Performance Analysis")
        
        # Performance overview
        perf_data = self._get_performance_metrics()
        
        # Key performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("P50 Response Time", f"{perf_data['response_times']['p50_ms']:.1f}ms")
        
        with col2:
            st.metric("P95 Response Time", f"{perf_data['response_times']['p95_ms']:.1f}ms")
        
        with col3:
            st.metric("P99 Response Time", f"{perf_data['response_times']['p99_ms']:.1f}ms")
        
        with col4:
            st.metric("Max Response Time", f"{perf_data['response_times']['max_ms']:.1f}ms")
        
        # Performance trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Response Time Percentiles")
            
            percentile_data = self._get_response_time_percentiles()
            fig = px.line(
                percentile_data,
                x='timestamp',
                y=['p50', 'p95', 'p99'],
                title='Response Time Percentiles (Last 24 Hours)',
                labels={'value': 'Response Time (ms)', 'timestamp': 'Time'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Throughput vs Latency")
            
            throughput_latency = self._get_throughput_latency_correlation()
            fig = px.scatter(
                throughput_latency,
                x='throughput_rps',
                y='avg_latency_ms',
                size='error_rate_pct',
                title='Throughput vs Latency Correlation',
                labels={'throughput_rps': 'Throughput (req/s)', 'avg_latency_ms': 'Avg Latency (ms)'},
                color_discrete_sequence=['#002466']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Endpoint performance analysis
        st.markdown("### 🔗 Endpoint Performance Analysis")
        
        endpoint_perf = perf_data['endpoints']
        
        # Top slowest endpoints
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Slowest Endpoints")
            
            slowest_endpoints = sorted(endpoint_perf.items(), key=lambda x: x[1]['avg_response_ms'], reverse=True)[:10]
            
            endpoints = [ep[0] for ep in slowest_endpoints]
            response_times = [ep[1]['avg_response_ms'] for ep in slowest_endpoints]
            
            fig = px.bar(
                x=response_times,
                y=endpoints,
                orientation='h',
                title='Top 10 Slowest Endpoints',
                labels={'x': 'Avg Response Time (ms)', 'y': 'Endpoint'},
                color_discrete_sequence=['#dc3545']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Highest Traffic Endpoints")
            
            busiest_endpoints = sorted(endpoint_perf.items(), key=lambda x: x[1]['requests_per_hour'], reverse=True)[:10]
            
            endpoints = [ep[0] for ep in busiest_endpoints]
            request_counts = [ep[1]['requests_per_hour'] for ep in busiest_endpoints]
            
            fig = px.bar(
                x=request_counts,
                y=endpoints,
                orientation='h',
                title='Top 10 Busiest Endpoints',
                labels={'x': 'Requests per Hour', 'y': 'Endpoint'},
                color_discrete_sequence=['#28a745']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        st.markdown("### 🚨 Error Analysis")
        
        error_data = perf_data['errors']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Errors (24h)", error_data['total_errors'])
        
        with col2:
            st.metric("Error Rate", f"{error_data['error_rate_pct']:.2f}%")
        
        with col3:
            st.metric("Most Common Error", error_data['most_common_error'])
        
        # Error breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Errors by type
            error_types = error_data['by_type']
            fig = px.pie(
                values=list(error_types.values()),
                names=list(error_types.keys()),
                title='Errors by Type (Last 24 Hours)'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error timeline
            error_timeline = error_data['timeline']
            fig = px.line(
                error_timeline,
                x='timestamp',
                y='error_count',
                title='Error Count Timeline (Last 24 Hours)',
                color_discrete_sequence=['#dc3545']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance recommendations
        st.markdown("### 💡 Performance Recommendations")
        
        recommendations = perf_data['recommendations']
        
        for rec in recommendations:
            priority_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            
            with st.expander(f"{priority_colors[rec['priority']]} {rec['title']}", expanded=rec['priority'] == 'HIGH'):
                st.markdown(f"**Impact:** {rec['impact']}")
                st.markdown(f"**Description:** {rec['description']}")
                st.markdown(f"**Implementation:** {rec['implementation']}")
                
                if rec.get('estimated_improvement'):
                    st.markdown(f"**Estimated Improvement:** {rec['estimated_improvement']}")
    
    def _render_optimization_tab(self):
        """Render optimization recommendations tab"""
        st.markdown("## 🔧 System Optimization")
        
        # Optimization overview
        opt_data = self._get_optimization_data()
        
        # Optimization score
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            opt_score = opt_data['overall_score']
            score_color = "🟢" if opt_score >= 90 else "🟡" if opt_score >= 70 else "🔴"
            st.metric("Optimization Score", f"{score_color} {opt_score:.1f}%")
        
        with col2:
            st.metric("Potential Savings", f"£{opt_data['potential_savings']:,.0f}/month")
        
        with col3:
            st.metric("Performance Gain", f"+{opt_data['performance_gain_pct']:.1f}%")
        
        with col4:
            st.metric("Recommendations", len(opt_data['recommendations']))
        
        # Optimization categories
        st.markdown("### 📊 Optimization Categories")
        
        categories = opt_data['categories']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category scores
            cat_names = list(categories.keys())
            cat_scores = [categories[cat]['score'] for cat in cat_names]
            
            fig = px.bar(
                x=cat_scores,
                y=cat_names,
                orientation='h',
                title='Optimization Scores by Category',
                labels={'x': 'Score (%)', 'y': 'Category'},
                color=cat_scores,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Potential impact
            cat_impacts = [categories[cat]['potential_impact'] for cat in cat_names]
            
            fig = px.pie(
                values=cat_impacts,
                names=cat_names,
                title='Potential Impact by Category'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed recommendations
        st.markdown("### 💡 Optimization Recommendations")
        
        recommendations = opt_data['recommendations']
        
        # Filter and sort recommendations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            priority_filter = st.selectbox("Priority", ["All", "HIGH", "MEDIUM", "LOW"])
        
        with col2:
            category_filter = st.selectbox("Category", ["All"] + list(categories.keys()))
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Priority", "Impact", "Effort"])
        
        # Apply filters
        filtered_recs = recommendations.copy()
        if priority_filter != "All":
            filtered_recs = [r for r in filtered_recs if r['priority'] == priority_filter]
        if category_filter != "All":
            filtered_recs = [r for r in filtered_recs if r['category'] == category_filter]
        
        # Sort recommendations
        if sort_by == "Priority":
            priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            filtered_recs.sort(key=lambda x: priority_order[x['priority']], reverse=True)
        elif sort_by == "Impact":
            filtered_recs.sort(key=lambda x: x['impact_score'], reverse=True)
        else:  # Effort
            filtered_recs.sort(key=lambda x: x['effort_score'])
        
        # Display recommendations
        for rec in filtered_recs:
            priority_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            
            with st.expander(f"{priority_colors[rec['priority']]} {rec['title']} - {rec['category']}", expanded=rec['priority'] == 'HIGH'):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {rec['description']}")
                    st.markdown(f"**Current State:** {rec['current_state']}")
                    st.markdown(f"**Proposed Solution:** {rec['proposed_solution']}")
                    
                    if rec.get('implementation_steps'):
                        st.markdown("**Implementation Steps:**")
                        for i, step in enumerate(rec['implementation_steps'], 1):
                            st.markdown(f"{i}. {step}")
                
                with col2:
                    st.markdown("**Metrics:**")
                    st.metric("Impact Score", f"{rec['impact_score']}/10")
                    st.metric("Effort Score", f"{rec['effort_score']}/10")
                    st.metric("ROI", f"{rec['roi_months']:.1f} months")
                    
                    if rec.get('estimated_savings'):
                        st.metric("Monthly Savings", f"£{rec['estimated_savings']:,.0f}")
                    
                    if st.button(f"✅ Mark as Implemented", key=f"impl_{rec['id']}"):
                        st.success("Recommendation marked as implemented!")
        
        # Resource optimization
        st.markdown("### 💻 Resource Optimization")
        
        resource_opt = opt_data['resource_optimization']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### CPU Optimization")
            
            cpu_opt = resource_opt['cpu']
            st.markdown(f"**Current Usage:** {cpu_opt['current_usage_pct']:.1f}%")
            st.markdown(f"**Optimal Usage:** {cpu_opt['optimal_usage_pct']:.1f}%")
            st.markdown(f"**Potential Savings:** £{cpu_opt['potential_savings']:,.0f}/month")
            
            if cpu_opt['recommendations']:
                st.markdown("**Recommendations:**")
                for rec in cpu_opt['recommendations']:
                    st.markdown(f"- {rec}")
        
        with col2:
            st.markdown("#### Memory Optimization")
            
            mem_opt = resource_opt['memory']
            st.markdown(f"**Current Usage:** {mem_opt['current_usage_pct']:.1f}%")
            st.markdown(f"**Optimal Usage:** {mem_opt['optimal_usage_pct']:.1f}%")
            st.markdown(f"**Potential Savings:** £{mem_opt['potential_savings']:,.0f}/month")
            
            if mem_opt['recommendations']:
                st.markdown("**Recommendations:**")
                for rec in mem_opt['recommendations']:
                    st.markdown(f"- {rec}")
        
        # Cost optimization
        st.markdown("### 💰 Cost Optimization")
        
        cost_opt = opt_data['cost_optimization']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Monthly Cost", f"£{cost_opt['current_monthly_cost']:,.0f}")
        
        with col2:
            st.metric("Optimized Monthly Cost", f"£{cost_opt['optimized_monthly_cost']:,.0f}")
        
        with col3:
            savings = cost_opt['current_monthly_cost'] - cost_opt['optimized_monthly_cost']
            st.metric("Monthly Savings", f"£{savings:,.0f}")
        
        # Cost breakdown
        cost_breakdown = cost_opt['cost_breakdown']
        
        fig = px.bar(
            x=list(cost_breakdown['current'].values()),
            y=list(cost_breakdown['current'].keys()),
            orientation='h',
            title='Current vs Optimized Costs',
            labels={'x': 'Monthly Cost (£)', 'y': 'Service'},
            color_discrete_sequence=['#c8712d']
        )
        
        # Add optimized costs as a second series
        fig.add_trace(go.Bar(
            x=list(cost_breakdown['optimized'].values()),
            y=list(cost_breakdown['optimized'].keys()),
            orientation='h',
            name='Optimized',
            marker_color='#28a745'
        ))
        
        fig.update_layout(height=400, barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Helper methods for data retrieval (mock implementations)
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status (mock)"""
        return {
            'status': 'healthy',
            'uptime': '15d 4h 23m',
            'requests_per_sec': 12.3,
            'avg_response_ms': 23.5,
            'error_rate_pct': 0.08
        }
    
    def _get_live_metrics(self) -> Dict[str, Any]:
        """Get live metrics data (mock)"""
        return {
            'throughput': {
                'current_rps': 12.3,
                'target_rps': 15.0,
                'max_rps': 50.0
            },
            'response_time': {
                'current_ms': 23.5,
                'target_ms': 50.0,
                'max_ms': 200.0
            },
            'error_rate': {
                'current_pct': 0.08
            },
            'memory': {
                'usage_pct': 67.3
            },
            'decision_processing': {
                'decisions_per_minute': 45.2,
                'avg_processing_ms': 18.7,
                'approvals': 892,
                'total_decisions': 1247,
                'queue_depth': 3
            }
        }
    
    def _get_request_timeline(self) -> pd.DataFrame:
        """Get request timeline data (mock)"""
        import numpy as np
        
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), end=datetime.now(), freq='min')
        requests = np.random.poisson(12, len(timestamps)) * 60  # Convert to per minute
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'requests_per_minute': requests
        })
    
    def _get_response_time_distribution(self) -> pd.DataFrame:
        """Get response time distribution (mock)"""
        import numpy as np
        
        response_times = np.random.gamma(2, 12, 1000)  # Gamma distribution for realistic response times
        
        return pd.DataFrame({
            'response_time_ms': response_times
        })
    
    def _get_decision_timeline(self) -> pd.DataFrame:
        """Get decision timeline (mock)"""
        import numpy as np
        
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=4), end=datetime.now(), freq='15min')
        
        data = []
        for ts in timestamps:
            for decision_type in ['APPROVE', 'REJECT', 'AMEND']:
                count = max(0, np.random.poisson(20) + {'APPROVE': 15, 'REJECT': -5, 'AMEND': -10}[decision_type])
                data.append({
                    'timestamp': ts,
                    'decision_type': decision_type,
                    'count': count
                })
        
        return pd.DataFrame(data)
    
    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary (mock)"""
        return {
            'active': 5,
            'critical': 1,
            'critical_change': 0,
            'warnings': 3,
            'warnings_change': 1,
            'info': 1
        }
    
    def _acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in st.session_state.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'acknowledged'
                alert['acknowledged_at'] = datetime.now().isoformat()
                break
    
    def _silence_alert(self, alert_id: str, duration_seconds: int):
        """Silence an alert for specified duration"""
        for alert in st.session_state.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'silenced'
                alert['silenced_until'] = (datetime.now() + timedelta(seconds=duration_seconds)).isoformat()
                break
    
    def _get_alert_history(self) -> List[Dict[str, Any]]:
        """Get alert history (mock)"""
        # Initialize some mock alerts if none exist
        if not st.session_state.alerts:
            st.session_state.alerts = [
                {
                    'id': 'alert_001',
                    'title': 'High Response Time',
                    'severity': 'WARNING',
                    'component': 'API Gateway',
                    'description': 'Average response time exceeded 100ms threshold',
                    'current_value': '125ms',
                    'threshold': '100ms',
                    'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
                    'status': 'active',
                    'recommendations': ['Check database performance', 'Review recent deployments']
                },
                {
                    'id': 'alert_002',
                    'title': 'Memory Usage High',
                    'severity': 'CRITICAL',
                    'component': 'DGL Engine',
                    'description': 'Memory usage exceeded 90% threshold',
                    'current_value': '92%',
                    'threshold': '90%',
                    'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat(),
                    'status': 'active',
                    'recommendations': ['Restart service', 'Check for memory leaks', 'Scale horizontally']
                }
            ]
        
        return st.session_state.alerts
    
    def _get_alert_severity_timeline(self) -> pd.DataFrame:
        """Get alert severity timeline (mock)"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
        
        data = []
        for date in dates:
            for severity in ['CRITICAL', 'WARNING', 'INFO']:
                count = max(0, np.random.poisson(3) + {'CRITICAL': -2, 'WARNING': 0, 'INFO': 1}[severity])
                data.append({
                    'date': date.date(),
                    'severity': severity,
                    'count': count
                })
        
        return pd.DataFrame(data)
    
    def _get_alerts_by_component(self) -> Dict[str, int]:
        """Get alerts by component (mock)"""
        return {
            'API Gateway': 12,
            'DGL Engine': 8,
            'Database': 5,
            'Cache': 3,
            'Load Balancer': 2
        }
    
    def _get_resource_metrics(self) -> Dict[str, Any]:
        """Get resource metrics (mock)"""
        return {
            'cpu': {'usage_pct': 67.3},
            'memory': {'usage_pct': 72.1},
            'disk': {'usage_pct': 45.8},
            'network': {'usage_mbps': 125.5, 'max_bandwidth_mbps': 1000.0},
            'processes': {
                'top_cpu': [
                    {'name': 'dgl-engine', 'pid': 1234, 'cpu_pct': 15.2, 'memory_mb': 512},
                    {'name': 'postgres', 'pid': 5678, 'cpu_pct': 12.8, 'memory_mb': 256},
                    {'name': 'redis', 'pid': 9012, 'cpu_pct': 8.5, 'memory_mb': 128}
                ],
                'top_memory': [
                    {'name': 'dgl-engine', 'pid': 1234, 'cpu_pct': 15.2, 'memory_mb': 512},
                    {'name': 'postgres', 'pid': 5678, 'cpu_pct': 12.8, 'memory_mb': 256},
                    {'name': 'nginx', 'pid': 3456, 'cpu_pct': 2.1, 'memory_mb': 64}
                ]
            },
            'database': {
                'active_connections': 25,
                'max_connections': 100,
                'avg_query_ms': 15.3
            },
            'cache': {
                'hit_rate_pct': 94.2,
                'memory_usage_mb': 256
            },
            'storage': {
                'by_type': {
                    'Audit Logs': 45.2,
                    'Database': 32.1,
                    'Cache': 12.8,
                    'System Logs': 8.7,
                    'Other': 1.2
                },
                'growth_trend': pd.DataFrame({
                    'date': pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D'),
                    'size_gb': [95 + i * 0.5 + (i % 7) * 0.2 for i in range(30)]
                })
            }
        }
    
    def _get_resource_timeline(self) -> pd.DataFrame:
        """Get resource timeline (mock)"""
        import numpy as np
        
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
        
        cpu_usage = 60 + 20 * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
        memory_usage = 65 + 15 * np.sin(np.linspace(0, 3*np.pi, len(timestamps))) + np.random.normal(0, 3, len(timestamps))
        disk_usage = 45 + np.random.normal(0, 2, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'cpu_pct': np.clip(cpu_usage, 0, 100),
            'memory_pct': np.clip(memory_usage, 0, 100),
            'disk_pct': np.clip(disk_usage, 0, 100)
        })
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics (mock)"""
        return {
            'response_times': {
                'p50_ms': 18.5,
                'p95_ms': 67.2,
                'p99_ms': 145.8,
                'max_ms': 892.3
            },
            'endpoints': {
                '/api/v1/governance/evaluate': {
                    'avg_response_ms': 25.3,
                    'requests_per_hour': 2340
                },
                '/api/v1/exposure/current': {
                    'avg_response_ms': 12.1,
                    'requests_per_hour': 1890
                },
                '/api/v1/rules/config': {
                    'avg_response_ms': 45.7,
                    'requests_per_hour': 156
                }
            },
            'errors': {
                'total_errors': 23,
                'error_rate_pct': 0.08,
                'most_common_error': 'Validation Error',
                'by_type': {
                    'Validation Error': 12,
                    'Network Timeout': 8,
                    'Database Error': 3
                },
                'timeline': pd.DataFrame({
                    'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H'),
                    'error_count': np.random.poisson(1, 24)
                })
            },
            'recommendations': [
                {
                    'priority': 'HIGH',
                    'title': 'Optimize Database Queries',
                    'impact': 'Reduce P95 response time by 30%',
                    'description': 'Several endpoints are experiencing slow database queries',
                    'implementation': 'Add database indexes and optimize query patterns',
                    'estimated_improvement': '30% faster P95 response time'
                },
                {
                    'priority': 'MEDIUM',
                    'title': 'Implement Response Caching',
                    'impact': 'Reduce load by 40%',
                    'description': 'Many requests could benefit from caching',
                    'implementation': 'Add Redis caching layer for frequently accessed data',
                    'estimated_improvement': '40% reduction in database load'
                }
            ]
        }
    
    def _get_response_time_percentiles(self) -> pd.DataFrame:
        """Get response time percentiles (mock)"""
        import numpy as np
        
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
        
        p50 = 15 + 10 * np.sin(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 2, len(timestamps))
        p95 = 45 + 20 * np.sin(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
        p99 = 85 + 30 * np.sin(np.linspace(0, 2*np.pi, len(timestamps))) + np.random.normal(0, 8, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'p50': np.maximum(p50, 5),
            'p95': np.maximum(p95, 20),
            'p99': np.maximum(p99, 40)
        })
    
    def _get_throughput_latency_correlation(self) -> pd.DataFrame:
        """Get throughput vs latency correlation (mock)"""
        import numpy as np
        
        throughput = np.random.uniform(5, 25, 100)
        latency = 10 + (throughput - 5) * 2 + np.random.normal(0, 5, 100)
        error_rate = np.maximum(0, (throughput - 15) * 0.1 + np.random.normal(0, 0.2, 100))
        
        return pd.DataFrame({
            'throughput_rps': throughput,
            'avg_latency_ms': np.maximum(latency, 5),
            'error_rate_pct': np.maximum(error_rate, 0)
        })
    
    def _get_optimization_data(self) -> Dict[str, Any]:
        """Get optimization data (mock)"""
        return {
            'overall_score': 78.5,
            'potential_savings': 2340,
            'performance_gain_pct': 25.3,
            'categories': {
                'Performance': {'score': 75.2, 'potential_impact': 35},
                'Cost': {'score': 82.1, 'potential_impact': 25},
                'Security': {'score': 91.5, 'potential_impact': 15},
                'Reliability': {'score': 68.9, 'potential_impact': 25}
            },
            'recommendations': [
                {
                    'id': 'opt_001',
                    'title': 'Implement Database Connection Pooling',
                    'category': 'Performance',
                    'priority': 'HIGH',
                    'description': 'Database connections are not being pooled efficiently',
                    'current_state': 'Each request creates new database connection',
                    'proposed_solution': 'Implement connection pooling with pgbouncer',
                    'impact_score': 8,
                    'effort_score': 3,
                    'roi_months': 2.1,
                    'estimated_savings': 450,
                    'implementation_steps': [
                        'Install and configure pgbouncer',
                        'Update application database configuration',
                        'Monitor connection pool metrics',
                        'Tune pool size based on load'
                    ]
                },
                {
                    'id': 'opt_002',
                    'title': 'Optimize Memory Allocation',
                    'category': 'Cost',
                    'priority': 'MEDIUM',
                    'description': 'Memory is over-allocated for current workload',
                    'current_state': '8GB allocated, 5.5GB average usage',
                    'proposed_solution': 'Reduce allocation to 6GB with auto-scaling',
                    'impact_score': 6,
                    'effort_score': 2,
                    'roi_months': 1.5,
                    'estimated_savings': 280,
                    'implementation_steps': [
                        'Analyze memory usage patterns',
                        'Implement auto-scaling policies',
                        'Gradually reduce base allocation',
                        'Monitor for performance impact'
                    ]
                }
            ],
            'resource_optimization': {
                'cpu': {
                    'current_usage_pct': 67.3,
                    'optimal_usage_pct': 75.0,
                    'potential_savings': 340,
                    'recommendations': [
                        'Enable CPU auto-scaling',
                        'Optimize compute-intensive algorithms'
                    ]
                },
                'memory': {
                    'current_usage_pct': 72.1,
                    'optimal_usage_pct': 80.0,
                    'potential_savings': 280,
                    'recommendations': [
                        'Implement memory pooling',
                        'Optimize data structures'
                    ]
                }
            },
            'cost_optimization': {
                'current_monthly_cost': 4500,
                'optimized_monthly_cost': 3200,
                'cost_breakdown': {
                    'current': {
                        'Compute': 2500,
                        'Storage': 800,
                        'Network': 600,
                        'Database': 400,
                        'Monitoring': 200
                    },
                    'optimized': {
                        'Compute': 1800,
                        'Storage': 600,
                        'Network': 450,
                        'Database': 250,
                        'Monitoring': 100
                    }
                }
            }
        }


def main():
    """Main entry point for monitoring panel"""
    monitoring_panel = MonitoringPanel()
    monitoring_panel.render()


if __name__ == "__main__":
    main()
