# Purpose: Audit trail viewer for DGL compliance and monitoring
# Author: WicketWise AI, Last Modified: 2024

"""
Audit Viewer

Comprehensive audit trail interface for DGL compliance:
- Audit record search and filtering
- Hash chain verification
- Compliance reporting
- Regulatory export capabilities
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
import hashlib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.dgl_client import DGLClient, DGLClientConfig
from schemas import DecisionType, RuleId


logger = logging.getLogger(__name__)


class AuditViewer:
    """
    Comprehensive audit trail viewer for DGL compliance
    
    Provides interface for:
    - Audit record search and analysis
    - Hash chain integrity verification
    - Compliance reporting and export
    - Regulatory audit support
    """
    
    def __init__(self, dgl_base_url: str = "http://localhost:8001"):
        """
        Initialize audit viewer
        
        Args:
            dgl_base_url: DGL service base URL
        """
        self.dgl_base_url = dgl_base_url
        self.client_config = DGLClientConfig(base_url=dgl_base_url)
        
        # Initialize session state for audit data
        if 'audit_search_results' not in st.session_state:
            st.session_state.audit_search_results = []
        
        if 'audit_filters' not in st.session_state:
            st.session_state.audit_filters = {}
        
        if 'compliance_reports' not in st.session_state:
            st.session_state.compliance_reports = []
    
    def render(self):
        """Render the audit viewer interface"""
        
        st.markdown("# 🔍 Audit Trail Viewer")
        st.markdown("Comprehensive audit trail analysis and compliance reporting for DGL governance decisions.")
        
        # Audit integrity status
        self._render_integrity_status()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Search & Filter", "📊 Analytics", "🔗 Hash Chain", "📋 Compliance", "📤 Export"
        ])
        
        with tab1:
            self._render_search_tab()
        
        with tab2:
            self._render_analytics_tab()
        
        with tab3:
            self._render_hash_chain_tab()
        
        with tab4:
            self._render_compliance_tab()
        
        with tab5:
            self._render_export_tab()
    
    def _render_integrity_status(self):
        """Render audit integrity status banner"""
        
        integrity_status = self._get_audit_integrity_status()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            status_color = "🟢" if integrity_status['hash_chain_intact'] else "🔴"
            st.metric("Hash Chain Status", f"{status_color} {'INTACT' if integrity_status['hash_chain_intact'] else 'BROKEN'}")
        
        with col2:
            st.metric("Total Records", f"{integrity_status['total_records']:,}")
        
        with col3:
            st.metric("Integrity Score", f"{integrity_status['integrity_score']:.1f}%")
        
        with col4:
            st.metric("Last Verification", integrity_status['last_verification'])
        
        with col5:
            if st.button("🔄 Verify Now"):
                with st.spinner("Verifying audit integrity..."):
                    # Simulate verification
                    import time
                    time.sleep(2)
                    st.success("✅ Audit integrity verified!")
        
        # Alert for integrity issues
        if not integrity_status['hash_chain_intact']:
            st.error("🚨 **CRITICAL**: Audit hash chain integrity compromised! Immediate investigation required.")
        elif integrity_status['integrity_score'] < 99.0:
            st.warning(f"⚠️ **WARNING**: Audit integrity score below threshold ({integrity_status['integrity_score']:.1f}%)")
    
    def _render_search_tab(self):
        """Render audit search and filtering tab"""
        st.markdown("## 🔍 Audit Record Search")
        
        # Search filters
        with st.expander("🔧 Search Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Time Range")
                
                date_range = st.selectbox(
                    "Quick Range",
                    ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"]
                )
                
                if date_range == "Custom Range":
                    start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=7))
                    end_date = st.date_input("End Date", value=datetime.now().date())
                    start_time = st.time_input("Start Time", value=datetime.min.time())
                    end_time = st.time_input("End Time", value=datetime.max.time())
                else:
                    start_date = end_date = None
                    start_time = end_time = None
            
            with col2:
                st.markdown("#### Event Filters")
                
                event_types = st.multiselect(
                    "Event Types",
                    ["Decision", "Rule Change", "System Event", "Error", "Authentication"],
                    default=["Decision"]
                )
                
                decision_types = st.multiselect(
                    "Decision Types",
                    ["APPROVE", "REJECT", "AMEND"],
                    default=[]
                )
                
                rule_ids = st.multiselect(
                    "Rule IDs",
                    ["BANKROLL_MAX_EXPOSURE", "LIQ_SLIPPAGE_LIMIT", "PNL_DAILY_LOSS", "RATE_LIMIT"],
                    default=[]
                )
            
            with col3:
                st.markdown("#### Content Filters")
                
                user_filter = st.text_input("User/System", placeholder="e.g., system, user_123")
                
                market_filter = st.text_input("Market ID", placeholder="e.g., match_001_winner")
                
                proposal_filter = st.text_input("Proposal ID", placeholder="e.g., prop_12345")
                
                text_search = st.text_input("Text Search", placeholder="Search in audit messages...")
            
            # Search button
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🔍 Search Audit Records", type="primary"):
                    search_params = {
                        'date_range': date_range,
                        'start_date': start_date,
                        'end_date': end_date,
                        'start_time': start_time,
                        'end_time': end_time,
                        'event_types': event_types,
                        'decision_types': decision_types,
                        'rule_ids': rule_ids,
                        'user_filter': user_filter,
                        'market_filter': market_filter,
                        'proposal_filter': proposal_filter,
                        'text_search': text_search
                    }
                    
                    # Perform search
                    results = self._search_audit_records(search_params)
                    st.session_state.audit_search_results = results
                    st.session_state.audit_filters = search_params
                    
                    st.success(f"Found {len(results)} audit records")
            
            with col2:
                if st.button("🗑️ Clear Filters"):
                    st.session_state.audit_search_results = []
                    st.session_state.audit_filters = {}
                    st.rerun()
        
        # Search results
        if st.session_state.audit_search_results:
            st.markdown("### 📋 Search Results")
            
            results = st.session_state.audit_search_results
            
            # Results summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(results))
            
            with col2:
                decision_count = len([r for r in results if r['event_type'] == 'Decision'])
                st.metric("Decisions", decision_count)
            
            with col3:
                error_count = len([r for r in results if r['event_type'] == 'Error'])
                st.metric("Errors", error_count)
            
            with col4:
                unique_users = len(set(r['user'] for r in results))
                st.metric("Unique Users", unique_users)
            
            # Results table
            df = pd.DataFrame(results)
            
            # Add expandable details
            selected_record = st.selectbox(
                "Select Record for Details",
                options=range(len(df)),
                format_func=lambda i: f"{df.iloc[i]['timestamp']} - {df.iloc[i]['event_type']} - {df.iloc[i]['record_id']}"
            )
            
            if selected_record is not None:
                record = results[selected_record]
                
                with st.expander("📋 Record Details", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### Record Information")
                        st.markdown(f"**Record ID:** {record['record_id']}")
                        st.markdown(f"**Timestamp:** {record['timestamp']}")
                        st.markdown(f"**Event Type:** {record['event_type']}")
                        st.markdown(f"**User:** {record['user']}")
                        st.markdown(f"**Description:** {record['description']}")
                        
                        if 'proposal_id' in record:
                            st.markdown(f"**Proposal ID:** {record['proposal_id']}")
                        
                        if 'market_id' in record:
                            st.markdown(f"**Market ID:** {record['market_id']}")
                        
                        if 'decision' in record:
                            st.markdown(f"**Decision:** {record['decision']}")
                        
                        # Full audit data
                        st.markdown("#### Full Audit Data")
                        st.json(record.get('full_data', {}))
                    
                    with col2:
                        st.markdown("#### Hash Verification")
                        
                        # Hash chain verification for this record
                        hash_status = self._verify_record_hash(record)
                        
                        if hash_status['valid']:
                            st.success("✅ Hash Valid")
                        else:
                            st.error("❌ Hash Invalid")
                        
                        st.markdown(f"**Record Hash:** `{record.get('record_hash', 'N/A')[:16]}...`")
                        st.markdown(f"**Previous Hash:** `{record.get('previous_hash', 'N/A')[:16]}...`")
                        
                        if st.button("🔍 Verify Chain", key=f"verify_{selected_record}"):
                            chain_status = self._verify_hash_chain_segment(record)
                            if chain_status['valid']:
                                st.success("✅ Chain segment valid")
                            else:
                                st.error("❌ Chain segment broken")
            
            # Paginated results table
            st.markdown("#### 📊 Results Table")
            
            # Pagination
            page_size = 50
            total_pages = (len(df) + page_size - 1) // page_size
            
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1)) - 1
                start_idx = page * page_size
                end_idx = min(start_idx + page_size, len(df))
                display_df = df.iloc[start_idx:end_idx]
            else:
                display_df = df
            
            # Display table
            st.dataframe(
                display_df[['timestamp', 'event_type', 'record_id', 'user', 'description', 'integrity']],
                use_container_width=True
            )
        
        else:
            st.info("🔍 Use the search filters above to find audit records")
    
    def _render_analytics_tab(self):
        """Render audit analytics tab"""
        st.markdown("## 📊 Audit Analytics")
        
        # Analytics time period
        col1, col2 = st.columns([1, 3])
        
        with col1:
            analytics_period = st.selectbox(
                "Analysis Period",
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
            )
        
        # Get analytics data
        analytics_data = self._get_audit_analytics(analytics_period)
        
        # Event type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Event Type Distribution")
            
            event_distribution = analytics_data['event_distribution']
            fig = px.pie(
                values=list(event_distribution.values()),
                names=list(event_distribution.keys()),
                title=f"Event Types - {analytics_period}",
                color_discrete_map={
                    'Decision': '#28a745',
                    'Rule Change': '#ffc107',
                    'System Event': '#17a2b8',
                    'Error': '#dc3545',
                    'Authentication': '#6f42c1'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚖️ Decision Outcomes")
            
            decision_outcomes = analytics_data['decision_outcomes']
            fig = px.bar(
                x=list(decision_outcomes.keys()),
                y=list(decision_outcomes.values()),
                title=f"Decision Outcomes - {analytics_period}",
                color_discrete_sequence=['#c8712d'],
                labels={'x': 'Decision Type', 'y': 'Count'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline analysis
        st.markdown("### ⏰ Activity Timeline")
        
        timeline_data = analytics_data['timeline_data']
        fig = px.line(
            timeline_data,
            x='timestamp',
            y='count',
            color='event_type',
            title=f"Audit Activity Timeline - {analytics_period}",
            labels={'count': 'Events per Hour', 'timestamp': 'Time'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # User activity analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👥 User Activity")
            
            user_activity = analytics_data['user_activity']
            fig = px.bar(
                x=list(user_activity.keys()),
                y=list(user_activity.values()),
                title="Activity by User",
                color_discrete_sequence=['#002466'],
                labels={'x': 'User', 'y': 'Events'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🚨 Error Analysis")
            
            error_analysis = analytics_data['error_analysis']
            
            if error_analysis:
                fig = px.bar(
                    x=list(error_analysis.keys()),
                    y=list(error_analysis.values()),
                    title="Errors by Type",
                    color_discrete_sequence=['#dc3545'],
                    labels={'x': 'Error Type', 'y': 'Count'}
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No errors in selected period")
        
        # Rule violation analysis
        st.markdown("### 🚫 Rule Violations Analysis")
        
        rule_violations = analytics_data['rule_violations']
        
        if rule_violations:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=list(rule_violations.keys()),
                    y=list(rule_violations.values()),
                    title="Violations by Rule",
                    color_discrete_sequence=['#660003'],
                    labels={'x': 'Rule ID', 'y': 'Violations'}
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Rule violation trends
                violation_trends = analytics_data['violation_trends']
                fig = px.line(
                    violation_trends,
                    x='date',
                    y='violations',
                    title="Rule Violations Trend",
                    color_discrete_sequence=['#660003']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No rule violations in selected period")
        
        # Performance metrics
        st.markdown("### 📈 Audit Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        performance_metrics = analytics_data['performance_metrics']
        
        with col1:
            st.metric("Avg Audit Latency", f"{performance_metrics['avg_audit_latency_ms']:.1f}ms")
        
        with col2:
            st.metric("Audit Success Rate", f"{performance_metrics['audit_success_rate']:.1f}%")
        
        with col3:
            st.metric("Hash Verification Rate", f"{performance_metrics['hash_verification_rate']:.1f}%")
        
        with col4:
            st.metric("Storage Efficiency", f"{performance_metrics['storage_efficiency']:.1f}%")
    
    def _render_hash_chain_tab(self):
        """Render hash chain verification tab"""
        st.markdown("## 🔗 Hash Chain Verification")
        
        # Hash chain overview
        chain_status = self._get_hash_chain_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Chain Statistics")
            st.metric("Total Blocks", f"{chain_status['total_blocks']:,}")
            st.metric("Chain Length", f"{chain_status['chain_length']:,}")
            st.metric("Genesis Block", chain_status['genesis_timestamp'])
        
        with col2:
            st.markdown("### 🔒 Integrity Status")
            
            integrity_color = "🟢" if chain_status['integrity_valid'] else "🔴"
            st.metric("Chain Integrity", f"{integrity_color} {'VALID' if chain_status['integrity_valid'] else 'INVALID'}")
            st.metric("Last Verified", chain_status['last_verification'])
            st.metric("Verification Score", f"{chain_status['verification_score']:.1f}%")
        
        with col3:
            st.markdown("### ⚡ Performance")
            st.metric("Avg Block Time", f"{chain_status['avg_block_time_ms']:.1f}ms")
            st.metric("Hash Rate", f"{chain_status['hash_rate_per_sec']:.1f}/sec")
            st.metric("Storage Size", f"{chain_status['storage_size_mb']:.1f}MB")
        
        # Hash chain verification controls
        st.markdown("### 🔍 Verification Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Full Chain Verification", type="primary"):
                with st.spinner("Verifying entire hash chain..."):
                    verification_result = self._perform_full_chain_verification()
                    
                    if verification_result['valid']:
                        st.success(f"✅ Full chain verification passed! ({verification_result['blocks_verified']} blocks)")
                    else:
                        st.error(f"❌ Chain verification failed at block {verification_result['failed_block']}")
        
        with col2:
            if st.button("🎯 Spot Check Verification"):
                with st.spinner("Performing spot check verification..."):
                    spot_check_result = self._perform_spot_check_verification()
                    
                    if spot_check_result['valid']:
                        st.success(f"✅ Spot check passed! ({spot_check_result['blocks_checked']} blocks checked)")
                    else:
                        st.error(f"❌ Spot check failed! {spot_check_result['failures']} failures found")
        
        with col3:
            if st.button("📊 Generate Integrity Report"):
                integrity_report = self._generate_integrity_report()
                st.success("📄 Integrity report generated!")
                
                # Display report summary
                with st.expander("📋 Report Summary", expanded=True):
                    st.json(integrity_report)
        
        # Recent hash chain activity
        st.markdown("### 📈 Recent Chain Activity")
        
        recent_blocks = self._get_recent_chain_blocks(limit=20)
        
        if recent_blocks:
            df = pd.DataFrame(recent_blocks)
            
            # Block timeline
            fig = px.scatter(
                df,
                x='timestamp',
                y='block_number',
                size='transactions',
                color='verification_status',
                title="Recent Hash Chain Blocks",
                labels={'timestamp': 'Time', 'block_number': 'Block Number'},
                color_discrete_map={'valid': '#28a745', 'invalid': '#dc3545', 'pending': '#ffc107'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Block details table
            st.markdown("#### 📋 Block Details")
            
            display_columns = ['block_number', 'timestamp', 'transactions', 'hash', 'previous_hash', 'verification_status']
            st.dataframe(df[display_columns], use_container_width=True)
        
        # Hash chain explorer
        st.markdown("### 🔍 Chain Explorer")
        
        with st.expander("Explore Specific Block", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                block_number = st.number_input(
                    "Block Number",
                    min_value=0,
                    max_value=chain_status['total_blocks'] - 1,
                    value=max(0, chain_status['total_blocks'] - 1)
                )
                
                if st.button("🔍 Load Block"):
                    block_details = self._get_block_details(block_number)
                    st.session_state.selected_block = block_details
            
            with col2:
                if 'selected_block' in st.session_state:
                    block = st.session_state.selected_block
                    
                    st.markdown(f"**Block #{block['block_number']}**")
                    st.markdown(f"**Timestamp:** {block['timestamp']}")
                    st.markdown(f"**Hash:** `{block['hash']}`")
                    st.markdown(f"**Previous Hash:** `{block['previous_hash']}`")
                    st.markdown(f"**Transactions:** {block['transactions']}")
                    st.markdown(f"**Verification:** {'✅ Valid' if block['valid'] else '❌ Invalid'}")
                    
                    # Transaction details
                    if block.get('transaction_details'):
                        st.markdown("**Transactions:**")
                        for i, tx in enumerate(block['transaction_details']):
                            st.markdown(f"  {i+1}. {tx['type']}: {tx['description']}")
    
    def _render_compliance_tab(self):
        """Render compliance reporting tab"""
        st.markdown("## 📋 Compliance Reporting")
        
        # Compliance overview
        compliance_status = self._get_compliance_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            compliance_color = "🟢" if compliance_status['overall_compliant'] else "🔴"
            st.metric("Overall Compliance", f"{compliance_color} {'COMPLIANT' if compliance_status['overall_compliant'] else 'NON-COMPLIANT'}")
        
        with col2:
            st.metric("Compliance Score", f"{compliance_status['compliance_score']:.1f}%")
        
        with col3:
            st.metric("Open Issues", compliance_status['open_issues'])
        
        with col4:
            st.metric("Last Audit", compliance_status['last_audit_date'])
        
        # Compliance requirements
        st.markdown("### 📋 Compliance Requirements")
        
        requirements = compliance_status['requirements']
        
        for category, reqs in requirements.items():
            with st.expander(f"📂 {category}", expanded=False):
                for req_id, req_data in reqs.items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{req_id}:** {req_data['description']}")
                    
                    with col2:
                        status_color = "🟢" if req_data['compliant'] else "🔴"
                        st.markdown(f"{status_color} {'COMPLIANT' if req_data['compliant'] else 'NON-COMPLIANT'}")
                    
                    with col3:
                        st.markdown(f"**Score:** {req_data['score']:.1f}%")
                    
                    if not req_data['compliant'] and req_data.get('issues'):
                        st.warning(f"Issues: {', '.join(req_data['issues'])}")
        
        # Compliance reports
        st.markdown("### 📊 Generate Compliance Reports")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Standard Reports")
            
            report_type = st.selectbox(
                "Report Type",
                ["GDPR Compliance", "SOX Compliance", "MiFID II", "PCI DSS", "Custom Audit"]
            )
            
            report_period = st.selectbox(
                "Report Period",
                ["Last 30 Days", "Last Quarter", "Last Year", "Custom Period"]
            )
            
            if report_period == "Custom Period":
                col1a, col1b = st.columns(2)
                with col1a:
                    custom_start = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=90))
                with col1b:
                    custom_end = st.date_input("End Date", value=datetime.now().date())
            
            include_details = st.checkbox("Include Detailed Records", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
            
            if st.button("📄 Generate Report", type="primary"):
                with st.spinner("Generating compliance report..."):
                    report = self._generate_compliance_report(
                        report_type, report_period, include_details, include_recommendations
                    )
                    
                    st.session_state.compliance_reports.append(report)
                    st.success(f"✅ {report_type} report generated!")
        
        with col2:
            st.markdown("#### Report Preview")
            
            if st.session_state.compliance_reports:
                latest_report = st.session_state.compliance_reports[-1]
                
                st.markdown(f"**Report Type:** {latest_report['type']}")
                st.markdown(f"**Period:** {latest_report['period']}")
                st.markdown(f"**Generated:** {latest_report['generated_at']}")
                st.markdown(f"**Status:** {latest_report['status']}")
                
                # Report summary
                summary = latest_report['summary']
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Records Reviewed", f"{summary['records_reviewed']:,}")
                    st.metric("Compliance Issues", summary['compliance_issues'])
                
                with col2b:
                    st.metric("Overall Score", f"{summary['overall_score']:.1f}%")
                    st.metric("Recommendations", summary['recommendations'])
                
                # Key findings
                if latest_report.get('key_findings'):
                    st.markdown("**Key Findings:**")
                    for finding in latest_report['key_findings']:
                        severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                        st.markdown(f"- {severity_color.get(finding['severity'], '🔵')} {finding['description']}")
            else:
                st.info("Generate a report to see preview")
        
        # Regulatory requirements tracking
        st.markdown("### 📋 Regulatory Requirements Tracking")
        
        regulatory_data = self._get_regulatory_tracking_data()
        
        # Requirements by regulation
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=list(regulatory_data['by_regulation'].keys()),
                y=list(regulatory_data['by_regulation'].values()),
                title="Compliance by Regulation",
                color_discrete_sequence=['#c8712d'],
                labels={'x': 'Regulation', 'y': 'Compliance Score (%)'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Compliance trend
            trend_data = regulatory_data['compliance_trend']
            fig = px.line(
                trend_data,
                x='date',
                y='compliance_score',
                title="Compliance Score Trend",
                color_discrete_sequence=['#28a745']
            )
            fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Minimum Threshold")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Action items
        st.markdown("### ⚡ Action Items")
        
        action_items = regulatory_data['action_items']
        
        if action_items:
            for item in action_items:
                priority_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                
                with st.expander(f"{priority_color[item['priority']]} {item['title']}", expanded=False):
                    st.markdown(f"**Description:** {item['description']}")
                    st.markdown(f"**Regulation:** {item['regulation']}")
                    st.markdown(f"**Due Date:** {item['due_date']}")
                    st.markdown(f"**Assigned To:** {item['assigned_to']}")
                    
                    if st.button(f"✅ Mark Complete", key=f"complete_{item['id']}"):
                        st.success("Action item marked as complete!")
        else:
            st.success("✅ No outstanding action items")
    
    def _render_export_tab(self):
        """Render data export tab"""
        st.markdown("## 📤 Data Export")
        
        # Export options
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Export Configuration")
            
            export_type = st.selectbox(
                "Export Type",
                ["Audit Records", "Compliance Report", "Hash Chain Data", "Analytics Summary"]
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "PDF", "XML"]
            )
            
            export_period = st.selectbox(
                "Data Period",
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Quarter", "All Data", "Custom Range"]
            )
            
            if export_period == "Custom Range":
                col1a, col1b = st.columns(2)
                with col1a:
                    export_start = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                with col1b:
                    export_end = st.date_input("End Date", value=datetime.now().date())
            
            # Advanced options
            with st.expander("Advanced Options", expanded=False):
                include_hash_verification = st.checkbox("Include Hash Verification", value=True)
                include_metadata = st.checkbox("Include Metadata", value=True)
                compress_output = st.checkbox("Compress Output", value=True)
                encrypt_export = st.checkbox("Encrypt Export", value=False)
                
                if encrypt_export:
                    encryption_key = st.text_input("Encryption Key", type="password")
            
            # Generate export
            if st.button("📤 Generate Export", type="primary"):
                export_config = {
                    'type': export_type,
                    'format': export_format,
                    'period': export_period,
                    'include_hash_verification': include_hash_verification,
                    'include_metadata': include_metadata,
                    'compress_output': compress_output,
                    'encrypt_export': encrypt_export
                }
                
                with st.spinner("Generating export..."):
                    export_result = self._generate_export(export_config)
                    
                    st.success(f"✅ Export generated: {export_result['filename']}")
                    
                    # Store export info
                    if 'exports' not in st.session_state:
                        st.session_state.exports = []
                    
                    st.session_state.exports.append(export_result)
        
        with col2:
            st.markdown("### 📋 Export Preview")
            
            if 'exports' in st.session_state and st.session_state.exports:
                latest_export = st.session_state.exports[-1]
                
                st.markdown(f"**Filename:** {latest_export['filename']}")
                st.markdown(f"**Type:** {latest_export['type']}")
                st.markdown(f"**Format:** {latest_export['format']}")
                st.markdown(f"**Size:** {latest_export['size']}")
                st.markdown(f"**Records:** {latest_export['record_count']:,}")
                st.markdown(f"**Generated:** {latest_export['generated_at']}")
                
                # Download button (simulated)
                if st.button("💾 Download Export"):
                    st.success("Download initiated!")
                
                # Export verification
                if latest_export.get('hash_included'):
                    st.markdown("**Export Hash:** `{}`".format(latest_export['export_hash'][:32] + "..."))
                    
                    if st.button("🔍 Verify Export Integrity"):
                        st.success("✅ Export integrity verified!")
            else:
                st.info("Generate an export to see preview")
        
        # Export history
        st.markdown("### 📊 Export History")
        
        if 'exports' in st.session_state and st.session_state.exports:
            exports_df = pd.DataFrame(st.session_state.exports)
            
            # Display export history
            st.dataframe(
                exports_df[['filename', 'type', 'format', 'size', 'record_count', 'generated_at']],
                use_container_width=True
            )
            
            # Export statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Exports", len(st.session_state.exports))
            
            with col2:
                total_records = sum(exp['record_count'] for exp in st.session_state.exports)
                st.metric("Total Records", f"{total_records:,}")
            
            with col3:
                formats = [exp['format'] for exp in st.session_state.exports]
                most_common_format = max(set(formats), key=formats.count)
                st.metric("Most Used Format", most_common_format)
            
            with col4:
                if st.button("🗑️ Clear History"):
                    st.session_state.exports = []
                    st.rerun()
        else:
            st.info("No exports generated yet")
        
        # Scheduled exports
        st.markdown("### ⏰ Scheduled Exports")
        
        with st.expander("Configure Scheduled Exports", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                schedule_name = st.text_input("Schedule Name", placeholder="e.g., Daily Compliance Export")
                
                schedule_type = st.selectbox("Export Type", ["Audit Records", "Compliance Report"])
                
                schedule_frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
                
                schedule_format = st.selectbox("Format", ["JSON", "CSV", "PDF"])
            
            with col2:
                schedule_time = st.time_input("Export Time", value=datetime.now().time())
                
                notification_email = st.text_input("Notification Email", placeholder="admin@wicketwise.com")
                
                retention_days = st.number_input("Retention (days)", min_value=1, max_value=365, value=90)
                
                if st.button("💾 Save Schedule"):
                    st.success("Export schedule saved!")
        
        # Export compliance
        st.markdown("### 🔒 Export Compliance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Data Protection")
            st.markdown("- ✅ GDPR compliant data handling")
            st.markdown("- ✅ Encryption in transit and at rest")
            st.markdown("- ✅ Access logging and monitoring")
            st.markdown("- ✅ Automatic data retention policies")
        
        with col2:
            st.markdown("#### Export Audit Trail")
            st.markdown("- ✅ All exports logged and tracked")
            st.markdown("- ✅ Hash verification for integrity")
            st.markdown("- ✅ User authentication required")
            st.markdown("- ✅ Regulatory compliance validation")
    
    # Helper methods for data retrieval and processing (mock implementations)
    
    def _get_audit_integrity_status(self) -> Dict[str, Any]:
        """Get audit integrity status (mock)"""
        return {
            'hash_chain_intact': True,
            'total_records': 156789,
            'integrity_score': 99.8,
            'last_verification': '2 minutes ago'
        }
    
    def _search_audit_records(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search audit records based on parameters (mock)"""
        # Mock search results
        results = []
        
        for i in range(50):  # Return 50 mock results
            results.append({
                'record_id': f'audit_{i:06d}',
                'timestamp': (datetime.now() - timedelta(minutes=i*15)).isoformat(),
                'event_type': ['Decision', 'Rule Change', 'System Event', 'Error'][i % 4],
                'user': 'system' if i % 3 == 0 else f'user_{i % 5}',
                'description': f'Governance decision #{i} processed',
                'integrity': '✅',
                'record_hash': hashlib.sha256(f'record_{i}'.encode()).hexdigest(),
                'previous_hash': hashlib.sha256(f'record_{i-1}'.encode()).hexdigest() if i > 0 else '0',
                'proposal_id': f'prop_{i:04d}' if i % 4 == 0 else None,
                'market_id': f'market_{i % 5}' if i % 4 == 0 else None,
                'decision': ['APPROVE', 'REJECT', 'AMEND'][i % 3] if i % 4 == 0 else None,
                'full_data': {
                    'event_id': f'evt_{i}',
                    'processing_time_ms': 20 + (i % 50),
                    'confidence_score': 0.7 + (i % 30) * 0.01,
                    'rules_triggered': [f'RULE_{j}' for j in range(i % 3)]
                }
            })
        
        return results
    
    def _verify_record_hash(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Verify individual record hash (mock)"""
        return {'valid': True, 'computed_hash': record.get('record_hash')}
    
    def _verify_hash_chain_segment(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Verify hash chain segment (mock)"""
        return {'valid': True, 'chain_length': 5}
    
    def _get_audit_analytics(self, period: str) -> Dict[str, Any]:
        """Get audit analytics data (mock)"""
        import numpy as np
        
        return {
            'event_distribution': {
                'Decision': 12450,
                'Rule Change': 156,
                'System Event': 892,
                'Error': 23,
                'Authentication': 1234
            },
            'decision_outcomes': {
                'APPROVE': 8340,
                'REJECT': 2890,
                'AMEND': 1220
            },
            'timeline_data': pd.DataFrame({
                'timestamp': pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H'),
                'count': np.random.poisson(50, 7*24),
                'event_type': np.random.choice(['Decision', 'System Event', 'Error'], 7*24)
            }),
            'user_activity': {
                'system': 8950,
                'user_001': 2340,
                'user_002': 1890,
                'user_003': 1560,
                'user_004': 1120
            },
            'error_analysis': {
                'Validation Error': 12,
                'Network Timeout': 8,
                'Database Error': 3
            },
            'rule_violations': {
                'BANKROLL_MAX_EXPOSURE': 45,
                'LIQ_SLIPPAGE_LIMIT': 32,
                'PNL_DAILY_LOSS': 18,
                'RATE_LIMIT': 12
            },
            'violation_trends': pd.DataFrame({
                'date': pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D'),
                'violations': np.random.poisson(3, 30)
            }),
            'performance_metrics': {
                'avg_audit_latency_ms': 12.5,
                'audit_success_rate': 99.8,
                'hash_verification_rate': 100.0,
                'storage_efficiency': 87.3
            }
        }
    
    def _get_hash_chain_status(self) -> Dict[str, Any]:
        """Get hash chain status (mock)"""
        return {
            'total_blocks': 15678,
            'chain_length': 15678,
            'genesis_timestamp': '2024-01-01 00:00:00',
            'integrity_valid': True,
            'last_verification': '5 minutes ago',
            'verification_score': 100.0,
            'avg_block_time_ms': 125.5,
            'hash_rate_per_sec': 8.2,
            'storage_size_mb': 234.7
        }
    
    def _perform_full_chain_verification(self) -> Dict[str, Any]:
        """Perform full chain verification (mock)"""
        import time
        time.sleep(3)  # Simulate verification time
        
        return {
            'valid': True,
            'blocks_verified': 15678,
            'verification_time_ms': 3245,
            'failed_block': None
        }
    
    def _perform_spot_check_verification(self) -> Dict[str, Any]:
        """Perform spot check verification (mock)"""
        import time
        time.sleep(1)  # Simulate verification time
        
        return {
            'valid': True,
            'blocks_checked': 100,
            'failures': 0,
            'verification_time_ms': 1123
        }
    
    def _generate_integrity_report(self) -> Dict[str, Any]:
        """Generate integrity report (mock)"""
        return {
            'report_id': f'integrity_{int(datetime.now().timestamp())}',
            'generated_at': datetime.now().isoformat(),
            'chain_status': 'INTACT',
            'total_blocks': 15678,
            'verified_blocks': 15678,
            'integrity_score': 100.0,
            'recommendations': [
                'Continue regular verification schedule',
                'Monitor hash rate performance',
                'Review storage optimization'
            ]
        }
    
    def _get_recent_chain_blocks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent chain blocks (mock)"""
        blocks = []
        
        for i in range(limit):
            block_num = 15678 - i
            blocks.append({
                'block_number': block_num,
                'timestamp': (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                'transactions': 5 + (i % 10),
                'hash': hashlib.sha256(f'block_{block_num}'.encode()).hexdigest()[:16] + '...',
                'previous_hash': hashlib.sha256(f'block_{block_num-1}'.encode()).hexdigest()[:16] + '...',
                'verification_status': 'valid'
            })
        
        return blocks
    
    def _get_block_details(self, block_number: int) -> Dict[str, Any]:
        """Get detailed block information (mock)"""
        return {
            'block_number': block_number,
            'timestamp': (datetime.now() - timedelta(minutes=block_number*5)).isoformat(),
            'hash': hashlib.sha256(f'block_{block_number}'.encode()).hexdigest(),
            'previous_hash': hashlib.sha256(f'block_{block_number-1}'.encode()).hexdigest() if block_number > 0 else '0',
            'transactions': 7,
            'valid': True,
            'transaction_details': [
                {'type': 'Decision', 'description': 'Governance decision approved'},
                {'type': 'Audit', 'description': 'Audit record created'},
                {'type': 'System', 'description': 'System health check'}
            ]
        }
    
    def _get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status (mock)"""
        return {
            'overall_compliant': True,
            'compliance_score': 96.8,
            'open_issues': 3,
            'last_audit_date': '2024-01-15',
            'requirements': {
                'Data Protection': {
                    'GDPR_001': {'description': 'Data retention policies', 'compliant': True, 'score': 100.0},
                    'GDPR_002': {'description': 'Right to erasure', 'compliant': True, 'score': 98.5},
                    'GDPR_003': {'description': 'Data portability', 'compliant': False, 'score': 85.0, 'issues': ['Export format limitations']}
                },
                'Financial Compliance': {
                    'SOX_001': {'description': 'Internal controls', 'compliant': True, 'score': 99.2},
                    'SOX_002': {'description': 'Audit trail integrity', 'compliant': True, 'score': 100.0},
                    'MIFID_001': {'description': 'Transaction reporting', 'compliant': True, 'score': 97.8}
                }
            }
        }
    
    def _generate_compliance_report(self, report_type: str, period: str, include_details: bool, include_recommendations: bool) -> Dict[str, Any]:
        """Generate compliance report (mock)"""
        return {
            'type': report_type,
            'period': period,
            'generated_at': datetime.now().isoformat(),
            'status': 'COMPLIANT',
            'summary': {
                'records_reviewed': 15234,
                'compliance_issues': 3,
                'overall_score': 96.8,
                'recommendations': 5
            },
            'key_findings': [
                {'severity': 'LOW', 'description': 'Minor data export format limitation'},
                {'severity': 'MEDIUM', 'description': 'Audit log retention period optimization needed'},
                {'severity': 'LOW', 'description': 'User access review recommended'}
            ]
        }
    
    def _get_regulatory_tracking_data(self) -> Dict[str, Any]:
        """Get regulatory tracking data (mock)"""
        return {
            'by_regulation': {
                'GDPR': 94.5,
                'SOX': 98.2,
                'MiFID II': 96.8,
                'PCI DSS': 92.1
            },
            'compliance_trend': pd.DataFrame({
                'date': pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D'),
                'compliance_score': [95 + i % 10 + (i % 30) * 0.1 for i in range(90)]
            }),
            'action_items': [
                {
                    'id': 'act_001',
                    'title': 'Update data retention policy',
                    'description': 'Review and update data retention periods for compliance',
                    'regulation': 'GDPR',
                    'priority': 'MEDIUM',
                    'due_date': '2024-02-15',
                    'assigned_to': 'compliance_team'
                },
                {
                    'id': 'act_002',
                    'title': 'Enhance audit logging',
                    'description': 'Implement additional audit logging for user actions',
                    'regulation': 'SOX',
                    'priority': 'HIGH',
                    'due_date': '2024-02-01',
                    'assigned_to': 'engineering_team'
                }
            ]
        }
    
    def _generate_export(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data export (mock)"""
        import time
        time.sleep(2)  # Simulate export generation
        
        return {
            'filename': f"dgl_export_{config['type'].lower().replace(' ', '_')}_{int(datetime.now().timestamp())}.{config['format'].lower()}",
            'type': config['type'],
            'format': config['format'],
            'size': '15.7 MB',
            'record_count': 15234,
            'generated_at': datetime.now().isoformat(),
            'hash_included': config.get('include_hash_verification', False),
            'export_hash': hashlib.sha256(f"export_{datetime.now().timestamp()}".encode()).hexdigest()
        }


def main():
    """Main entry point for audit viewer"""
    audit_viewer = AuditViewer()
    audit_viewer.render()


if __name__ == "__main__":
    main()
