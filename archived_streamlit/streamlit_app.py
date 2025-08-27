# Purpose: Main Streamlit application entry point for DGL UI
# Author: WicketWise AI, Last Modified: 2024

"""
DGL Streamlit Application

Main entry point for the WicketWise DGL Streamlit interface:
- Multi-page application structure
- Navigation and routing
- Authentication integration (future)
- Theme and styling
"""

import streamlit as st
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from governance_dashboard import GovernanceDashboard
from limits_manager import LimitsManager
from audit_viewer import AuditViewer
from monitoring_panel import MonitoringPanel

# Add SIM module to path
sim_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'sim')
if os.path.exists(sim_path):
    sys.path.append(sim_path)
    try:
        import ui_streamlit
        from ui_streamlit import render_simulator_tab
        SIM_AVAILABLE = True
    except ImportError as e:
        print(f"SIM import failed: {e}")
        SIM_AVAILABLE = False
else:
    SIM_AVAILABLE = False


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="WicketWise DGL",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for WicketWise branding
    st.markdown("""
    <style>
    /* WicketWise Cricket Theme */
    .main-header {
        background: linear-gradient(90deg, #c8712d 0%, #002466 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .cricket-accent { 
        color: #c8712d; 
        font-weight: bold; 
    }
    
    .bowling-accent { 
        color: #002466; 
        font-weight: bold; 
    }
    
    .wicket-accent { 
        color: #660003; 
        font-weight: bold; 
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c8712d;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .status-healthy { 
        color: #28a745; 
        font-weight: bold;
    }
    
    .status-warning { 
        color: #ffc107; 
        font-weight: bold;
    }
    
    .status-error { 
        color: #dc3545; 
        font-weight: bold;
    }
    
    /* Navigation styling */
    .nav-button {
        width: 100%;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        border: 2px solid #c8712d;
        background: white;
        color: #c8712d;
        padding: 0.75rem;
        text-align: center;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        background: #c8712d;
        color: white;
    }
    
    .nav-button.active {
        background: #c8712d;
        color: white;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: #f8f9fa;
        border-top: 3px solid #c8712d;
        text-align: center;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ WicketWise Deterministic Governance Layer</h1>
        <p>AI-Independent Safety Engine for Cricket Betting Governance</p>
        <p><em>Phi1618 Engineering - Built for Precision, Designed for Trust</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Dashboard'
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        # Navigation buttons
        pages = {
            "📊 Dashboard": "Dashboard",
            "🔧 Limits Manager": "Limits",
            "🔍 Audit Viewer": "Audit", 
            "📈 Monitoring": "Monitoring"
        }
        
        # Add Simulator tab if available
        if SIM_AVAILABLE:
            pages["🎯 Simulator"] = "Simulator"
        
        for display_name, page_key in pages.items():
            if st.button(display_name, key=f"nav_{page_key}"):
                st.session_state.current_page = page_key
        
        st.markdown("---")
        
        # System status in sidebar
        st.markdown("### 🚦 System Status")
        
        # Mock system status
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="status-healthy">🟢 Online</p>', unsafe_allow_html=True)
        with col2:
            st.markdown('<p class="cricket-accent">SHADOW</p>', unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.metric("Decisions Today", "1,247", "+156")
        st.metric("Approval Rate", "67.3%", "+2.1%")
        st.metric("Avg Response", "23.5ms", "-1.2ms")
        
        st.markdown("---")
        
        # Connection settings
        st.markdown("### 🔗 Connection")
        dgl_url = st.text_input(
            "DGL Service URL", 
            value="http://localhost:8001",
            help="URL of the DGL service"
        )
        
        if st.button("🔄 Test Connection"):
            st.success("✅ Connection successful!")
        
        st.markdown("---")
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Version:** 1.0.0  
        **Build:** G6.1.0  
        **Mode:** Shadow Testing  
        **Uptime:** 15d 4h 23m  
        
        Built with ❤️ by the  
        WicketWise Engineering Team
        """)
    
    # Main content area - route to appropriate page
    if st.session_state.current_page == 'Dashboard':
        render_dashboard_page(dgl_url)
    elif st.session_state.current_page == 'Limits':
        render_limits_page(dgl_url)
    elif st.session_state.current_page == 'Audit':
        render_audit_page(dgl_url)
    elif st.session_state.current_page == 'Monitoring':
        render_monitoring_page(dgl_url)
    elif st.session_state.current_page == 'Simulator' and SIM_AVAILABLE:
        render_simulator_page()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>WicketWise Deterministic Governance Layer</strong></p>
        <p>Ensuring responsible cricket betting through deterministic risk management</p>
        <p><em>"Think like a trader, build like your weekend bet depends on it"</em></p>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard_page(dgl_url: str):
    """Render the main dashboard page"""
    try:
        dashboard = GovernanceDashboard(dgl_url)
        dashboard.render()
    except Exception as e:
        st.error(f"Failed to load dashboard: {str(e)}")
        st.info("Please check the DGL service connection and try again.")


def render_limits_page(dgl_url: str):
    """Render the limits management page"""
    try:
        limits_manager = LimitsManager(dgl_url)
        limits_manager.render()
    except Exception as e:
        st.error(f"Failed to load limits manager: {str(e)}")
        st.info("Please check the DGL service connection and try again.")


def render_audit_page(dgl_url: str):
    """Render the audit viewer page"""
    try:
        audit_viewer = AuditViewer(dgl_url)
        audit_viewer.render()
    except Exception as e:
        st.error(f"Failed to load audit viewer: {str(e)}")
        st.info("Please check the DGL service connection and try again.")


def render_monitoring_page(dgl_url: str):
    """Render the monitoring panel page"""
    try:
        monitoring_panel = MonitoringPanel(dgl_url)
        monitoring_panel.render()
    except Exception as e:
        st.error(f"Failed to load monitoring panel: {str(e)}")
        st.info("Please check the DGL service connection and try again.")


def render_simulator_page():
    """Render the simulator page"""
    if SIM_AVAILABLE:
        try:
            render_simulator_tab()
        except Exception as e:
            st.error(f"Failed to load simulator: {str(e)}")
            st.info("Please check the SIM system installation and try again.")
    else:
        st.warning("🎯 Simulator module not available")
        st.info("The WicketWise Simulator & Market Replay (SIM) system is not installed or accessible.")


if __name__ == "__main__":
    main()
