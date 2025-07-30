# Purpose: Implements the Phi 1.618 brand theme for the Streamlit UI.
# Author: Gemini, Last Modified: 2024-07-19

import streamlit as st

def set_streamlit_theme():
    """
    Injects modern, sophisticated CSS theme with excellent readability and professional aesthetics.
    """
    css = """
<style>
    /* --- Font Imports --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* --- Modern Color Variables --- */
    :root {
        --bg-primary: #0F1419;
        --bg-secondary: #1A2332;
        --bg-tertiary: #252B37;
        --accent-blue: #4A90E2;
        --accent-teal: #50C878;
        --text-primary: #FFFFFF;
        --text-secondary: #B8BCC8;
        --text-muted: #8A8D93;
        --border-subtle: #3A3F4B;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
        --hover: rgba(74, 144, 226, 0.1);
    }

    /* --- Base Layout & Typography --- */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0A0E13 100%);
        color: var(--text-primary);
    }

    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        font-weight: 400;
        line-height: 1.6;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.3;
        margin-bottom: 1rem;
    }

    h1 { font-size: 2.5rem; font-weight: 700; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.5rem; }
    
    p, .stMarkdown p {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* --- Sidebar Styling --- */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-subtle);
        box-shadow: 4px 0 12px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stSidebar"] .stMarkdown h1 {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* --- Navigation Menu Styling --- */
    .nav-link {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        text-decoration: none;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }

    .nav-link:hover {
        background: var(--hover);
        color: var(--accent-blue);
        border-color: var(--accent-blue);
        transform: translateX(4px);
    }

    .nav-link.active {
        background: var(--accent-blue);
        color: var(--text-primary);
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }

    /* --- Main Content Area --- */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* --- Cards & Containers --- */
    .stContainer, [data-testid="column"] {
        background: var(--bg-secondary);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--border-subtle);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }

    /* --- Input Fields --- */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2) !important;
    }

    /* --- Buttons --- */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), #357ABD);
        color: var(--text-primary);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #357ABD, var(--accent-blue));
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(74, 144, 226, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }

    /* --- Success/Warning/Error States --- */
    .success-button > button {
        background: linear-gradient(135deg, var(--success), #059669);
    }

    .warning-button > button {
        background: linear-gradient(135deg, var(--warning), #D97706);
    }

    .error-button > button {
        background: linear-gradient(135deg, var(--error), #DC2626);
    }

    /* --- File Uploader --- */
    [data-testid="stFileUploader"] {
        background: var(--bg-tertiary);
        border: 2px dashed var(--border-subtle);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-blue);
        background: rgba(74, 144, 226, 0.05);
    }

    /* --- Metrics & Stats --- */
    [data-testid="metric-container"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* --- Progress Bars --- */
    .stProgress .st-bo {
        background-color: var(--bg-tertiary);
        border-radius: 10px;
    }

    .stProgress .st-bp {
        background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue));
        border-radius: 10px;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        border-radius: 6px;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-blue);
        color: var(--text-primary);
        font-weight: 600;
    }

    /* --- Code Blocks --- */
    .stCode {
        font-family: 'JetBrains Mono', monospace;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
    }

    /* --- Scrollbars --- */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-subtle);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }

    /* --- Remove default Streamlit branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Example of how to use this in your main app:
    # from ui_theme import set_streamlit_theme
    # set_streamlit_theme() 