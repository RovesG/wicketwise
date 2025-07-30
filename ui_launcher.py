# Purpose: Cricket AI Streamlit UI with a modern, user-centric design.
# Author: Phi1618 Cricket AI Team, Last Modified: 2024-12-19

import streamlit as st
import time
import pandas as pd
import io
from admin_tools import admin_tools
from chat_tools import handle_chat_query
from chat_agent import ask_llm_with_tools
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space

# Import the new theme and style modules
from ui_theme import set_streamlit_theme
from ui_style import (
    render_player_card, 
    render_win_probability_bar, 
    render_odds_panel, 
    render_chat_bubble
)

# Import match simulator
from ui_match_simulator import render_match_simulator

# Apply the theme at the start of the app
set_streamlit_theme()

# --- Page Configuration ---
st.set_page_config(
    page_title="WicketWise Cricket AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Modern UI ---
def load_css():
    st.markdown("""
        <style>
            /* --- Main App Styling --- */
            .stApp {
                background-color: #f0f2f6; /* Light grey background */
            }

            /* --- Sidebar Styling --- */
            .css-1d391kg {
                background-color: #001e3c; /* Dark blue sidebar */
                border-right: 2px solid #c8712d; /* Batting color accent */
            }
            .css-1d391kg .st-emotion-cache-16txtl3 {
                color: white;
            }
            
            /* --- Option Menu Styling --- */
            .st-emotion-cache-16txtl3 {
                color: white; /* Menu title color */
            }
            .st-emotion-cache-6qob1r {
                background-color: #003366; /* Bowling color for menu background */
            }
            .st-emotion-cache-6qob1r:hover {
                background-color: #c8712d; /* Batting color on hover */
                color: white;
            }
            .st-emotion-cache-10trblm {
                color: white;
            }
            
            /* --- Buttons --- */
            .stButton>button {
                border-radius: 20px;
                border: 1px solid #c8712d;
                background-color: #d38c55;
                color: white;
            }
            .stButton>button:hover {
                background-color: #c8712d;
                color: white;
                border: 1px solid #d38c55;
            }

        </style>
    """, unsafe_allow_html=True)

load_css()

# --- Main Application ---
def main():
    """Main application router."""
    
    # --- Sidebar Navigation ---
    with st.sidebar:
        st.title("🏏 WicketWise AI")
        add_vertical_space(1)
        
        # Modern, icon-based navigation menu
        selected = option_menu(
            menu_title=None,  # Hides the default menu title
            options=["Live Match", "Match Simulator", "Admin Panel"],
            icons=["bar-chart-line-fill", "shuffle", "sliders"],  # Bootstrap icons
            menu_icon="cast",  # Optional
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#001e3c"},
                "icon": {"color": "white", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#c8712d"},
                "nav-link-selected": {"background-color": "#003366"},
            }
        )
        
        add_vertical_space(5) # Pushes the footer down

    # --- View Routing ---
    if selected == "Live Match":
        render_live_dashboard()
    elif selected == "Match Simulator":
        render_match_simulator()
    elif selected == "Admin Panel":
        render_admin_panel()

    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.info("WicketWise Version 0.3 - Modern UI")

def render_live_dashboard():
    """Renders the main user-facing dashboard for live match analysis."""
    st.title("🏏 Live Match Analysis")
    st.markdown("Real-time analytics, player statistics, and tactical insights.")
    st.markdown("---")
    
    # Placeholder for the main dashboard content
    # We will build this out with player cards, a video player, and odds.
    st.info("Dashboard under construction. Please upload a match file to see a preview.")

    # Simplified data loader for now to power the components
    uploaded_file = st.file_uploader(
        "Upload Match Data (CSV) to Preview Dashboard",
        type=['csv'],
        help="Upload a CSV file with ball-by-ball match data to see the components."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.simulator_data = df
            
            # --- Main Dashboard Layout ---
            
            # Top Row: Player Cards
            st.subheader("Current Matchup")
            current_row = df.iloc[0] # Use first ball for preview
            render_player_card(current_row, df)
            st.markdown("---")

            # Middle Row: Video Player and Odds
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Live Feed / Replay")
                # Placeholder for video player
                st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") # Sample video
                st.caption("Video player for live feed or ball-by-ball replay.")

            with col2:
                st.subheader("Odds & Predictions")
                render_odds_panel(current_row, df)

            # Bottom Row: Chat Assistant
            st.markdown("---")
            st.subheader("Tactical Assistant")
            render_chat_interface()

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.session_state.simulator_data = None
    else:
        st.image("https://via.placeholder.com/1200x600.png?text=Upload+Match+Data+To+See+Dashboard", 
                 caption="A modern dashboard with player cards, video, and live odds will appear here.")


# (Keep existing functions: render_admin_panel, render_player_cards, render_odds_panel, etc.)
# ... existing code ...

def render_admin_panel():
    """Render the admin panel tab."""
    st.title("🔧 Admin Panel")
    st.markdown("Manage data, trigger ML pipelines, and configure the system.")
    st.markdown("---")
    
    # Create two columns: main content and sidebar
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🔧 Backend Jobs")
        st.markdown("Trigger machine learning and data processing tasks")
        
        # Create a grid of buttons (2x3)
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            # Match Alignment Button
            if st.button("🔗 Align Matches", 
                        help="Run hybrid match alignment using decimal and nvplay files",
                        use_container_width=True):
                decimal_path = st.session_state.get('path_decimal')
                nvplay_path = st.session_state.get('path_nvplay')
                openai_key = st.session_state.get('api_openai')
                
                if not decimal_path or not nvplay_path:
                    st.error("❌ Please provide both decimal and nvplay files in Configuration section")
                else:
                    with st.spinner("Running match alignment..."):
                        # Handle file upload objects vs file paths
                        if hasattr(decimal_path, 'read'):  # File upload object
                            result = admin_tools.align_matches(decimal_path, nvplay_path, openai_key)
                        else:  # File path string
                            result = admin_tools.align_matches(decimal_path, nvplay_path, openai_key)
                        st.session_state.status_messages.append(result)
                        if result.startswith("✅"):
                            st.success(result)
                        else:
                            st.error(result)
            
            # Knowledge Graph Button
            if st.button("📊 Build Knowledge Graph", 
                        help="Build cricket knowledge graph from ball-by-ball data",
                        use_container_width=True):
                with st.spinner("Building knowledge graph..."):
                    result = admin_tools.build_knowledge_graph()
                    st.session_state.status_messages.append(result)
                    if result.startswith("✅"):
                        st.success(result)
                    else:
                        st.error(result)
        
        with button_col2:
            # GNN Embeddings Button
            if st.button("🧠 Train GNN Embeddings",
                        help="Train graph neural network embeddings",
                        use_container_width=True):
                with st.spinner("Training GNN embeddings..."):
                    result = admin_tools.train_gnn_embeddings()
                    st.session_state.status_messages.append(result)
                    if result.startswith("✅"):
                        st.success(result)
                    else:
                        st.error(result)
            
            # Crickformer Model Button
            if st.button("🤖 Train Crickformer Model",
                        help="Train transformer model for cricket predictions",
                        use_container_width=True):
                with st.spinner("Training Crickformer model..."):
                    result = admin_tools.train_crickformer_model()
                    st.session_state.status_messages.append(result)
                    if result.startswith("✅"):
                        st.success(result)
                    else:
                        st.error(result)
        
        with button_col3:
            # Evaluation Button
            if st.button("📈 Run Evaluation",
                        help="Evaluate model performance on test data",
                        use_container_width=True):
                with st.spinner("Running evaluation..."):
                    result = admin_tools.run_evaluation()
                    st.session_state.status_messages.append(result)
                    if result.startswith("✅"):
                        st.success(result)
                    else:
                        st.error(result)
        
        # Configuration Section
        st.markdown("---")
        st.header("⚙️ Configuration")
        st.markdown("API keys and data file management")
        
        # Create two columns for configuration
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.subheader("🔑 API Configuration")
            
            # Betfair API Key input
            betfair_key = st.text_input(
                "Betfair API Key",
                type="password",
                placeholder="Enter your Betfair API key...",
                help="API key for Betfair betting exchange integration"
            )
            if betfair_key:
                st.session_state.api_betfair = betfair_key
            
            # OpenAI API Key input
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password", 
                placeholder="Enter your OpenAI API key...",
                help="API key for OpenAI language model integration"
            )
            if openai_key:
                st.session_state.api_openai = openai_key
        
        with config_col2:
            st.subheader("📁 Data Files")
            
            # File input method selection
            file_method = st.radio(
                "Choose file input method:",
                ["Upload Files", "File Paths"],
                help="Upload: For files <1GB | File Paths: For large files on disk"
            )
            
            if file_method == "Upload Files":
                st.info("📤 Upload limit: 1GB per file")
                
                # Decimal CSV file selector
                decimal_file = st.file_uploader(
                    "Decimal CSV File",
                    type=['csv'],
                    help="Upload the decimal odds CSV file"
                )
                if decimal_file:
                    st.session_state.path_decimal = decimal_file
                
                # NVPlay CSV file selector  
                nvplay_file = st.file_uploader(
                    "NVPlay CSV File",
                    type=['csv'],
                    help="Upload the NVPlay CSV file"
                )
                if nvplay_file:
                    st.session_state.path_nvplay = nvplay_file
                
                # Aligned Matches CSV file selector
                aligned_file = st.file_uploader(
                    "Aligned Matches CSV File",
                    type=['csv'],
                    help="Upload the aligned matches CSV file"
                )
                if aligned_file:
                    st.session_state.path_aligned = aligned_file
            
            else:  # File Paths
                st.info("📂 Enter full file paths for large files on disk")
                
                # Decimal CSV file path
                decimal_path = st.text_input(
                    "Decimal CSV File Path",
                    placeholder="/path/to/decimal_data.csv",
                    help="Full path to the decimal odds CSV file on disk"
                )
                if decimal_path:
                    st.session_state.path_decimal = decimal_path
                
                # NVPlay CSV file path
                nvplay_path = st.text_input(
                    "NVPlay CSV File Path",
                    placeholder="/path/to/nvplay_data.csv",
                    help="Full path to the NVPlay CSV file on disk"
                )
                if nvplay_path:
                    st.session_state.path_nvplay = nvplay_path
                
                # Aligned Matches CSV file path
                aligned_path = st.text_input(
                    "Aligned Matches CSV File Path",
                    placeholder="/path/to/aligned_matches.csv",
                    help="Full path to the aligned matches CSV file on disk"
                )
                if aligned_path:
                    st.session_state.path_aligned = aligned_path
        
        # Display current session state (for debugging)
        if st.checkbox("Show Configuration State (Debug)"):
            st.subheader("Current Configuration State")
            config_keys = ['api_betfair', 'api_openai', 'path_decimal', 'path_nvplay', 'path_aligned']
            for key in config_keys:
                if key in st.session_state:
                    if key.startswith('api_'):
                        st.write(f"✅ {key}: {'*' * len(str(st.session_state[key]))}")
                    else:
                        value = st.session_state[key]
                        if hasattr(value, 'name'):  # File upload object
                            st.write(f"✅ {key}: {value.name} ({value.size if hasattr(value, 'size') else 'unknown size'} bytes)")
                        else:  # File path string
                            st.write(f"✅ {key}: {str(value)}")
                else:
                    st.write(f"❌ {key}: Not set")
        
        # System Status Section
        st.markdown("---")
        st.header("📊 System Status")
        
        # Get system status
        status = admin_tools.get_system_status()
        
        # Display status in columns
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.metric("Aligned Matches", 
                     "✅ Ready" if status['aligned_matches_exist'] else "❌ Not Created")
            st.metric("Knowledge Graph", 
                     "✅ Built" if status['knowledge_graph_built'] else "❌ Not Built")
        
        with status_col2:
            st.metric("GNN Embeddings", 
                     "✅ Trained" if status['gnn_embeddings_trained'] else "❌ Not Trained")
            st.metric("Crickformer Model", 
                     "✅ Trained" if status['crickformer_trained'] else "❌ Not Trained")
        
        with status_col3:
            st.metric("System Health", 
                     f"🟢 {status['system_health'].title()}")
            st.metric("Last Evaluation", 
                     status['last_evaluation'] if status['last_evaluation'] else "❌ Never")
    
    # Sidebar for status messages
    with st.sidebar:
        st.header("📋 Status Log")
        
        # Clear button
        if st.button("Clear Log", type="secondary"):
            st.session_state.status_messages = []
            st.rerun()
        
        # Display status messages
        if st.session_state.status_messages:
            st.markdown("**Recent Actions:**")
            for i, message in enumerate(reversed(st.session_state.status_messages[-10:])):
                st.success(f"{i+1}. {message}")
        else:
            st.info("No actions performed yet.")
        
        # Debug section
        st.markdown("---")
        st.header("🐛 Debug")
        
        if st.button("Reset System", type="secondary"):
            st.session_state.status_messages = []
            st.info("System reset complete")
        
        # Show current time
        st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

def render_simulator_mode():
    """Render the simulator mode tab."""
    
    # Create main layout with sidebar for chat
    main_col, chat_col = st.columns([2, 1])
    
    with main_col:
        st.header("🎮 Cricket Match Simulator")
        st.markdown("Load and simulate cricket matches ball-by-ball")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Match Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with ball-by-ball match data"
        )
        
        if uploaded_file is not None:
            try:
                # Load the data
                df = pd.read_csv(uploaded_file)
                st.session_state.simulator_data = df
                st.success(f"✅ Loaded {len(df)} balls from match data")
                
                # Display basic file info
                st.info(f"📊 Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show column names
                st.markdown("**Available columns:**")
                st.code(", ".join(df.columns.tolist()))
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.session_state.simulator_data = None
        
        # Only show controls if data is loaded
        if st.session_state.simulator_data is not None:
            df = st.session_state.simulator_data
            max_balls = len(df)
            
            # Control panel
            st.markdown("---")
            st.subheader("🎛️ Simulator Controls")
            
            # Create two columns for basic controls
            control_col1, control_col2 = st.columns(2)
            
            with control_col1:
                # Play/Pause toggle
                if st.button("⏯️ Play/Pause", 
                            help="Toggle playback of the match",
                            use_container_width=True):
                    st.session_state.simulator_playing = not st.session_state.simulator_playing
                    if st.session_state.simulator_playing:
                        st.success("▶️ Playing...")
                    else:
                        st.info("⏸️ Paused")
            
            with control_col2:
                # Reset button
                if st.button("🔄 Reset", 
                            help="Reset to first ball",
                            use_container_width=True):
                    st.session_state.current_ball = 1
                    st.session_state.simulator_playing = False
                    st.info("🔄 Reset to ball 1")
            
            # Advanced controls
            st.markdown("### ⚡ Advanced Controls")
            
            # Create two columns for advanced controls
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                # Fast-Forward button
                if st.button("⏩ Fast-Forward (+5)", 
                            help="Advance by 5 balls",
                            use_container_width=True):
                    new_ball = min(st.session_state.current_ball + 5, max_balls)
                    st.session_state.current_ball = new_ball
                    st.info(f"⏩ Advanced to ball {new_ball}")
            
            with adv_col2:
                # Initialize autoplay state
                if 'autoplay_enabled' not in st.session_state:
                    st.session_state.autoplay_enabled = False
                
                # Autoplay checkbox
                autoplay_enabled = st.checkbox(
                    "🎬 Autoplay (1s intervals)",
                    value=st.session_state.autoplay_enabled,
                    help="Automatically advance ball every 1 second"
                )
                
                # Update autoplay state
                st.session_state.autoplay_enabled = autoplay_enabled
                
                # Handle autoplay logic
                if autoplay_enabled and st.session_state.current_ball < max_balls:
                    # Check if we should advance (simple time-based approach)
                    import time
                    current_time = time.time()
                    
                    # Initialize autoplay timer if not exists
                    if 'autoplay_last_update' not in st.session_state:
                        st.session_state.autoplay_last_update = current_time
                    
                    # Check if 1 second has passed
                    if current_time - st.session_state.autoplay_last_update >= 1.0:
                        # Advance ball
                        new_ball = min(st.session_state.current_ball + 1, max_balls)
                        st.session_state.current_ball = new_ball
                        st.session_state.autoplay_last_update = current_time
                        
                        # Only show message if we actually advanced
                        if new_ball > st.session_state.current_ball - 1:
                            st.info(f"🎬 Autoplay: Ball {new_ball}")
                        
                        # Trigger rerun to update the display
                        st.rerun()
                    else:
                        # Wait a bit and rerun to check again
                        time.sleep(0.1)
                        st.rerun()
                
                # Show autoplay status
                if autoplay_enabled:
                    if st.session_state.current_ball >= max_balls:
                        st.warning("🎬 Autoplay: Reached end of match")
                    else:
                        st.success(f"🎬 Autoplay: Active")
                else:
                    st.info("🎬 Autoplay: Disabled")
            
            # Ball slider
            st.markdown("---")
            st.subheader("🏏 Ball Selection")
            
            current_ball = st.slider(
                "Current Ball",
                min_value=1,
                max_value=max_balls,
                value=st.session_state.current_ball,
                help=f"Select ball number (1 to {max_balls})"
            )
            
            # Update session state
            st.session_state.current_ball = current_ball
            
            # Display area for current ball info
            st.markdown("---")
            st.subheader("📊 Current Ball Information")
            
            # Get current ball data
            current_row = df.iloc[current_ball - 1]  # Convert to 0-based index
            
            # Create columns for display
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("Ball Number", current_ball)
                # Try to display batter info if available
                if 'batter' in df.columns:
                    st.metric("Batter", current_row['batter'])
                elif 'batsman' in df.columns:
                    st.metric("Batsman", current_row['batsman'])
                else:
                    st.metric("Batter", "Not available")
            
            with info_col2:
                # Try to display bowler info if available
                if 'bowler' in df.columns:
                    st.metric("Bowler", current_row['bowler'])
                else:
                    st.metric("Bowler", "Not available")
                
                # Try to display over info
                if 'over' in df.columns:
                    st.metric("Over", current_row['over'])
                else:
                    st.metric("Over", "Not available")
            
            with info_col3:
                # Try to display runs info
                if 'runs' in df.columns:
                    st.metric("Runs", current_row['runs'])
                elif 'runs_off_bat' in df.columns:
                    st.metric("Runs", current_row['runs_off_bat'])
                else:
                    st.metric("Runs", "Not available")
                
                # Try to display extras
                if 'extras' in df.columns:
                    st.metric("Extras", current_row['extras'])
                else:
                    st.metric("Extras", "Not available")
            
            # Player Cards Section
            st.markdown("---")
            st.subheader("👥 Player Cards")
            render_player_card(current_row, df)
            
            # Odds Panel Section
            st.markdown("---")
            render_odds_panel(current_row, df)
            
            # Show raw data for current ball
            st.markdown("---")
            st.subheader("📋 Raw Ball Data")
            
            # Display as a formatted table
            ball_data = current_row.to_dict()
            df_display = pd.DataFrame([ball_data])
            st.dataframe(df_display, use_container_width=True)
            
            # Show playback status
            if st.session_state.simulator_playing:
                st.success("▶️ Simulator is playing")
            else:
                st.info("⏸️ Simulator is paused")
        
        else:
            st.info("📁 Please upload a CSV file to start simulation")
            st.markdown("""
            **Expected CSV format:**
            - One row per ball
            - Columns may include: batter, bowler, over, runs, extras, etc.
            - Example: `test_matches.csv`
            """)
    
    # Chat interface in right column
    with chat_col:
        render_chat_interface()

def render_chat_interface():
    """Render the chat interface for the simulator."""
    
    # Only render chat if simulator is initialized
    if 'simulator_data' not in st.session_state or st.session_state.simulator_data is None:
        st.info("🚫 Chat available after uploading match data")
        return
    
    st.markdown("### 💬 Chat with Assistant")
    st.markdown("---")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'tool_used' in message:
                st.info(f"🔧 Tool used: {message['tool_used']}")
    
    # Chat input
    if prompt := st.chat_input("Ask something tactical or odds-related..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Create context from current simulator state
        context = {}
        if st.session_state.simulator_data is not None:
            df = st.session_state.simulator_data
            current_ball = st.session_state.current_ball
            
            if current_ball <= len(df):
                current_row = df.iloc[current_ball - 1]
                context = {
                    'ball_number': current_ball,
                    'total_balls': len(df),
                    'current_data': current_row.to_dict(),
                    'match_state': {
                        'playing': st.session_state.simulator_playing,
                        'progress': (current_ball / len(df)) * 100
                    }
                }
        
        # Get response from chat agent
        with st.spinner("Thinking..."):
            result = ask_llm_with_tools(prompt, context)
        
        # Extract response and tool information
        response = result.get('answer', 'Sorry, I encountered an error.')
        tool_used = result.get('tool_used', 'unknown')
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'tool_used': tool_used
        })
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            if tool_used and tool_used != 'unknown':
                st.info(f"🔧 Tool used: {tool_used}")
        
        # Rerun to update the display
        st.rerun()

def extract_tool_used(response):
    """Extract tool information from chat response."""
    if "**Form Analysis Tool Used**" in response:
        return "get_form_vector()"
    elif "**Knowledge Graph Tool Used**" in response:
        return "query_kg_relationship()"
    elif "**Prediction Tool Used**" in response:
        return "predict_ball_outcome()"
    elif "**General Query Handler**" in response:
        return "fallback_handler()"
    else:
        return "unknown_tool()"

def create_sample_data():
    """Create sample cricket data for testing."""
    data = {
        'ball_id': range(1, 21),
        'over': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4],
        'batter': ['Smith'] * 12 + ['Jones'] * 8,
        'bowler': ['Kumar'] * 6 + ['Patel'] * 6 + ['Singh'] * 8,
        'runs': [1, 0, 4, 0, 2, 1, 0, 0, 6, 1, 0, 0, 1, 1, 0, 4, 0, 2, 1, 0],
        'extras': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'wicket': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)

# Application footer
def show_footer():
    """Display application footer with engineering principles."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with 🏏 Phi1618 Cricket AI Engineering Principles</p>
        <p>Scalable • Modular • Agent-Ready • Cloud-Deployable</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state for all components
    if 'status_messages' not in st.session_state:
        st.session_state.status_messages = []
    if 'simulator_data' not in st.session_state:
        st.session_state.simulator_data = None
    if 'simulator_playing' not in st.session_state:
        st.session_state.simulator_playing = False
    if 'current_ball' not in st.session_state:
        st.session_state.current_ball = 1
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    main()
    show_footer() 