# Purpose: Demo of responsive layout system for cricket analysis dashboard
# Author: Claude, Last Modified: 2025-01-17

import streamlit as st
from layout_utils import (
    render_responsive_dashboard, 
    LayoutMode, 
    create_component_wrapper,
    get_layout_recommendations
)

# Demo component functions
def demo_video_component():
    """Demo video player component"""
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    st.caption("🎬 Match highlights and ball-by-ball replay")
    st.info("📱 Video adapts to screen size automatically")

def demo_batter_card():
    """Demo batter card component"""
    st.subheader("🏏 Virat Kohli")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Runs", "89", "12")
        st.metric("Strike Rate", "142.5", "8.3")
        st.metric("Boundaries", "8", "2")
    with col2:
        st.metric("Balls", "63", "5")
        st.metric("Average", "52.4", "1.2")
        st.metric("High Score", "183", "0")

def demo_bowler_card():
    """Demo bowler card component"""
    st.subheader("🎳 Jasprit Bumrah")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Wickets", "3", "1")
        st.metric("Economy", "6.25", "-0.5")
        st.metric("Overs", "4.0", "0")
    with col2:
        st.metric("Runs", "25", "8")
        st.metric("Best Figures", "4/19", "0")
        st.metric("Maidens", "0", "0")

def demo_win_probability():
    """Demo win probability component"""
    st.subheader("🎯 Win Probability")
    
    # Simulate win probability data
    team_a_prob = 0.67
    team_b_prob = 1 - team_a_prob
    
    # Progress bar for Team A
    st.write("**Mumbai Indians**")
    st.progress(team_a_prob)
    st.write(f"{team_a_prob:.1%} chance to win")
    
    # Progress bar for Team B
    st.write("**Chennai Super Kings**")
    st.progress(team_b_prob)
    st.write(f"{team_b_prob:.1%} chance to win")
    
    st.caption("📊 Real-time AI predictions based on match situation")

def demo_chat_component():
    """Demo chat interface component"""
    st.subheader("💬 Cricket Analysis Chat")
    
    # Chat messages
    messages = [
        ("user", "What's the predicted score at the end of innings?"),
        ("assistant", "Based on current run rate and wickets, Mumbai Indians are likely to score around 185-190 runs."),
        ("user", "Who has the highest strike rate this season?"),
        ("assistant", "Jos Buttler leads with a strike rate of 149.05 this season.")
    ]
    
    for role, message in messages:
        if role == "user":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**AI:** {message}")
    
    # Chat input
    user_input = st.text_input("Ask about cricket analytics...", placeholder="Type your question here")
    if user_input:
        st.write(f"**You:** {user_input}")
        st.write("**AI:** I'm analyzing that for you...")

def demo_additional_tools():
    """Demo additional tools component"""
    st.subheader("🔧 Analysis Tools")
    
    with st.expander("📈 Performance Metrics"):
        st.selectbox("Select metric", ["Strike Rate", "Economy", "Boundary %", "Dot Ball %"])
        st.slider("Time range (overs)", 1, 20, 6)
        st.button("Generate Analysis")
    
    with st.expander("🎯 Prediction Settings"):
        st.checkbox("Include weather conditions")
        st.checkbox("Consider pitch report")
        st.selectbox("Model type", ["XGBoost", "Neural Network", "Ensemble"])

def main():
    """Main demo application"""
    st.set_page_config(
        page_title="Responsive Layout Demo",
        page_icon="🏏",
        layout="wide"
    )
    
    st.title("🏏 Responsive Cricket Dashboard Demo")
    st.markdown("This demo showcases the responsive layout system that adapts to different screen sizes.")
    
    # Layout mode selector
    st.sidebar.subheader("📱 Layout Controls")
    layout_mode_options = {
        "Auto Detect": None,
        "Wide Desktop": LayoutMode.WIDE_DESKTOP,
        "Medium Tablet": LayoutMode.MEDIUM_TABLET,
        "Mobile": LayoutMode.MOBILE
    }
    
    selected_mode = st.sidebar.selectbox(
        "Force Layout Mode",
        list(layout_mode_options.keys()),
        help="Select 'Auto Detect' to let the system choose based on screen size"
    )
    
    layout_mode = layout_mode_options[selected_mode]
    
    # Container width simulator
    if layout_mode is None:
        container_width = st.sidebar.slider(
            "Simulate Container Width (px)",
            300, 2000, 1200,
            help="Simulate different screen sizes for testing"
        )
    else:
        container_width = None
    
    # Show layout recommendations
    if layout_mode:
        recommendations = get_layout_recommendations(layout_mode)
        st.sidebar.subheader("📋 Layout Recommendations")
        st.sidebar.json(recommendations)
    
    # Create component wrappers
    player_cards = [
        create_component_wrapper(demo_batter_card),
        create_component_wrapper(demo_bowler_card)
    ]
    
    additional_components = [
        create_component_wrapper(demo_additional_tools)
    ]
    
    # Render responsive dashboard
    render_responsive_dashboard(
        video_component=demo_video_component,
        player_cards=player_cards,
        win_probability=demo_win_probability,
        chat_component=demo_chat_component,
        additional_components=additional_components,
        layout_mode=layout_mode,
        container_width=container_width
    )
    
    # Show demo information
    st.sidebar.markdown("---")
    st.sidebar.subheader("📖 Demo Info")
    st.sidebar.markdown("""
    **Features Demonstrated:**
    - Responsive breakpoints (768px, 1024px)
    - Adaptive component layouts
    - Mobile-first design
    - CSS media queries
    - Component wrappers
    - Debug information
    
    **Layout Modes:**
    - **Mobile**: Stacked layout, collapsible chat
    - **Tablet**: Two-column cards, visible sidebar
    - **Desktop**: Side-by-side layout, full features
    """)

if __name__ == "__main__":
    main() 