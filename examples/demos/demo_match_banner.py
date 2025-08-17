# Purpose: Demo of match banner component for cricket analysis
# Author: Claude, Last Modified: 2025-01-17

import streamlit as st
from ui_style import render_match_banner

def main():
    """Main demo application"""
    st.set_page_config(
        page_title="Match Banner Demo",
        page_icon="🏏",
        layout="wide"
    )
    
    st.title("🏏 Match Banner Component Demo")
    st.markdown("This demo showcases the match banner component with different match scenarios.")
    
    # Demo 1: Live T20 Match
    st.subheader("🔴 Live T20 Match")
    live_match = {
        'team1_name': 'Mumbai Indians',
        'team1_score': 187,
        'team1_wickets': 5,
        'team2_name': 'Chennai Super Kings',
        'team2_score': 45,
        'team2_wickets': 2,
        'current_over': 7,
        'current_ball': 3,
        'current_innings': 2,
        'match_phase': 'Powerplay',
        'team1_color': '#004BA0',
        'team2_color': '#FFFF3C'
    }
    
    render_match_banner(live_match)
    
    # Demo 2: First Innings - Middle Overs
    st.subheader("🏏 First Innings - Middle Overs")
    first_innings = {
        'team1_name': 'Royal Challengers Bangalore',
        'team1_score': 89,
        'team1_wickets': 3,
        'team2_name': 'Kolkata Knight Riders',
        'team2_score': 0,
        'team2_wickets': 0,
        'current_over': 12,
        'current_ball': 4,
        'current_innings': 1,
        'match_phase': 'Middle Overs',
        'team1_color': '#d41e3a',
        'team2_color': '#3a225d'
    }
    
    render_match_banner(first_innings)
    
    # Demo 3: Death Overs - High Pressure
    st.subheader("⚡ Death Overs - High Pressure")
    death_overs = {
        'team1_name': 'Rajasthan Royals',
        'team1_score': 165,
        'team1_wickets': 7,
        'team2_name': 'Sunrisers Hyderabad',
        'team2_score': 142,
        'team2_wickets': 4,
        'current_over': 18,
        'current_ball': 2,
        'current_innings': 2,
        'match_phase': 'Death Overs',
        'team1_color': '#254AA5',
        'team2_color': '#FF822A'
    }
    
    render_match_banner(death_overs)
    
    # Demo 4: Super Over Thriller
    st.subheader("🔥 Super Over Thriller")
    super_over = {
        'team1_name': 'Delhi Capitals',
        'team1_score': 156,
        'team1_wickets': 8,
        'team2_name': 'Kings XI Punjab',
        'team2_score': 156,
        'team2_wickets': 9,
        'current_over': 20,
        'current_ball': 6,
        'current_innings': 3,
        'match_phase': 'Super Over',
        'team1_color': '#17479e',
        'team2_color': '#ed1b24'
    }
    
    render_match_banner(super_over)
    
    # Demo 5: Team All Out
    st.subheader("🎯 Team All Out")
    all_out = {
        'team1_name': 'Gujarat Titans',
        'team1_score': 213,
        'team1_wickets': 4,
        'team2_name': 'Lucknow Super Giants',
        'team2_score': 124,
        'team2_wickets': 10,
        'current_over': 16,
        'current_ball': 3,
        'current_innings': 2,
        'match_phase': 'All Out',
        'team1_color': '#1e3a8a',
        'team2_color': '#0ea5e9'
    }
    
    render_match_banner(all_out)
    
    # Interactive Demo Section
    st.subheader("🎮 Interactive Match Banner")
    st.markdown("Customize your own match banner:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Team 1**")
        team1_name = st.text_input("Team 1 Name", "Team A")
        team1_score = st.number_input("Team 1 Score", 0, 500, 120)
        team1_wickets = st.number_input("Team 1 Wickets", 0, 10, 3)
        team1_color = st.color_picker("Team 1 Color", "#004BA0")
        
    with col2:
        st.markdown("**Team 2**")
        team2_name = st.text_input("Team 2 Name", "Team B")
        team2_score = st.number_input("Team 2 Score", 0, 500, 85)
        team2_wickets = st.number_input("Team 2 Wickets", 0, 10, 2)
        team2_color = st.color_picker("Team 2 Color", "#FFFF3C")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Match Status**")
        current_over = st.number_input("Current Over", 0, 50, 14)
        current_ball = st.number_input("Current Ball", 0, 6, 2)
        current_innings = st.selectbox("Current Innings", [1, 2, 3], index=1)
        
    with col4:
        st.markdown("**Match Phase**")
        match_phase = st.selectbox(
            "Match Phase", 
            ["Powerplay", "Middle Overs", "Death Overs", "Super Over", "All Out"],
            index=1
        )
    
    # Generate interactive banner
    custom_match = {
        'team1_name': team1_name,
        'team1_score': team1_score,
        'team1_wickets': team1_wickets,
        'team2_name': team2_name,
        'team2_score': team2_score,
        'team2_wickets': team2_wickets,
        'current_over': current_over,
        'current_ball': current_ball,
        'current_innings': current_innings,
        'match_phase': match_phase,
        'team1_color': team1_color,
        'team2_color': team2_color
    }
    
    st.markdown("**Your Custom Match Banner:**")
    render_match_banner(custom_match)
    
    # Display the configuration
    with st.expander("View Configuration"):
        st.json(custom_match)
    
    # Features overview
    st.markdown("---")
    st.subheader("📋 Match Banner Features")
    
    features = [
        "🎨 **Team Colors**: Dynamic border colors based on current batting team",
        "📱 **Responsive Design**: Adapts to mobile and desktop screen sizes",
        "🏏 **Cricket Emojis**: Visual indicators for overs, innings, and match phase",
        "⚡ **Live Updates**: Real-time match information display",
        "🎯 **Fallback Values**: Graceful handling of missing data",
        "💎 **Modern UI**: Dark gradient background with elegant styling",
        "🔥 **Phase Indicators**: Visual cues for different match phases"
    ]
    
    for feature in features:
        st.markdown(feature)
    
    # Technical details
    st.markdown("---")
    st.subheader("⚙️ Technical Implementation")
    
    tech_details = """
    **CSS Features:**
    - Flexbox layout for responsive design
    - CSS gradients for modern appearance
    - Media queries for mobile optimization
    - Team color integration
    - Hover effects and transitions
    
    **Data Structure:**
    ```python
    match_info = {
        'team1_name': 'Mumbai Indians',
        'team1_score': 187,
        'team1_wickets': 5,
        'team2_name': 'Chennai Super Kings',
        'team2_score': 45,
        'team2_wickets': 2,
        'current_over': 7,
        'current_ball': 3,
        'current_innings': 2,
        'match_phase': 'Powerplay',
        'team1_color': '#004BA0',
        'team2_color': '#FFFF3C'
    }
    ```
    """
    
    st.markdown(tech_details)
    
    # Usage examples
    st.markdown("---")
    st.subheader("💻 Usage Examples")
    
    usage_code = """
    ```python
    from ui_style import render_match_banner
    
    # Basic usage
    match_data = {
        'team1_name': 'Mumbai Indians',
        'team1_score': 187,
        'team1_wickets': 5,
        'team2_name': 'Chennai Super Kings',
        'team2_score': 45,
        'team2_wickets': 2,
        'current_over': 7,
        'current_ball': 3,
        'current_innings': 2,
        'match_phase': 'Powerplay'
    }
    
    render_match_banner(match_data)
    ```
    """
    
    st.markdown(usage_code)

if __name__ == "__main__":
    main() 