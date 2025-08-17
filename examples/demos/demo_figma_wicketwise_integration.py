# Purpose: Demo showing Figma design integrated with Wicketwise cricket AI
# Author: Assistant, Last Modified: 2024-12-19

import streamlit as st
import pandas as pd
import time
from ui_theme import set_streamlit_theme
from ui_style import (
    render_figma_hero_section, 
    render_figma_card, 
    render_figma_navigation,
    render_figma_stats_panel,
    render_player_card,
    render_win_probability_bar
)

def main():
    """
    Demo showing your Figma design (meta-ethics-63199039.figma.site) 
    integrated with Wicketwise cricket AI
    """
    
    # Apply existing Wicketwise theme
    set_streamlit_theme()
    
    st.set_page_config(
        page_title="WicketWise - Figma Integration Demo",
        page_icon="🏏",
        layout="wide"
    )
    
    # Custom CSS to enhance the Figma integration
    st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    
    /* Hide Streamlit default elements for cleaner Figma integration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Enhance scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(26, 35, 50, 0.5);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(74, 144, 226, 0.5);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(74, 144, 226, 0.8);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section - Figma-inspired design
    st.markdown("---")
    render_figma_hero_section(
        "WicketWise Cricket AI",
        "Advanced cricket analytics with AI-powered predictions, real-time insights, and meta-ethical decision making",
        "Start Analyzing"
    )
    st.markdown("---")
    
    # Navigation - Figma style
    render_figma_navigation(
        ["Dashboard", "Live Match", "Predictions", "Player Analytics", "Betting", "Reports"],
        "Live Match"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏏 Live Match Analysis")
        
        # Match status using Figma cards
        match_col1, match_col2, match_col3 = st.columns(3)
        
        with match_col1:
            render_figma_card(
                "Current Match", 
                "India vs Australia\nTest Match, Day 2\nIndia: 287/4 (78 overs)",
                "🏏",
                "#c8712d"  # Batting color from your style guide
            )
        
        with match_col2:
            render_figma_card(
                "Win Probability", 
                "India 68% chance to win\nBased on current conditions\nAI confidence: 94%",
                "🎯",
                "#002466"  # Bowling color
            )
        
        with match_col3:
            render_figma_card(
                "Next Wicket", 
                "Probability: 23%\nNext 5 overs\nRisk level: Medium",
                "⚡",
                "#660003"  # Wicket color
            )
        
        # Live statistics panel
        st.markdown("### 📊 Real-Time Statistics")
        
        render_figma_stats_panel({
            "Run Rate": "3.67",
            "Strike Rate": "142.5",
            "Boundaries": "23",
            "Sixes": "7",
            "Partnership": "156",
            "Target": "387"
        }, "Live Match Statistics")
        
        # Player performance - mixing Figma and existing components
        st.markdown("### 👥 Player Performance")
        
        player_col1, player_col2 = st.columns(2)
        
        with player_col1:
            # Use existing Wicketwise component with enhanced data
            render_player_card(
                "Virat Kohli",
                {
                    "Runs": "74*",
                    "Balls": "89",
                    "Strike Rate": "83.15",
                    "Boundaries": "9",
                    "Form": "Excellent"
                },
                "#c8712d"
            )
        
        with player_col2:
            render_player_card(
                "Cheteshwar Pujara",
                {
                    "Runs": "45*", 
                    "Balls": "108",
                    "Strike Rate": "41.67",
                    "Boundaries": "4",
                    "Form": "Steady"
                },
                "#819f3d"  # Signals color
            )
    
    with col2:
        st.markdown("### 🤖 AI Insights")
        
        # AI predictions using Figma styling
        render_figma_card(
            "Meta-Ethical Analysis",
            "AI recommends conservative approach based on pitch conditions and historical data. Risk-reward ratio favors building partnership.",
            "🧠",
            "#4A90E2"
        )
        
        render_figma_card(
            "Betting Insights",
            "Current odds favor India (1.45). AI suggests value in 'Total runs over 420' market.",
            "💰",
            "#50C878"
        )
        
        render_figma_card(
            "Weather Impact",
            "15% chance of rain in next 2 hours. May affect play and strategy decisions.",
            "🌧️",
            "#F59E0B"
        )
        
        # Win probability bar (existing component)
        st.markdown("### 📈 Win Probability Tracker")
        render_win_probability_bar(68.0, 32.0, "India", "Australia")
        
        # Recent activity
        st.markdown("### 📰 Recent Activity")
        
        activity_items = [
            "🏏 Kohli reaches 50 (confidence: 97%)",
            "⚡ Wicket probability increased to 23%",
            "📊 Run rate dropped below required rate",
            "🎯 AI suggests aggressive approach",
            "💡 Weather update: Clear skies"
        ]
        
        for item in activity_items:
            st.markdown(f"""
            <div style="
                background: rgba(26, 35, 50, 0.6);
                border-left: 3px solid #4A90E2;
                padding: 12px 16px;
                margin: 8px 0;
                border-radius: 8px;
                font-family: 'Inter', sans-serif;
                color: rgba(255,255,255,0.9);
                font-size: 0.9rem;
            ">
                {item}
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with integration notes
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        **🎨 Figma Integration**
        - Hero section from meta-ethics design
        - Glass morphism cards
        - Animated navigation
        - Enhanced color scheme
        """)
    
    with footer_col2:
        st.markdown("""
        **🏏 Cricket AI Features**
        - Real-time predictions
        - Player analytics
        - Win probability tracking
        - Betting insights
        """)
    
    with footer_col3:
        st.markdown("""
        **🔧 Technical Stack**
        - Streamlit framework
        - Figma design tokens
        - CSS animations
        - Responsive layout
        """)
    
    # Success message
    st.success("🎉 Figma design successfully integrated with Wicketwise cricket AI!")
    
    # Integration instructions
    with st.expander("📖 How This Integration Works", expanded=False):
        st.markdown("""
        ## 🔄 Integration Process
        
        1. **Analyzed your Figma site**: https://meta-ethics-63199039.figma.site
        2. **Extracted design elements**: Colors, typography, layout patterns
        3. **Created Streamlit components** that match the Figma aesthetic
        4. **Integrated with existing Wicketwise** features and data
        
        ## 🎨 Design Elements Used
        
        - **Hero Section**: Large gradient background with animated elements
        - **Glass Morphism Cards**: Semi-transparent with blur effects
        - **Navigation**: Pill-shaped buttons with active states
        - **Color Scheme**: Dark backgrounds with blue/green accents
        - **Typography**: Inter font family throughout
        
        ## 🚀 Next Steps
        
        1. **Customize the design** to match your specific brand colors
        2. **Connect real cricket data** from your APIs
        3. **Add more interactive elements** based on user feedback
        4. **Optimize for mobile** devices
        
        ## 📁 Files Modified
        
        - `ui_style.py`: Added Figma-inspired components
        - `ui_theme.py`: Enhanced with Figma color tokens
        - Created integration demo files
        """)

if __name__ == "__main__":
    main()