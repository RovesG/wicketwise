# Purpose: Simple Figma integration demo that actually renders properly
# Author: Assistant, Last Modified: 2024-12-19

import streamlit as st

def main():
    """
    Simple demo that uses Streamlit's native components with Figma-inspired styling
    """
    
    st.set_page_config(
        page_title="WicketWise - Simple Figma Demo",
        page_icon="🏏",
        layout="wide"
    )
    
    # Simple, reliable CSS that works with Streamlit
    st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
        color: white;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: white;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
        color: white;
    }
    
    .cta-button {
        background: linear-gradient(45deg, #4A90E2, #50C878);
        color: white;
        padding: 1rem 2rem;
        border-radius: 2rem;
        border: none;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
        text-decoration: none;
    }
    
    .figma-card {
        background: rgba(26, 35, 50, 0.9);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    .card-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .card-content {
        color: rgba(255,255,255,0.8);
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section using simple HTML
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">WicketWise Cricket AI</h1>
        <p class="hero-subtitle">Advanced cricket analytics with AI-powered predictions</p>
        <div class="cta-button">Start Analyzing</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation using Streamlit columns
    st.markdown("### 🧭 Navigation")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🏏 Dashboard", use_container_width=True):
            st.success("Dashboard selected!")
    with col2:
        if st.button("📊 Live Match", use_container_width=True, type="primary"):
            st.success("Live Match selected!")
    with col3:
        if st.button("🎯 Predictions", use_container_width=True):
            st.success("Predictions selected!")
    with col4:
        if st.button("👥 Analytics", use_container_width=True):
            st.success("Analytics selected!")
    with col5:
        if st.button("💰 Betting", use_container_width=True):
            st.success("Betting selected!")
    
    # Cards using simple HTML
    st.markdown("### 🏏 Match Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="figma-card">
            <div class="card-icon">🏏</div>
            <div class="card-title">Current Match</div>
            <div class="card-content">
                India vs Australia<br>
                Test Match, Day 2<br>
                India: 287/4 (78 overs)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="figma-card">
            <div class="card-icon">🎯</div>
            <div class="card-title">Win Probability</div>
            <div class="card-content">
                India 68% chance to win<br>
                Based on current conditions<br>
                AI confidence: 94%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="figma-card">
            <div class="card-icon">⚡</div>
            <div class="card-title">Next Wicket</div>
            <div class="card-content">
                Probability: 23%<br>
                Next 5 overs<br>
                Risk level: Medium
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Live Statistics using Streamlit metrics
    st.markdown("### 📊 Live Statistics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Run Rate", "3.67", "0.12")
    with col2:
        st.metric("Strike Rate", "142.5", "5.2")
    with col3:
        st.metric("Boundaries", "23", "2")
    with col4:
        st.metric("Sixes", "7", "1")
    with col5:
        st.metric("Partnership", "156", "45")
    with col6:
        st.metric("Target", "387", "-")
    
    # Player Performance
    st.markdown("### 👥 Player Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="figma-card" style="border-top: 3px solid #c8712d;">
            <div class="card-title">Virat Kohli</div>
            <div class="card-content">
                <strong>Runs:</strong> 74*<br>
                <strong>Balls:</strong> 89<br>
                <strong>Strike Rate:</strong> 83.15<br>
                <strong>Boundaries:</strong> 9<br>
                <strong>Form:</strong> Excellent
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="figma-card" style="border-top: 3px solid #819f3d;">
            <div class="card-title">Cheteshwar Pujara</div>
            <div class="card-content">
                <strong>Runs:</strong> 45*<br>
                <strong>Balls:</strong> 108<br>
                <strong>Strike Rate:</strong> 41.67<br>
                <strong>Boundaries:</strong> 4<br>
                <strong>Form:</strong> Steady
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Insights
    st.markdown("### 🤖 AI Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Meta-Ethical Analysis**: AI recommends conservative approach based on pitch conditions and historical data. Risk-reward ratio favors building partnership.")
        
        st.success("**Betting Insights**: Current odds favor India (1.45). AI suggests value in 'Total runs over 420' market.")
    
    with col2:
        st.warning("**Weather Impact**: 15% chance of rain in next 2 hours. May affect play and strategy decisions.")
        
        # Win Probability Bar using Streamlit progress bar
        st.markdown("**Win Probability**")
        st.progress(0.68, text="India: 68% | Australia: 32%")
    
    # Recent Activity
    st.markdown("### 📰 Recent Activity")
    
    activities = [
        "🏏 Kohli reaches 50 (confidence: 97%)",
        "⚡ Wicket probability increased to 23%", 
        "📊 Run rate dropped below required rate",
        "🎯 AI suggests aggressive approach",
        "💡 Weather update: Clear skies"
    ]
    
    for activity in activities:
        st.markdown(f"- {activity}")
    
    # Success message
    st.success("🎉 Simple Figma integration working! This uses Streamlit's native components with Figma-inspired styling.")
    
    # Show the difference
    with st.expander("📖 Why This Works vs The Previous Version"):
        st.markdown("""
        **✅ This Simple Version Works Because:**
        - Uses basic HTML that Streamlit can reliably render
        - Combines Streamlit native components (buttons, metrics, progress bars)
        - Simple CSS classes instead of complex inline styles
        - No complex nested HTML structures
        - Reliable `st.markdown()` with `unsafe_allow_html=True`
        
        **❌ Previous Version Had Issues Because:**
        - Complex nested HTML with many inline styles
        - JavaScript event handlers (`onmouseover`, `onmouseout`)
        - Complex CSS animations and transforms
        - F-string formatting issues with HTML templates
        
        **🎨 Design Elements Preserved:**
        - Figma color scheme (dark gradients, blues, greens)
        - Glass morphism effects (semi-transparent backgrounds)
        - Modern card layouts with rounded corners
        - Professional typography and spacing
        - Cricket-specific color coding
        """)

if __name__ == "__main__":
    main()