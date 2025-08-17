# Purpose: Demo showing how to integrate Figma designs into Wicketwise UI
# Author: Assistant, Last Modified: 2024-12-19

import streamlit as st
import base64
from pathlib import Path

# Import existing Wicketwise UI components
from ui_theme import set_streamlit_theme
from ui_style import render_player_card, render_win_probability_bar

def demo_figma_integration():
    """
    Demo showing Figma integration with existing Wicketwise components
    """
    
    # Apply existing theme
    set_streamlit_theme()
    
    st.title("🎨 Figma Integration Demo")
    st.markdown("See how your Figma design integrates with existing Wicketwise components")
    
    # Tabs for different integration methods
    tab1, tab2, tab3 = st.tabs(["📤 Upload Figma Export", "🎨 Design Tokens", "🧩 Component Preview"])
    
    with tab1:
        st.subheader("Method 1: Upload Figma Export")
        st.markdown("""
        **Steps to export from Figma:**
        1. Select your frames in Figma
        2. Use "Figma to Code" plugin → Export as HTML/CSS
        3. Upload the files below
        """)
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload your Figma export files",
            accept_multiple_files=True,
            type=['html', 'css', 'json', 'png', 'svg']
        )
        
        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} files")
            
            for file in uploaded_files:
                with st.expander(f"📄 {file.name}"):
                    if file.name.endswith(('.html', '.css')):
                        content = file.read().decode('utf-8')
                        st.code(content[:500] + "..." if len(content) > 500 else content)
                    elif file.name.endswith(('.png', '.jpg', '.svg')):
                        st.image(file)
                    
            if st.button("🔄 Convert to Streamlit Components"):
                st.info("This would generate Streamlit component functions from your Figma export")
                st.code("""
# Generated component example:
def render_figma_hero_section(title, subtitle):
    hero_html = f'''
    <div style="
        background: linear-gradient(135deg, #4A90E2 0%, #50C878 100%);
        padding: 48px 32px;
        border-radius: 24px;
        text-align: center;
    ">
        <h1 style="color: white; font-size: 2.5rem;">{title}</h1>
        <p style="color: rgba(255,255,255,0.9);">{subtitle}</p>
    </div>
    '''
    st.markdown(hero_html, unsafe_allow_html=True)
                """, language='python')
    
    with tab2:
        st.subheader("Method 2: Extract Design Tokens")
        st.markdown("Extract colors, fonts, and spacing from your Figma design")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Wicketwise Tokens:**")
            current_tokens = {
                "colors": {
                    "primary_bg": "#0F1419",
                    "secondary_bg": "#1A2332", 
                    "surface": "#252B37",
                    "accent_blue": "#4A90E2",
                    "text_primary": "#FFFFFF",
                    "text_secondary": "#B8BCC8"
                },
                "typography": {
                    "font_family": "Inter, sans-serif",
                    "heading_1": "2.5rem",
                    "heading_2": "2rem", 
                    "body": "1rem"
                },
                "spacing": {
                    "sm": "8px",
                    "md": "16px",
                    "lg": "24px",
                    "xl": "32px"
                }
            }
            st.json(current_tokens)
        
        with col2:
            st.markdown("**Your Figma Tokens:**")
            st.markdown("*Upload your design tokens or manually enter:*")
            
            # Interactive token input
            figma_primary = st.color_picker("Primary Color", "#4A90E2")
            figma_secondary = st.color_picker("Secondary Color", "#50C878") 
            figma_bg = st.color_picker("Background Color", "#0F1419")
            
            figma_font = st.selectbox("Font Family", [
                "Inter, sans-serif",
                "Roboto, sans-serif", 
                "Poppins, sans-serif",
                "Montserrat, sans-serif",
                "Custom font"
            ])
            
            if figma_font == "Custom font":
                figma_font = st.text_input("Enter custom font family")
            
            # Generate token object
            figma_tokens = {
                "colors": {
                    "primary": figma_primary,
                    "secondary": figma_secondary,
                    "background": figma_bg
                },
                "typography": {
                    "font_family": figma_font
                }
            }
            
            st.json(figma_tokens)
    
    with tab3:
        st.subheader("Component Preview")
        st.markdown("See how Figma components integrate with existing Wicketwise UI")
        
        # Sample integration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Existing Wicketwise Components:**")
            
            # Existing player card
            render_player_card(
                "Virat Kohli",
                {
                    "Runs": "8,074",
                    "Average": "52.73",
                    "Strike Rate": "137.96",
                    "Sixes": "117"
                },
                "#4A90E2"
            )
            
            # Existing win probability bar
            render_win_probability_bar(0.75, "Team A", "Team B")
        
        with col2:
            st.markdown("**Your Figma Components (Preview):**")
            
            # Demo Figma-style component
            figma_card_demo = """
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                padding: 24px;
                margin: 16px 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                color: white;
                text-align: center;
            ">
                <h3 style="margin: 0 0 16px 0; font-size: 1.5rem;">Figma Card Style</h3>
                <p style="margin: 0; opacity: 0.9;">This shows how your Figma design would look</p>
                <div style="
                    background: rgba(255,255,255,0.2);
                    border-radius: 12px;
                    padding: 16px;
                    margin-top: 16px;
                    backdrop-filter: blur(10px);
                ">
                    <strong>Match Prediction: 85%</strong>
                </div>
            </div>
            """
            st.markdown(figma_card_demo, unsafe_allow_html=True)
            
            # Figma-style button
            figma_button_demo = """
            <div style="text-align: center; margin: 24px 0;">
                <button style="
                    background: linear-gradient(45deg, #ff6b6b, #ff8e53);
                    border: none;
                    border-radius: 25px;
                    color: white;
                    padding: 16px 32px;
                    font-size: 1.1rem;
                    font-weight: 600;
                    cursor: pointer;
                    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
                    transition: all 0.3s ease;
                " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(255, 107, 107, 0.6)'"
                   onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 15px rgba(255, 107, 107, 0.4)'">
                    Place Bet Now
                </button>
            </div>
            """
            st.markdown(figma_button_demo, unsafe_allow_html=True)
    
    # Integration guide
    st.markdown("---")
    st.subheader("🚀 Ready to Integrate?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📖 View Integration Guide", use_container_width=True):
            st.info("Check `FIGMA_INTEGRATION_STEPS.md` for detailed steps")
    
    with col2:
        if st.button("🔧 Run Converter Tool", use_container_width=True):
            st.info("Run: `streamlit run figma_to_streamlit_converter.py`")
    
    with col3:
        if st.button("💬 Get Help", use_container_width=True):
            st.info("Share your Figma file or exported code for specific help")

def create_sample_figma_component():
    """
    Example of how to create a Figma-inspired component
    """
    
    def render_figma_hero_section(title: str, subtitle: str, cta_text: str = "Get Started"):
        """Figma-inspired hero section component"""
        
        hero_html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 60px 40px;
            border-radius: 24px;
            text-align: center;
            margin: 32px 0;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
        ">
            <!-- Background pattern -->
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-image: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
                                  radial-gradient(circle at 80% 20%, rgba(255,255,255,0.1) 0%, transparent 50%);
            "></div>
            
            <!-- Content -->
            <div style="position: relative; z-index: 1;">
                <h1 style="
                    font-family: 'Inter', sans-serif;
                    font-size: 3rem;
                    font-weight: 700;
                    color: #FFFFFF;
                    margin-bottom: 16px;
                    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
                ">{title}</h1>
                
                <p style="
                    font-family: 'Inter', sans-serif;
                    font-size: 1.25rem;
                    color: rgba(255,255,255,0.9);
                    margin-bottom: 32px;
                    max-width: 600px;
                    margin-left: auto;
                    margin-right: auto;
                ">{subtitle}</p>
                
                <button style="
                    background: rgba(255,255,255,0.2);
                    backdrop-filter: blur(10px);
                    border: 2px solid rgba(255,255,255,0.3);
                    color: #FFFFFF;
                    padding: 16px 32px;
                    border-radius: 50px;
                    font-family: 'Inter', sans-serif;
                    font-size: 1.1rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
                " onmouseover="this.style.background='rgba(255,255,255,0.3)'; this.style.transform='translateY(-2px)'"
                   onmouseout="this.style.background='rgba(255,255,255,0.2)'; this.style.transform='translateY(0px)'">{cta_text}</button>
            </div>
        </div>
        """
        
        st.markdown(hero_html, unsafe_allow_html=True)
    
    return render_figma_hero_section

if __name__ == "__main__":
    demo_figma_integration()
    
    # Show sample component
    st.markdown("---")
    st.subheader("Sample Figma Component")
    
    sample_hero = create_sample_figma_component()
    sample_hero(
        "WicketWise AI",
        "Transform your cricket betting with advanced AI predictions",
        "Start Analyzing"
    )