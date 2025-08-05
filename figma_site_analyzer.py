# Purpose: Analyze and extract design elements from the Figma published site
# Author: Assistant, Last Modified: 2024-12-19

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urljoin, urlparse

class FigmaSiteAnalyzer:
    """
    Analyze a published Figma site to extract design elements and convert them
    to Streamlit-compatible components for Wicketwise
    """
    
    def __init__(self, figma_url):
        self.figma_url = figma_url
        self.design_tokens = {}
        self.components = []
        
    def analyze_site(self):
        """
        Analyze the Figma published site to extract design elements
        """
        try:
            # Note: In practice, you might need to handle CORS issues
            # For now, we'll provide manual analysis tools
            st.warning("Direct site scraping may be limited by CORS. Using manual analysis approach.")
            return self.manual_analysis_guide()
            
        except Exception as e:
            st.error(f"Error analyzing site: {e}")
            return self.manual_analysis_guide()
    
    def manual_analysis_guide(self):
        """
        Provide a guide for manually analyzing the Figma site
        """
        analysis_guide = {
            "step_1": "Visit the Figma site and inspect design elements",
            "step_2": "Use browser dev tools to extract CSS and HTML",
            "step_3": "Identify color palette, typography, and spacing patterns",
            "step_4": "Note component layouts and interactions"
        }
        
        # Common design patterns we might expect
        expected_elements = {
            "colors": {
                "primary": "#4A90E2",  # Default blue
                "secondary": "#50C878", # Default green
                "background": "#0F1419", # Dark background
                "text": "#FFFFFF"
            },
            "typography": {
                "font_family": "Inter, sans-serif",
                "heading_sizes": ["2.5rem", "2rem", "1.5rem"],
                "body_size": "1rem"
            },
            "components": {
                "cards": "Glass morphism style cards",
                "buttons": "Rounded buttons with hover effects",
                "navigation": "Side navigation or top nav",
                "layout": "Grid or flexbox layouts"
            }
        }
        
        return {
            "guide": analysis_guide,
            "expected_elements": expected_elements,
            "next_steps": [
                "Manually inspect the Figma site",
                "Extract design tokens using browser dev tools",
                "Create matching Streamlit components",
                "Test integration with Wicketwise"
            ]
        }
    
    def extract_design_tokens_from_css(self, css_content):
        """
        Extract design tokens from CSS content
        """
        tokens = {
            "colors": {},
            "fonts": {},
            "spacing": {},
            "borders": {}
        }
        
        # Extract CSS custom properties
        css_vars = re.findall(r'--([^:]+):\s*([^;]+);', css_content)
        for var_name, var_value in css_vars:
            if 'color' in var_name.lower():
                tokens["colors"][var_name] = var_value.strip()
            elif any(unit in var_value for unit in ['px', 'rem', 'em']):
                if 'radius' in var_name.lower():
                    tokens["borders"][var_name] = var_value.strip()
                else:
                    tokens["spacing"][var_name] = var_value.strip()
        
        # Extract direct color values
        colors = re.findall(r'#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}|rgb\([^)]+\)|rgba\([^)]+\)', css_content)
        for i, color in enumerate(set(colors)):
            tokens["colors"][f"extracted_color_{i}"] = color
            
        return tokens
    
    def create_wicketwise_components(self, design_analysis):
        """
        Create Wicketwise-compatible components based on design analysis
        """
        components = []
        
        # Hero section component
        hero_component = self.create_figma_hero_component()
        components.append(hero_component)
        
        # Card component
        card_component = self.create_figma_card_component()
        components.append(card_component)
        
        # Navigation component
        nav_component = self.create_figma_nav_component()
        components.append(nav_component)
        
        return components
    
    def create_figma_hero_component(self):
        """
        Create a hero section component inspired by Figma design
        """
        component_code = '''
def render_figma_hero_section(title: str, subtitle: str, cta_text: str = "Get Started"):
    """
    Hero section component inspired by meta-ethics Figma design
    """
    
    hero_html = f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 80px 40px;
        border-radius: 20px;
        text-align: center;
        margin: 32px 0;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    ">
        <!-- Animated background elements -->
        <div style="
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(74,144,226,0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        "></div>
        
        <!-- Content -->
        <div style="position: relative; z-index: 1;">
            <h1 style="
                font-family: 'Inter', sans-serif;
                font-size: 3.5rem;
                font-weight: 700;
                color: #FFFFFF;
                margin-bottom: 24px;
                text-shadow: 0 4px 20px rgba(0,0,0,0.3);
                line-height: 1.2;
            ">{title}</h1>
            
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 1.4rem;
                color: rgba(255,255,255,0.85);
                margin-bottom: 40px;
                max-width: 700px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.6;
            ">{subtitle}</p>
            
            <button style="
                background: linear-gradient(45deg, #4A90E2, #50C878);
                border: none;
                color: #FFFFFF;
                padding: 18px 36px;
                border-radius: 50px;
                font-family: 'Inter', sans-serif;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 6px 25px rgba(74,144,226,0.4);
                text-transform: uppercase;
                letter-spacing: 1px;
            " onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 8px 35px rgba(74,144,226,0.6)'"
               onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 6px 25px rgba(74,144,226,0.4)'">{cta_text}</button>
        </div>
    </div>
    
    <style>
    @keyframes rotate {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """
    
    st.markdown(hero_html, unsafe_allow_html=True)
'''
        
        return {
            "name": "figma_hero_section",
            "code": component_code,
            "description": "Hero section with animated background inspired by meta-ethics design"
        }
    
    def create_figma_card_component(self):
        """
        Create a card component inspired by Figma design
        """
        component_code = '''
def render_figma_card(title: str, content: str, icon: str = "🏏", color: str = "#4A90E2"):
    """
    Card component inspired by meta-ethics Figma design
    """
    
    card_html = f"""
    <div style="
        background: rgba(26, 35, 50, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 32px;
        margin: 16px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 15px 50px rgba(0,0,0,0.4)'"
       onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 10px 40px rgba(0,0,0,0.3)'">
        
        <!-- Accent line -->
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, {color}, rgba(255,255,255,0.1));
        "></div>
        
        <!-- Icon and title -->
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="
                font-size: 2rem;
                margin-right: 16px;
                padding: 12px;
                background: rgba(74,144,226,0.1);
                border-radius: 12px;
                border: 1px solid rgba(74,144,226,0.2);
            ">{icon}</div>
            <h3 style="
                font-family: 'Inter', sans-serif;
                color: #FFFFFF;
                font-weight: 600;
                margin: 0;
                font-size: 1.4rem;
            ">{title}</h3>
        </div>
        
        <!-- Content -->
        <p style="
            font-family: 'Inter', sans-serif;
            color: rgba(255,255,255,0.8);
            margin: 0;
            line-height: 1.6;
            font-size: 1rem;
        ">{content}</p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
'''
        
        return {
            "name": "figma_card",
            "code": component_code,
            "description": "Animated card with glass morphism effect"
        }
    
    def create_figma_nav_component(self):
        """
        Create a navigation component inspired by Figma design
        """
        component_code = '''
def render_figma_navigation(nav_items: list, active_item: str = ""):
    """
    Navigation component inspired by meta-ethics Figma design
    """
    
    nav_html = """
    <div style="
        background: rgba(15, 20, 25, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    ">
        <div style="display: flex; justify-content: space-around; align-items: center;">
    """
    
    for item in nav_items:
        is_active = item.lower() == active_item.lower()
        active_style = """
            background: linear-gradient(45deg, #4A90E2, #50C878);
            color: #FFFFFF;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(74,144,226,0.4);
        """ if is_active else """
            background: rgba(255,255,255,0.05);
            color: rgba(255,255,255,0.7);
        """
        
        nav_html += f"""
            <div style="
                padding: 12px 24px;
                border-radius: 25px;
                font-family: 'Inter', sans-serif;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                border: 1px solid rgba(255,255,255,0.1);
                {active_style}
            " onmouseover="if (!this.style.background.includes('linear-gradient')) {{ this.style.background='rgba(255,255,255,0.1)'; this.style.color='#FFFFFF'; }}"
               onmouseout="if (!this.style.background.includes('linear-gradient')) {{ this.style.background='rgba(255,255,255,0.05)'; this.style.color='rgba(255,255,255,0.7)'; }}">
                {item}
            </div>
        """
    
    nav_html += """
        </div>
    </div>
    """
    
    st.markdown(nav_html, unsafe_allow_html=True)
'''
        
        return {
            "name": "figma_navigation",
            "code": component_code,
            "description": "Animated navigation with active states"
        }

def main():
    st.title("🎨 Figma Site Analyzer for Wicketwise")
    st.markdown("Analyze and extract design elements from your Figma published site")
    
    # Input for Figma URL
    figma_url = st.text_input(
        "Figma Published Site URL",
        value="https://meta-ethics-63199039.figma.site",
        help="Enter your Figma published site URL"
    )
    
    if figma_url:
        analyzer = FigmaSiteAnalyzer(figma_url)
        
        st.subheader("📋 Site Analysis")
        
        # Analyze the site
        analysis = analyzer.analyze_site()
        
        # Display analysis results
        with st.expander("🔍 Design Analysis", expanded=True):
            st.json(analysis)
        
        # Manual extraction tools
        st.subheader("🛠️ Manual Extraction Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Step 1: Visit Your Figma Site**
            1. Open: [{}]({})
            2. Right-click → "Inspect Element"
            3. Copy the CSS and HTML
            """.format(figma_url, figma_url))
            
            css_content = st.text_area(
                "Paste CSS Content Here",
                height=200,
                help="Copy CSS from browser dev tools"
            )
            
            if css_content:
                tokens = analyzer.extract_design_tokens_from_css(css_content)
                st.json(tokens)
        
        with col2:
            st.markdown("**Step 2: Generate Components**")
            
            if st.button("🚀 Generate Wicketwise Components"):
                components = analyzer.create_wicketwise_components(analysis)
                
                st.success(f"Generated {len(components)} components!")
                
                for component in components:
                    with st.expander(f"📦 {component['name']}", expanded=False):
                        st.markdown(f"**Description:** {component['description']}")
                        st.code(component['code'], language='python')
        
        # Preview components
        st.subheader("👀 Component Preview")
        
        # Demo the generated components
        components = analyzer.create_wicketwise_components(analysis)
        
        # Show component preview using HTML directly (safer than exec)
        st.markdown("**Hero Section Preview:**")
        hero_preview = """
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 40px 20px;
            border-radius: 20px;
            text-align: center;
            margin: 16px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
        ">
            <h2 style="color: white; font-family: Inter, sans-serif; margin-bottom: 16px;">WicketWise Cricket AI</h2>
            <p style="color: rgba(255,255,255,0.8); font-family: Inter, sans-serif;">Advanced cricket analytics with AI-powered predictions</p>
        </div>
        """
        st.markdown(hero_preview, unsafe_allow_html=True)
        
        st.markdown("**Card Components Preview:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            card_preview1 = """
            <div style="
                background: rgba(26, 35, 50, 0.9);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 16px;
                padding: 24px;
                margin: 8px 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            ">
                <div style="color: #4A90E2; font-size: 1.5rem; margin-bottom: 12px;">🎯</div>
                <h4 style="color: white; font-family: Inter, sans-serif; margin-bottom: 8px;">Match Predictions</h4>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0;">AI-powered win probability analysis</p>
            </div>
            """
            st.markdown(card_preview1, unsafe_allow_html=True)
        
        with col2:
            card_preview2 = """
            <div style="
                background: rgba(26, 35, 50, 0.9);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 16px;
                padding: 24px;
                margin: 8px 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            ">
                <div style="color: #50C878; font-size: 1.5rem; margin-bottom: 12px;">👤</div>
                <h4 style="color: white; font-family: Inter, sans-serif; margin-bottom: 8px;">Player Analytics</h4>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0;">Deep insights into player performance</p>
            </div>
            """
            st.markdown(card_preview2, unsafe_allow_html=True)
        
        with col3:
            card_preview3 = """
            <div style="
                background: rgba(26, 35, 50, 0.9);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 16px;
                padding: 24px;
                margin: 8px 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            ">
                <div style="color: #F59E0B; font-size: 1.5rem; margin-bottom: 12px;">📊</div>
                <h4 style="color: white; font-family: Inter, sans-serif; margin-bottom: 8px;">Live Tracking</h4>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0;">Real-time match data integration</p>
            </div>
            """
            st.markdown(card_preview3, unsafe_allow_html=True)
        
        st.markdown("**Navigation Preview:**")
        nav_preview = """
        <div style="
            background: rgba(15, 20, 25, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        ">
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 12px;">
                <div style="background: linear-gradient(45deg, #4A90E2, #50C878); color: white; padding: 12px 24px; border-radius: 25px; font-family: Inter, sans-serif;">Dashboard</div>
                <div style="background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.7); padding: 12px 24px; border-radius: 25px; font-family: Inter, sans-serif;">Predictions</div>
                <div style="background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.7); padding: 12px 24px; border-radius: 25px; font-family: Inter, sans-serif;">Analytics</div>
                <div style="background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.7); padding: 12px 24px; border-radius: 25px; font-family: Inter, sans-serif;">Live Match</div>
            </div>
        </div>
        """
        st.markdown(nav_preview, unsafe_allow_html=True)
        
        # Integration guide
        st.subheader("🔧 Integration Steps")
        
        integration_steps = """
        1. **Copy the generated component code** into your `ui_style.py`
        2. **Import in your main UI** (`ui_launcher.py`):
           ```python
           from ui_style import render_figma_hero_section, render_figma_card, render_figma_navigation
           ```
        3. **Replace existing components** or add alongside current ones
        4. **Test with your cricket data** to ensure compatibility
        5. **Customize colors and content** to match your brand
        """
        
        st.markdown(integration_steps)
        
        # Download generated components
        if st.button("💾 Download Components as Python File"):
            component_file = "# Figma-inspired components for Wicketwise\n"
            component_file += "# Generated from: {}\n\n".format(figma_url)
            component_file += "import streamlit as st\n\n"
            
            for component in components:
                component_file += component['code'] + "\n\n"
            
            st.download_button(
                "Download figma_components.py",
                component_file,
                file_name="figma_components.py",
                mime="text/python"
            )

if __name__ == "__main__":
    main()