# Purpose: Guide for integrating Figma designs into Wicketwise Streamlit app
# Author: Assistant, Last Modified: 2024-12-19

import streamlit as st
import base64
from pathlib import Path

def convert_figma_to_streamlit():
    """
    Complete guide for converting Figma designs to Streamlit components
    """
    
    # Step 1: Extract Figma Design Tokens
    figma_design_tokens = {
        "colors": {
            "primary": "#4A90E2",
            "secondary": "#F5A623",
            "background": "#1A2332",
            "surface": "#252B37",
            "text_primary": "#FFFFFF",
            "text_secondary": "#B8BCC8"
        },
        "typography": {
            "font_family": "'Inter', sans-serif",
            "heading_1": {"size": "2rem", "weight": "700"},
            "heading_2": {"size": "1.5rem", "weight": "600"},
            "body": {"size": "1rem", "weight": "400"}
        },
        "spacing": {
            "xs": "4px",
            "sm": "8px", 
            "md": "16px",
            "lg": "24px",
            "xl": "32px"
        },
        "borders": {
            "radius": {
                "sm": "8px",
                "md": "16px",
                "lg": "24px"
            },
            "width": "1px"
        }
    }
    
    return figma_design_tokens

def create_figma_component_from_html(html_content: str, css_content: str):
    """
    Convert exported Figma HTML/CSS into Streamlit component
    """
    
    # Wrap in Streamlit-compatible format
    streamlit_component = f"""
    <style>
    {css_content}
    </style>
    <div class="figma-component">
        {html_content}
    </div>
    """
    
    return streamlit_component

def integrate_figma_assets():
    """
    How to integrate Figma-exported assets (images, icons, etc.)
    """
    
    # Create assets directory structure
    assets_structure = {
        "ui/assets/": [
            "images/",
            "icons/", 
            "fonts/",
            "styles/"
        ]
    }
    
    # Asset loading helper
    def load_figma_asset(asset_path: str) -> str:
        """Load and encode asset for Streamlit"""
        try:
            with open(asset_path, "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data).decode()
                return f"data:image/png;base64,{encoded}"
        except FileNotFoundError:
            st.error(f"Asset not found: {asset_path}")
            return ""
    
    return load_figma_asset

# Example: Converting a Figma card component
def figma_card_component(title: str, content: str, figma_styles: dict):
    """
    Example of how to create a Streamlit component from Figma design
    """
    
    card_html = f"""
    <div style="
        background: {figma_styles['colors']['surface']};
        border-radius: {figma_styles['borders']['radius']['md']};
        padding: {figma_styles['spacing']['lg']};
        margin: {figma_styles['spacing']['md']} 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    ">
        <h3 style="
            font-family: {figma_styles['typography']['font_family']};
            font-size: {figma_styles['typography']['heading_2']['size']};
            font-weight: {figma_styles['typography']['heading_2']['weight']};
            color: {figma_styles['colors']['text_primary']};
            margin: 0 0 {figma_styles['spacing']['md']} 0;
        ">{title}</h3>
        <p style="
            font-family: {figma_styles['typography']['font_family']};
            font-size: {figma_styles['typography']['body']['size']};
            color: {figma_styles['colors']['text_secondary']};
            margin: 0;
            line-height: 1.6;
        ">{content}</p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

if __name__ == "__main__":
    st.title("🎨 Figma Integration Guide")
    
    # Demo the conversion process
    figma_tokens = convert_figma_to_streamlit()
    
    st.subheader("Design Tokens Extracted:")
    st.json(figma_tokens)
    
    st.subheader("Example Figma Component:")
    figma_card_component(
        "Sample Cricket Card", 
        "This component was created from Figma design tokens",
        figma_tokens
    )