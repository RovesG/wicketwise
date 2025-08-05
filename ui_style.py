# Purpose: Provides functions to render sophisticated, modern UI components.
# Author: Assistant, Last Modified: 2024-12-19
# Updated: Added Figma-inspired components from meta-ethics-63199039.figma.site

import streamlit as st

# =============================================================================
# FIGMA-INSPIRED COMPONENTS
# From: https://meta-ethics-63199039.figma.site
# =============================================================================

def render_figma_hero_section(title: str, subtitle: str, cta_text: str = "Get Started"):
    """
    Hero section component inspired by meta-ethics Figma design
    Integrates with existing Wicketwise dark theme
    """
    
    # Use Streamlit's native container and columns for better compatibility
    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 60px 40px;
            border-radius: 20px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.1);
        ">
            <h1 style="
                font-family: 'Inter', sans-serif;
                font-size: 2.5rem;
                font-weight: 700;
                color: #FFFFFF;
                margin-bottom: 20px;
                line-height: 1.2;
            ">{title}</h1>
            
            <p style="
                font-family: 'Inter', sans-serif;
                font-size: 1.2rem;
                color: rgba(255,255,255,0.85);
                margin-bottom: 30px;
                line-height: 1.6;
            ">{subtitle}</p>
            
            <div style="
                display: inline-block;
                background: linear-gradient(45deg, #4A90E2, #50C878);
                color: #FFFFFF;
                padding: 15px 30px;
                border-radius: 25px;
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                box-shadow: 0 6px 25px rgba(74,144,226,0.4);
            ">{cta_text}</div>
        </div>
        """, unsafe_allow_html=True)

def render_figma_card(title: str, content: str, icon: str = "🏏", color: str = "#4A90E2"):
    """
    Card component inspired by meta-ethics Figma design
    Enhanced with cricket-specific styling
    """
    
    # Use Streamlit's native container for better compatibility
    with st.container():
        st.markdown(f"""
        <div style="
            background: rgba(26, 35, 50, 0.9);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            border-top: 3px solid {color};
        ">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="
                    font-size: 1.8rem;
                    margin-right: 12px;
                    padding: 8px;
                    background: rgba(74,144,226,0.1);
                    border-radius: 8px;
                ">{icon}</div>
                <h3 style="
                    font-family: 'Inter', sans-serif;
                    color: #FFFFFF;
                    font-weight: 600;
                    margin: 0;
                    font-size: 1.2rem;
                ">{title}</h3>
            </div>
            <p style="
                font-family: 'Inter', sans-serif;
                color: rgba(255,255,255,0.8);
                margin: 0;
                line-height: 1.5;
                font-size: 0.95rem;
            ">{content}</p>
        </div>
        """, unsafe_allow_html=True)

def render_figma_navigation(nav_items: list, active_item: str = ""):
    """
    Navigation component inspired by meta-ethics Figma design
    Optimized for cricket analytics navigation
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
        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 12px;">
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
                min-width: 120px;
                text-align: center;
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

def render_figma_stats_panel(stats: dict, title: str = "Match Statistics"):
    """
    Statistics panel inspired by Figma design with cricket-specific enhancements
    """
    
    panel_html = f"""
    <div style="
        background: rgba(26, 35, 50, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 32px;
        margin: 16px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    ">
        <h3 style="
            font-family: 'Inter', sans-serif;
            color: #FFFFFF;
            font-weight: 600;
            margin-bottom: 24px;
            font-size: 1.5rem;
            text-align: center;
        ">{title}</h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px;">
    """
    
    for stat_name, stat_value in stats.items():
        panel_html += f"""
            <div style="
                background: rgba(37, 43, 55, 0.8);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 1px solid rgba(58, 63, 75, 0.8);
                transition: all 0.3s ease;
            " onmouseover="this.style.background='rgba(74,144,226,0.1)'; this.style.borderColor='rgba(74,144,226,0.3)'"
               onmouseout="this.style.background='rgba(37, 43, 55, 0.8)'; this.style.borderColor='rgba(58, 63, 75, 0.8)'">
                <div style="
                    font-family: 'Inter', sans-serif;
                    color: #4A90E2;
                    font-size: 2rem;
                    font-weight: 700;
                    margin-bottom: 8px;
                ">{stat_value}</div>
                <div style="
                    font-family: 'Inter', sans-serif;
                    color: rgba(255,255,255,0.7);
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                ">{stat_name}</div>
            </div>
        """
    
    panel_html += """
        </div>
    </div>
    """
    
    st.markdown(panel_html, unsafe_allow_html=True)

# =============================================================================
# EXISTING WICKETWISE COMPONENTS (Enhanced with Figma styling)
# =============================================================================

def render_player_card(player_name: str, player_stats: dict, team_color: str = "#4A90E2"):
    """Renders a sophisticated, modern player card with glass morphism effect."""
    card_style = f"""
        <div style="
            background: rgba(26, 35, 50, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(74, 144, 226, 0.2);
            border-radius: 16px;
            border-left: 4px solid {team_color};
            padding: 24px;
            margin-bottom: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <div style="
                    width: 8px;
                    height: 8px;
                    background: {team_color};
                    border-radius: 50%;
                    margin-right: 12px;
                "></div>
                <h3 style="
                    font-family: 'Inter', sans-serif; 
                    color: #FFFFFF;
                    font-weight: 600;
                    margin: 0;
                    font-size: 1.25rem;
                ">{player_name}</h3>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 16px;">
    """
    
    for stat, value in player_stats.items():
        card_style += f"""
            <div style="
                background: rgba(37, 43, 55, 0.6);
                border-radius: 8px;
                padding: 12px;
                text-align: center;
                border: 1px solid rgba(58, 63, 75, 0.8);
            ">
                <p style="
                    font-family: 'Inter', sans-serif; 
                    color: #B8BCC8; 
                    margin: 0 0 4px 0;
                    font-size: 0.75rem;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                ">{stat.replace('_', ' ')}</p>
                <p style="
                    font-family: 'Inter', sans-serif; 
                    font-weight: 700; 
                    color: {team_color}; 
                    margin: 0; 
                    font-size: 1.5rem;
                ">{value}</p>
            </div>
        """
    card_style += "</div></div>"
    st.markdown(card_style, unsafe_allow_html=True)

def render_win_probability_bar(home_prob: float, away_prob: float, home_team: str = "Home", away_team: str = "Away"):
    """Renders a sophisticated win probability bar with smooth gradients."""
    bar_style = f"""
        <div style="
            background: rgba(26, 35, 50, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(58, 63, 75, 0.8);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            ">
                <span style="
                    font-family: 'Inter', sans-serif;
                    color: #FFFFFF;
                    font-weight: 600;
                    font-size: 0.9rem;
                ">{home_team}</span>
                <span style="
                    font-family: 'Inter', sans-serif;
                    color: #B8BCC8;
                    font-size: 0.8rem;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                ">Win Probability</span>
                <span style="
                    font-family: 'Inter', sans-serif;
                    color: #FFFFFF;
                    font-weight: 600;
                    font-size: 0.9rem;
                ">{away_team}</span>
            </div>
            <div style="
                background: rgba(37, 43, 55, 0.8);
                border-radius: 8px;
                height: 12px;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    background: linear-gradient(90deg, #10B981 0%, #4A90E2 {home_prob}%, #EF4444 100%);
                    height: 100%;
                    border-radius: 8px;
                "></div>
            </div>
            <div style="
                display: flex;
                justify-content: space-between;
                margin-top: 8px;
            ">
                <span style="
                    font-family: 'Inter', sans-serif;
                    color: #10B981;
                    font-weight: 600;
                    font-size: 1rem;
                ">{home_prob:.1f}%</span>
                <span style="
                    font-family: 'Inter', sans-serif;
                    color: #EF4444;
                    font-weight: 600;
                    font-size: 1rem;
                ">{away_prob:.1f}%</span>
            </div>
        </div>
    """
    st.markdown(bar_style, unsafe_allow_html=True)

def render_odds_panel(market_odds: dict, model_odds: dict):
    """Renders a sophisticated odds comparison panel."""
    delta = model_odds['home_win'] - market_odds['home_win']
    delta_color = '#10B981' if delta > 0 else '#EF4444'
    delta_bg = 'rgba(16, 185, 129, 0.1)' if delta > 0 else 'rgba(239, 68, 68, 0.1)'

    panel_style = f"""
        <div style="
            background: rgba(26, 35, 50, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(58, 63, 75, 0.8);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        ">
            <h4 style="
                text-align: center; 
                font-family: 'Inter', sans-serif;
                color: #FFFFFF;
                font-weight: 600;
                margin: 0 0 20px 0;
                font-size: 1.1rem;
            ">Odds Comparison</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
                <div style="
                    background: rgba(37, 43, 55, 0.6);
                    border-radius: 8px;
                    padding: 16px;
                    text-align: center;
                    border: 1px solid rgba(58, 63, 75, 0.8);
                ">
                    <p style="
                        font-family: 'Inter', sans-serif; 
                        color: #B8BCC8;
                        margin: 0 0 8px 0;
                        font-size: 0.8rem;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">Market</p>
                    <p style="
                        font-family: 'Inter', sans-serif; 
                        font-size: 1.8rem; 
                        color: #FFFFFF;
                        font-weight: 700;
                        margin: 0;
                    ">{market_odds['home_win']:.2f}</p>
                </div>
                
                <div style="
                    background: rgba(37, 43, 55, 0.6);
                    border-radius: 8px;
                    padding: 16px;
                    text-align: center;
                    border: 1px solid rgba(74, 144, 226, 0.3);
                ">
                    <p style="
                        font-family: 'Inter', sans-serif; 
                        color: #B8BCC8;
                        margin: 0 0 8px 0;
                        font-size: 0.8rem;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">Model</p>
                    <p style="
                        font-family: 'Inter', sans-serif; 
                        font-size: 1.8rem; 
                        color: #4A90E2;
                        font-weight: 700;
                        margin: 0;
                    ">{model_odds['home_win']:.2f}</p>
                </div>
            </div>
            
            <div style="
                background: {delta_bg};
                border: 1px solid {delta_color}40;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            ">
                <p style="
                    color: {delta_color}; 
                    font-family: 'Inter', sans-serif;
                    font-weight: 600;
                    margin: 0;
                    font-size: 1rem;
                ">
                    Delta: {delta:+.2f}
                </p>
            </div>
        </div>
    """
    st.markdown(panel_style, unsafe_allow_html=True)

def render_chat_bubble(message: str, role: str):
    """Renders a modern chat bubble with sophisticated styling."""
    if role == "assistant":
        bubble_style = """
            display: flex;
            justify-content: flex-start;
            margin-bottom: 16px;
        """
        content_style = """
            background: rgba(74, 144, 226, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(74, 144, 226, 0.3);
            color: #FFFFFF;
            border-radius: 16px 16px 16px 4px;
            padding: 16px 20px;
            max-width: 80%;
            box-shadow: 0 4px 16px rgba(74, 144, 226, 0.2);
        """
        icon_style = """
            background: linear-gradient(135deg, #4A90E2, #357ABD);
            color: #FFFFFF;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-right: 12px;
            margin-top: 4px;
            box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
        """
        icon = "Φ"
    else:  # user
        bubble_style = """
            display: flex;
            justify-content: flex-end;
            margin-bottom: 16px;
        """
        content_style = """
            background: rgba(37, 43, 55, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(58, 63, 75, 0.8);
            color: #FFFFFF;
            border-radius: 16px 16px 4px 16px;
            padding: 16px 20px;
            max-width: 80%;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        """
        icon_style = """
            background: linear-gradient(135deg, #50C878, #059669);
            color: #FFFFFF;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-left: 12px;
            margin-top: 4px;
            box-shadow: 0 2px 8px rgba(80, 200, 120, 0.3);
        """
        icon = "👤"

    st.markdown(f"""
        <div style="{bubble_style}">
            {'<div style="' + icon_style + '">' + icon + '</div>' if role == "assistant" else ''}
            <div style="{content_style}; font-family: 'Inter', sans-serif; line-height: 1.5;">
                {message}
            </div>
            {'<div style="' + icon_style + '">' + icon + '</div>' if role == "user" else ''}
        </div>
    """, unsafe_allow_html=True) 