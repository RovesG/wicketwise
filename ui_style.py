# Purpose: Provides functions to render sophisticated, modern UI components.
# Author: Assistant, Last Modified: 2024-07-19

import streamlit as st

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