# Purpose: Demo enhanced player cards with hoverable containers and stat grids
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import streamlit as st
from ui_style import render_batter_card, render_bowler_card
from ui_theme import set_streamlit_theme, ThemeColors

def main():
    """Main demo function showcasing enhanced player cards"""
    
    # Apply global theme
    set_streamlit_theme()
    
    st.title("🏏 Enhanced Player Cards Demo")
    st.markdown("---")
    
    # Demo batter cards
    st.header("Batter Cards")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Virat Kohli")
        virat_data = {
            'name': 'Virat Kohli',
            'average': '45.2',
            'strike_rate': '142.3',
            'recent_shots': '4, 1, dot, 6, 2',
            'runs': 67,
            'balls_faced': 48,
            'image_url': 'https://via.placeholder.com/100x100/c8712d/ffffff?text=VK',
            'team_color': ThemeColors.BATTING,
            'highest_score': '183',
            'boundaries': '8'
        }
        render_batter_card(virat_data)
    
    with col2:
        st.subheader("Rohit Sharma")
        rohit_data = {
            'name': 'Rohit Sharma',
            'average': '32.1',
            'strike_rate': '139.7',
            'recent_shots': '6, 4, 2, dot, 1',
            'runs': 45,
            'balls_faced': 35,
            'team_color': '#ff6b35',
            'highest_score': '264',
            'boundaries': '6'
            # Note: No image_url - will show fallback placeholder
        }
        render_batter_card(rohit_data)
    
    # Demo bowler cards
    st.header("Bowler Cards")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Jasprit Bumrah")
        bumrah_data = {
            'name': 'Jasprit Bumrah',
            'economy': '6.8',
            'wickets': 3,
            'overs': '8.2',
            'runs_conceded': 56,
            'maidens': 1,
            'image_url': 'https://via.placeholder.com/100x100/002466/ffffff?text=JB',
            'team_color': ThemeColors.BOWLING,
            'best_figures': '4/17',
            'dot_balls': '14'
        }
        render_bowler_card(bumrah_data)
    
    with col4:
        st.subheader("Rashid Khan")
        rashid_data = {
            'name': 'Rashid Khan',
            'economy': '5.8',
            'wickets': 4,
            'overs': '10.0',
            'runs_conceded': 58,
            'maidens': 1,
            'team_color': '#ff4500',
            'best_figures': '5/27',
            'dot_balls': '24'
            # Note: No image_url - will show fallback placeholder
        }
        render_bowler_card(rashid_data)
    
    # Demo with failed image
    st.header("Error Handling Demo")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("MS Dhoni (Broken Image)")
        dhoni_data = {
            'name': 'MS Dhoni',
            'average': '38.0',
            'strike_rate': '135.0',
            'recent_shots': '2, 1, 4, dot, 6',
            'runs': 23,
            'balls_faced': 18,
            'image_url': 'https://broken.link/invalid.jpg',  # This will fail
            'team_color': '#ffcd00',
            'highest_score': '183*',
            'boundaries': '3'
        }
        render_batter_card(dhoni_data)
    
    with col6:
        st.subheader("Pat Cummins (Minimal Data)")
        cummins_data = {
            'name': 'Pat Cummins',
            'team_color': '#1e90ff'
            # Only name and team_color - all others will use fallbacks
        }
        render_bowler_card(cummins_data)
    
    # Feature explanation
    st.markdown("---")
    st.header("🎨 Enhanced Features")
    
    features = [
        "**Hoverable Containers**: Cards use cricket-card CSS classes with hover effects",
        "**Player Images**: Display with fallback to themed placeholders",
        "**6 Metrics Grid**: 2 rows × 3 columns layout (exceeds 4+ requirement)",
        "**Team Color Accents**: Dynamic color bars and gradients",
        "**Compact Layout**: Optimized vertical spacing",
        "**Error Handling**: Graceful fallbacks for missing/broken images",
        "**Responsive Design**: Adapts to different screen sizes",
        "**Accessibility**: Proper focus indicators and contrast ratios"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    st.markdown("---")
    st.caption("Enhanced player cards with ui_theme.py integration")

if __name__ == "__main__":
    main() 