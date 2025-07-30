# Purpose: Demo win probability bar with gradient colors and responsive design
# Author: Phi1618 Cricket AI, Last Modified: 2025-01-17

import streamlit as st
from ui_style import render_win_probability_bar
from ui_theme import set_streamlit_theme

def main():
    """Main demo function showcasing the win probability bar"""
    
    # Apply global theme
    set_streamlit_theme()
    
    st.title("🎯 Win Probability Bar Demo")
    st.markdown("---")
    
    # Interactive demo
    st.header("Interactive Demo")
    
    # Probability slider
    probability = st.slider(
        "Select Win Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.01,
        help="Adjust the probability to see the bar and colors change"
    )
    
    # Render the probability bar
    render_win_probability_bar(probability)
    
    # Show the current values
    st.info(f"**Current Probability:** {probability:.3f} ({probability*100:.1f}%)")
    
    # Multiple examples
    st.header("Common Probability Examples")
    
    examples = [
        ("Strong Favorite", 0.85),
        ("Moderate Favorite", 0.65),
        ("Balanced Match", 0.50),
        ("Slight Underdog", 0.35),
        ("Heavy Underdog", 0.15)
    ]
    
    for title, prob in examples:
        st.subheader(title)
        render_win_probability_bar(prob)
        st.caption(f"Probability: {prob:.2f} ({prob*100:.1f}%)")
        st.markdown("---")
    
    # Edge cases
    st.header("Edge Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Minimum (0%)")
        render_win_probability_bar(0.0)
        st.caption("No chance of winning")
    
    with col2:
        st.subheader("Maximum (100%)")
        render_win_probability_bar(1.0)
        st.caption("Guaranteed win")
    
    # Feature showcase
    st.header("🎨 Key Features")
    
    features = [
        "**Gradient Colors**: Smooth red-to-green gradient based on probability",
        "**Hover Tooltip**: Shows 'Model win prob: XX%' on hover",
        "**Bold Inline Label**: Percentage displayed prominently",
        "**Responsive Design**: Max width capped, fills container",
        "**Accessibility**: Screen reader friendly with title attributes",
        "**Theme Integration**: Uses ui_theme.py color system",
        "**Input Validation**: Handles invalid values gracefully",
        "**Smooth Animation**: CSS transitions for visual appeal"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    # Technical details
    st.header("🔧 Technical Implementation")
    
    st.markdown("""
    **Function Signature:**
    ```python
    def render_win_probability_bar(prob: float) -> None:
        # prob: Win probability as float between 0.0 and 1.0
    ```
    
    **Key Components:**
    - **Validation**: Automatically clamps probability between 0.0 and 1.0
    - **CSS Integration**: Uses `get_win_bar_style()` from ui_theme.py
    - **Gradient Definition**: Linear gradient from red (#dc3545) to green (#28a745)
    - **Responsive Layout**: Uses flexbox and percentage-based widths
    - **Accessibility**: Title attributes for screen readers
    """)
    
    # Usage examples
    st.header("💻 Usage Examples")
    
    st.code("""
# Basic usage
from ui_style import render_win_probability_bar

# Render with different probabilities
render_win_probability_bar(0.75)  # 75% chance
render_win_probability_bar(0.33)  # 33% chance
render_win_probability_bar(0.91)  # 91% chance

# Handles edge cases gracefully
render_win_probability_bar(0.0)   # Minimum
render_win_probability_bar(1.0)   # Maximum
render_win_probability_bar(-0.5)  # Invalid (becomes 0.0)
render_win_probability_bar(1.5)   # Invalid (becomes 1.0)
""")
    
    st.markdown("---")
    st.caption("Enhanced win probability bar with responsive design and gradient colors")

if __name__ == "__main__":
    main() 