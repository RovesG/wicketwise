# Responsive Layout Helper System

## Overview

A comprehensive responsive layout system for cricket analysis dashboards that automatically adapts to different screen sizes using Streamlit components and CSS media queries. The system provides three layout modes: Wide Desktop, Medium Tablet, and Mobile, with automatic detection based on container width.

## Key Components

### 1. Layout Modes (`LayoutMode` Enum)
- **WIDE_DESKTOP**: Side-by-side cards, full feature display
- **MEDIUM_TABLET**: Stacked cards, visible sidebar
- **MOBILE**: Vertical stack, collapsible chat

### 2. Breakpoint Configuration (`BreakpointConfig`)
- **Mobile**: ≤ 768px
- **Tablet**: 769px - 1024px  
- **Desktop**: ≥ 1025px

### 3. Core Functions

#### `render_responsive_dashboard()`
Main function that renders adaptive dashboard layouts:
```python
render_responsive_dashboard(
    video_component=video_func,
    player_cards=[batter_card, bowler_card],
    win_probability=win_prob_func,
    chat_component=chat_func,
    additional_components=[tools_func],
    layout_mode=LayoutMode.WIDE_DESKTOP,  # Optional override
    container_width=1200  # Optional width detection
)
```

#### `detect_layout_mode(container_width)`
Automatically detects appropriate layout mode:
```python
layout_mode = detect_layout_mode(900)  # Returns LayoutMode.MEDIUM_TABLET
```

#### `create_component_wrapper(component_func, **kwargs)`
Creates reusable component wrappers:
```python
batter_card = create_component_wrapper(render_batter_card, player_id="kohli")
```

## Layout Behavior

### Wide Desktop Layout (≥1025px)
- **Video**: Left column (2/3 width)
- **Win Probability**: Right column (1/3 width)
- **Player Cards**: Side-by-side in flexible grid
- **Chat**: Bottom left (2/3 width)
- **Additional Tools**: Bottom right (1/3 width)

### Medium Tablet Layout (769px-1024px)
- **Video**: Full width, stacked
- **Win Probability**: Full width, below video
- **Player Cards**: Two columns
- **Chat**: Expandable section (default expanded)
- **Additional Tools**: Expandable section (default collapsed)

### Mobile Layout (≤768px)
- **Video**: Full width, top priority
- **Win Probability**: Full width, below video
- **Player Cards**: Single column, vertically stacked
- **Chat**: Collapsible expander (default collapsed)
- **Additional Tools**: Collapsible expander (default collapsed)

## CSS Features

### Responsive Utilities
- `.responsive-container`: Full-width container with overflow handling
- `.layout-section`: Consistent section padding and styling
- `.side-by-side`: Flexbox layout for desktop cards
- `.collapsible-chat`: Smooth transitions for chat toggling

### Media Queries
Automatic CSS injection with breakpoint-specific styles:
```css
@media (max-width: 768px) {
    .mobile-layout { display: block !important; }
    .desktop-layout { display: none !important; }
}
```

## Usage Examples

### Basic Implementation
```python
import streamlit as st
from layout_utils import render_responsive_dashboard, create_component_wrapper

# Define component functions
def video_player():
    st.video("match_highlights.mp4")

def batter_stats():
    st.metric("Runs", "89", "12")

def bowler_stats():
    st.metric("Wickets", "3", "1")

def win_probability():
    st.progress(0.67)

def chat_interface():
    st.text_input("Ask about cricket...")

# Create component wrappers
player_cards = [
    create_component_wrapper(batter_stats),
    create_component_wrapper(bowler_stats)
]

# Render responsive dashboard
render_responsive_dashboard(
    video_component=video_player,
    player_cards=player_cards,
    win_probability=win_probability,
    chat_component=chat_interface
)
```

### Advanced Usage with Layout Override
```python
# Force mobile layout for testing
render_responsive_dashboard(
    video_component=video_player,
    player_cards=player_cards,
    win_probability=win_probability,
    chat_component=chat_interface,
    layout_mode=LayoutMode.MOBILE
)
```

### Width-Based Detection
```python
# Simulate different screen sizes
render_responsive_dashboard(
    video_component=video_player,
    player_cards=player_cards,
    win_probability=win_probability,
    chat_component=chat_interface,
    container_width=800  # Will use tablet layout
)
```

## Layout Recommendations

Get optimized settings for each layout mode:
```python
from layout_utils import get_layout_recommendations

desktop_config = get_layout_recommendations(LayoutMode.WIDE_DESKTOP)
# Returns: {
#   "max_columns": 4,
#   "card_width": "auto",
#   "sidebar_visible": True,
#   "video_aspect_ratio": "16:9",
#   "chat_height": "400px"
# }
```

## Testing

### Run All Tests
```bash
python3 -m pytest tests/test_layout_utils.py -v
```

### Test Coverage
- ✅ 25 comprehensive tests
- ✅ Layout mode detection
- ✅ CSS generation and injection
- ✅ Component rendering
- ✅ Responsive dashboard integration
- ✅ Component wrapper functionality
- ✅ Layout recommendations
- ✅ Breakpoint boundary testing

## Demo Application

Run the interactive demo:
```bash
streamlit run demo_responsive_layout.py
```

### Demo Features
- **Layout Mode Selector**: Test different layouts manually
- **Container Width Simulator**: Simulate various screen sizes
- **Layout Recommendations**: View optimal settings for each mode
- **Interactive Components**: Sample video, player cards, win probability, chat
- **Debug Information**: Real-time layout mode and width display

## Architecture Benefits

### 1. Mobile-First Design
- Prioritizes mobile experience
- Progressive enhancement for larger screens
- Optimal content ordering on small screens

### 2. No Hardcoded Breakpoints
- Configurable breakpoint system
- CSS-based responsive behavior
- Easy to modify for different requirements

### 3. Component Flexibility
- Function-based component system
- Reusable component wrappers
- Easy to integrate existing components

### 4. Automatic Adaptation
- Intelligent layout detection
- Seamless transitions between modes
- Consistent user experience across devices

## Integration Guidelines

### With Existing Cricket Analysis Tools
```python
# Import existing components
from ui_style import render_batter_card, render_bowler_card
from ui_style import render_win_probability_bar

# Create wrappers
player_cards = [
    create_component_wrapper(render_batter_card, player_data=batter_stats),
    create_component_wrapper(render_bowler_card, player_data=bowler_stats)
]

# Use with responsive system
render_responsive_dashboard(
    video_component=video_player,
    player_cards=player_cards,
    win_probability=lambda: render_win_probability_bar(0.67),
    chat_component=chat_interface
)
```

### Custom Component Integration
```python
# Custom component with arguments
def custom_scorecard(team_name, score, overs):
    st.subheader(f"{team_name}: {score}/{overs}")

# Wrap with pre-configured arguments
scorecard_wrapper = create_component_wrapper(
    custom_scorecard,
    team_name="Mumbai Indians",
    score="187/5",
    overs="20.0"
)
```

## Performance Considerations

### CSS Optimization
- Single CSS injection per page
- Efficient media queries
- Minimal DOM manipulation

### Component Rendering
- Lazy component evaluation
- Efficient column management
- Optimized context manager usage

### Memory Management
- Lightweight component wrappers
- Efficient layout detection
- Minimal state management

## Future Enhancements

### Planned Features
- Dynamic breakpoint adjustment
- Layout transition animations
- Advanced component positioning
- Touch-friendly mobile interactions
- Accessibility improvements

### Extensibility
- Custom layout mode support
- Plugin architecture for components
- Theme integration
- Multi-language support

## Summary

The responsive layout helper system provides a comprehensive solution for creating adaptive cricket analysis dashboards. With automatic layout detection, CSS-based responsive behavior, and flexible component integration, it ensures optimal user experience across all device types while maintaining clean, maintainable code architecture.

**Key Statistics:**
- 3 layout modes (Desktop, Tablet, Mobile)
- 25 comprehensive tests (100% pass rate)
- 400+ lines of production code
- Full CSS media query support
- Component wrapper system
- Debug and development tools 