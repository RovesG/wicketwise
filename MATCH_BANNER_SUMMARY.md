# Match Info Banner (Context Bar) Implementation

## Overview

A responsive match information banner component that displays real-time cricket match data including scoreline, overs, innings, and match phase. The banner features a dark gradient background with dynamic team-colored borders and uses cricket emojis for visual appeal.

## Key Features

### 🎨 **Visual Design**
- **Dark Gradient Background**: Modern dark grey gradient (#2c3e50 to #34495e)
- **Team-Colored Borders**: Dynamic border colors based on current batting team
- **Cricket Emojis**: 🏏 (overs), 🎯 (innings), ⚡ (match phase)
- **Responsive Layout**: CSS flexbox with media queries for mobile optimization

### 📱 **Responsive Behavior**
- **Desktop**: Horizontal layout with teams side-by-side and status on right
- **Mobile**: Vertical stack with centered alignment and compact status items
- **Tablet**: Balanced layout with appropriate spacing

### 🏏 **Cricket-Specific Features**
- **Live Score Display**: Team scores in format "Team Name 187/5"
- **Over Information**: Current over and ball (e.g., "Over 7.3")
- **Innings Tracking**: Clear indication of current innings
- **Match Phase**: Visual indicators for Powerplay, Middle Overs, Death Overs, etc.
- **Team Color Integration**: Borders change based on batting team

## Function Signature

```python
def render_match_banner(match_info: dict) -> None:
    """
    Render a responsive match info banner with scoreline, overs, innings, and phase.
    
    Args:
        match_info: Dictionary containing match data
    """
```

## Data Structure

### Required Fields
```python
match_info = {
    'team1_name': 'Mumbai Indians',        # Team 1 name
    'team1_score': 187,                   # Team 1 score
    'team1_wickets': 5,                   # Team 1 wickets
    'team2_name': 'Chennai Super Kings',  # Team 2 name
    'team2_score': 45,                    # Team 2 score
    'team2_wickets': 2,                   # Team 2 wickets
    'current_over': 7,                    # Current over
    'current_ball': 3,                    # Current ball
    'current_innings': 2,                 # Current innings (1 or 2)
    'match_phase': 'Powerplay',           # Match phase
    'team1_color': '#004BA0',             # Team 1 color
    'team2_color': '#FFFF3C'              # Team 2 color
}
```

### Fallback Values
- **Team Names**: 'Team 1', 'Team 2'
- **Scores**: 0/0
- **Overs**: 0.0
- **Innings**: 1
- **Phase**: 'In Progress'
- **Colors**: CricketColors.BATTING, CricketColors.BOWLING

## Usage Examples

### Basic Usage
```python
from ui_style import render_match_banner

# Live match example
match_data = {
    'team1_name': 'Mumbai Indians',
    'team1_score': 187,
    'team1_wickets': 5,
    'team2_name': 'Chennai Super Kings',
    'team2_score': 45,
    'team2_wickets': 2,
    'current_over': 7,
    'current_ball': 3,
    'current_innings': 2,
    'match_phase': 'Powerplay'
}

render_match_banner(match_data)
```

### Minimal Data
```python
# Banner with minimal data (uses fallbacks)
minimal_match = {
    'team1_name': 'Team A',
    'team2_name': 'Team B'
}

render_match_banner(minimal_match)
```

### Different Match Phases
```python
# Death overs scenario
death_overs = {
    'team1_name': 'Rajasthan Royals',
    'team1_score': 165,
    'team1_wickets': 7,
    'team2_name': 'Sunrisers Hyderabad',
    'team2_score': 142,
    'team2_wickets': 4,
    'current_over': 18,
    'current_ball': 2,
    'current_innings': 2,
    'match_phase': 'Death Overs',
    'team1_color': '#254AA5',
    'team2_color': '#FF822A'
}

render_match_banner(death_overs)
```

## CSS Implementation

### Responsive Design
```css
/* Mobile styles */
@media (max-width: 768px) {
    .match-banner-content {
        flex-direction: column;
        align-items: stretch;
    }
    
    .team-score {
        justify-content: center;
        min-width: auto;
    }
    
    .match-status {
        justify-content: center;
        gap: 0.5rem;
    }
}

/* Desktop styles */
@media (min-width: 769px) {
    .match-banner-content {
        flex-direction: row;
    }
    
    .team-scores {
        display: flex;
        align-items: center;
        gap: 1rem;
        flex: 2;
    }
    
    .match-status {
        flex: 1;
        justify-content: flex-end;
    }
}
```

### Team Color Integration
```css
.match-banner {
    border-left: 4px solid {current_team_color};
    border-right: 4px solid {current_team_color};
}

.team-name {
    color: {current_team_color};
}
```

## Testing

### Test Coverage
- ✅ **15 comprehensive tests** with 100% pass rate
- ✅ **Basic rendering** with mock Streamlit components
- ✅ **Content validation** for team names, scores, overs
- ✅ **Emoji presence** verification
- ✅ **Element ordering** tests
- ✅ **CSS styling** validation
- ✅ **Team color** logic testing
- ✅ **Fallback values** handling
- ✅ **Different match phases** testing
- ✅ **Responsive CSS** validation
- ✅ **Edge cases** handling

### Run Tests
```bash
python3 -m pytest tests/test_ui_match_banner.py -v
```

## Demo Application

### Features
- **5 Match Scenarios**: Live match, first innings, death overs, super over, all out
- **Interactive Builder**: Customize teams, scores, colors, and match status
- **Real-time Preview**: See changes instantly
- **Configuration View**: JSON export of current settings

### Run Demo
```bash
streamlit run demo_match_banner.py
```

## Technical Architecture

### Component Structure
1. **Data Extraction**: Extract match data with fallback values
2. **Color Logic**: Determine current team color based on innings
3. **CSS Generation**: Create responsive CSS with team colors
4. **HTML Construction**: Build banner HTML structure
5. **Streamlit Rendering**: Inject CSS and HTML into Streamlit

### Performance Considerations
- **Single CSS Injection**: Efficient CSS generation
- **Minimal DOM**: Lightweight HTML structure
- **Responsive Design**: No JavaScript dependencies
- **Fallback Handling**: Graceful degradation

## Integration Guidelines

### With Existing Cricket Analysis Tools
```python
# Example integration with match simulator
from match_simulator import MatchSimulator
from ui_style import render_match_banner

simulator = MatchSimulator()
current_match = simulator.get_current_match_state()

# Map simulator data to banner format
banner_data = {
    'team1_name': current_match.team1.name,
    'team1_score': current_match.team1.score,
    'team1_wickets': current_match.team1.wickets,
    'team2_name': current_match.team2.name,
    'team2_score': current_match.team2.score,
    'team2_wickets': current_match.team2.wickets,
    'current_over': current_match.current_over,
    'current_ball': current_match.current_ball,
    'current_innings': current_match.current_innings,
    'match_phase': current_match.phase,
    'team1_color': current_match.team1.color,
    'team2_color': current_match.team2.color
}

render_match_banner(banner_data)
```

### With Responsive Layout System
```python
from layout_utils import render_responsive_dashboard, create_component_wrapper
from ui_style import render_match_banner

# Create banner wrapper
banner_wrapper = create_component_wrapper(
    render_match_banner,
    match_info=current_match_data
)

# Use in responsive dashboard
render_responsive_dashboard(
    video_component=video_player,
    player_cards=player_cards,
    win_probability=win_probability,
    chat_component=chat_interface,
    additional_components=[banner_wrapper]
)
```

## Match Phase Support

### Supported Phases
- **Powerplay**: First 6 overs with field restrictions
- **Middle Overs**: Overs 7-15 in T20 format
- **Death Overs**: Final 5 overs (16-20)
- **Super Over**: Tie-breaker scenario
- **All Out**: Team has lost all wickets

### Custom Phases
```python
# Add custom match phases
custom_phases = {
    'Rain Delay': '🌧️',
    'Strategic Timeout': '⏱️',
    'Review': '🔍',
    'Drinks Break': '🥤'
}
```

## Best Practices

### Data Validation
```python
def validate_match_data(match_info):
    """Validate match data before rendering"""
    required_fields = ['team1_name', 'team2_name']
    
    for field in required_fields:
        if field not in match_info:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate score ranges
    if match_info.get('team1_score', 0) < 0:
        raise ValueError("Team scores cannot be negative")
    
    return True
```

### Error Handling
```python
try:
    render_match_banner(match_data)
except Exception as e:
    st.error(f"Error rendering match banner: {e}")
    # Render fallback banner
    render_match_banner({
        'team1_name': 'Team 1',
        'team2_name': 'Team 2'
    })
```

### Performance Tips
- **Cache Match Data**: Avoid frequent re-renders
- **Minimize Color Changes**: Reduce CSS regeneration
- **Use Fallbacks**: Always provide default values
- **Test Responsiveness**: Verify mobile layouts

## Future Enhancements

### Planned Features
- **Animation Support**: Smooth transitions for score updates
- **Real-time Updates**: WebSocket integration for live matches
- **Team Logos**: Image support for team branding
- **Multiple Formats**: Support for ODI, Test matches
- **Accessibility**: Screen reader support and keyboard navigation

### Extensibility
- **Custom Themes**: Support for different color schemes
- **Configurable Layout**: Adjustable banner dimensions
- **Plugin Architecture**: Support for additional match data
- **Internationalization**: Multi-language support

## Summary

The match banner component provides a comprehensive solution for displaying real-time cricket match information with:

**Key Statistics:**
- 15 comprehensive tests (100% pass rate)
- Responsive design with CSS flexbox
- 5 different match scenarios in demo
- Interactive customization interface
- Full team color integration
- Cricket emoji support
- Fallback value handling

The implementation follows cricket analysis best practices while maintaining clean, maintainable code architecture with extensive testing and documentation. 