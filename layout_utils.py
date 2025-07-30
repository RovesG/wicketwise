# Purpose: Responsive layout utilities for cricket analysis dashboard
# Author: Claude, Last Modified: 2025-01-17

import streamlit as st
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum


class LayoutMode(Enum):
    """Layout modes for different screen sizes"""
    WIDE_DESKTOP = "wide_desktop"
    MEDIUM_TABLET = "medium_tablet"
    MOBILE = "mobile"


class BreakpointConfig:
    """Configuration for responsive breakpoints"""
    MOBILE_MAX_WIDTH = 768
    TABLET_MAX_WIDTH = 1024
    DESKTOP_MIN_WIDTH = 1025
    
    @classmethod
    def get_css_breakpoints(cls) -> str:
        """Get CSS media queries for breakpoints"""
        return f"""
        /* Mobile styles */
        @media (max-width: {cls.MOBILE_MAX_WIDTH}px) {{
            .mobile-layout {{
                display: block !important;
            }}
            .desktop-layout {{
                display: none !important;
            }}
            .tablet-layout {{
                display: none !important;
            }}
            .stColumn {{
                width: 100% !important;
                margin-bottom: 1rem;
            }}
            .video-container {{
                order: -1;
                margin-bottom: 1.5rem;
            }}
            .chat-container {{
                order: 999;
            }}
        }}
        
        /* Tablet styles */
        @media (min-width: {cls.MOBILE_MAX_WIDTH + 1}px) and (max-width: {cls.TABLET_MAX_WIDTH}px) {{
            .tablet-layout {{
                display: block !important;
            }}
            .mobile-layout {{
                display: none !important;
            }}
            .desktop-layout {{
                display: none !important;
            }}
            .stColumn {{
                min-width: 300px;
                margin-bottom: 1rem;
            }}
        }}
        
        /* Desktop styles */
        @media (min-width: {cls.DESKTOP_MIN_WIDTH}px) {{
            .desktop-layout {{
                display: block !important;
            }}
            .mobile-layout {{
                display: none !important;
            }}
            .tablet-layout {{
                display: none !important;
            }}
            .side-by-side {{
                display: flex;
                gap: 1.5rem;
            }}
            .side-by-side > div {{
                flex: 1;
            }}
        }}
        
        /* Common responsive utilities */
        .responsive-container {{
            width: 100%;
            max-width: 100%;
            overflow-x: auto;
        }}
        
        .collapsible-chat {{
            transition: all 0.3s ease;
        }}
        
        .collapsible-chat.collapsed {{
            max-height: 60px;
            overflow: hidden;
        }}
        
        .layout-section {{
            margin-bottom: 1.5rem;
            padding: 1rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.02);
        }}
        
        .layout-section h3 {{
            margin-top: 0;
            margin-bottom: 1rem;
            color: #ffffff;
        }}
        """


def detect_layout_mode(container_width: Optional[int] = None) -> LayoutMode:
    """
    Detect the appropriate layout mode based on container width
    
    Args:
        container_width: Optional container width in pixels
        
    Returns:
        LayoutMode enum value
    """
    if container_width is None:
        # Default to tablet for server-side rendering
        return LayoutMode.MEDIUM_TABLET
    
    if container_width <= BreakpointConfig.MOBILE_MAX_WIDTH:
        return LayoutMode.MOBILE
    elif container_width <= BreakpointConfig.TABLET_MAX_WIDTH:
        return LayoutMode.MEDIUM_TABLET
    else:
        return LayoutMode.WIDE_DESKTOP


def inject_responsive_css():
    """Inject responsive CSS into the Streamlit app"""
    css = BreakpointConfig.get_css_breakpoints()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_wide_desktop_layout(
    video_component: Callable,
    player_cards: List[Callable],
    win_probability: Callable,
    chat_component: Callable,
    additional_components: Optional[List[Callable]] = None
) -> None:
    """
    Render layout for wide desktop screens
    
    Args:
        video_component: Function to render video player
        player_cards: List of functions to render player cards
        win_probability: Function to render win probability bar
        chat_component: Function to render chat interface
        additional_components: Optional additional components to render
    """
    st.markdown('<div class="desktop-layout">', unsafe_allow_html=True)
    
    # Top section: Video and win probability
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="layout-section">', unsafe_allow_html=True)
        st.subheader("📺 Match Video")
        video_component()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="layout-section">', unsafe_allow_html=True)
        st.subheader("📊 Win Probability")
        win_probability()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle section: Player cards side by side
    st.markdown('<div class="side-by-side">', unsafe_allow_html=True)
    
    card_cols = st.columns(len(player_cards))
    for i, (col, card_func) in enumerate(zip(card_cols, player_cards)):
        with col:
            st.markdown('<div class="layout-section">', unsafe_allow_html=True)
            card_func()
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom section: Chat and additional components
    bottom_cols = st.columns([2, 1])
    
    with bottom_cols[0]:
        st.markdown('<div class="layout-section">', unsafe_allow_html=True)
        st.subheader("💬 Chat Interface")
        chat_component()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with bottom_cols[1]:
        if additional_components:
            st.markdown('<div class="layout-section">', unsafe_allow_html=True)
            st.subheader("🔧 Additional Tools")
            for component in additional_components:
                component()
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_medium_tablet_layout(
    video_component: Callable,
    player_cards: List[Callable],
    win_probability: Callable,
    chat_component: Callable,
    additional_components: Optional[List[Callable]] = None
) -> None:
    """
    Render layout for medium/tablet screens
    
    Args:
        video_component: Function to render video player
        player_cards: List of functions to render player cards
        win_probability: Function to render win probability bar
        chat_component: Function to render chat interface
        additional_components: Optional additional components to render
    """
    st.markdown('<div class="tablet-layout">', unsafe_allow_html=True)
    
    # Top section: Video and win probability stacked
    st.markdown('<div class="layout-section">', unsafe_allow_html=True)
    st.subheader("📺 Match Video")
    video_component()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="layout-section">', unsafe_allow_html=True)
    st.subheader("📊 Win Probability")
    win_probability()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle section: Player cards in two columns
    st.subheader("👥 Player Information")
    if len(player_cards) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="layout-section">', unsafe_allow_html=True)
            player_cards[0]()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="layout-section">', unsafe_allow_html=True)
            player_cards[1]()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional cards below
        for card_func in player_cards[2:]:
            st.markdown('<div class="layout-section">', unsafe_allow_html=True)
            card_func()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        for card_func in player_cards:
            st.markdown('<div class="layout-section">', unsafe_allow_html=True)
            card_func()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom section: Chat with sidebar visible
    with st.expander("💬 Chat Interface", expanded=True):
        chat_component()
    
    # Additional components in sidebar or bottom
    if additional_components:
        with st.expander("🔧 Additional Tools", expanded=False):
            for component in additional_components:
                component()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_mobile_layout(
    video_component: Callable,
    player_cards: List[Callable],
    win_probability: Callable,
    chat_component: Callable,
    additional_components: Optional[List[Callable]] = None
) -> None:
    """
    Render layout for mobile screens
    
    Args:
        video_component: Function to render video player
        player_cards: List of functions to render player cards
        win_probability: Function to render win probability bar
        chat_component: Function to render chat interface
        additional_components: Optional additional components to render
    """
    st.markdown('<div class="mobile-layout">', unsafe_allow_html=True)
    
    # Top section: Video first (most important on mobile)
    st.markdown('<div class="layout-section video-container">', unsafe_allow_html=True)
    st.subheader("📺 Video")
    video_component()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Win probability bar
    st.markdown('<div class="layout-section">', unsafe_allow_html=True)
    st.subheader("📊 Win Probability")
    win_probability()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Player cards stacked vertically
    st.subheader("👥 Players")
    for i, card_func in enumerate(player_cards):
        st.markdown('<div class="layout-section">', unsafe_allow_html=True)
        card_func()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat interface - collapsible on mobile
    with st.expander("💬 Chat", expanded=False):
        st.markdown('<div class="collapsible-chat">', unsafe_allow_html=True)
        chat_component()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional components - minimal on mobile
    if additional_components:
        with st.expander("🔧 Tools", expanded=False):
            for component in additional_components:
                component()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_responsive_dashboard(
    video_component: Callable,
    player_cards: List[Callable],
    win_probability: Callable,
    chat_component: Callable,
    additional_components: Optional[List[Callable]] = None,
    layout_mode: Optional[LayoutMode] = None,
    container_width: Optional[int] = None
) -> None:
    """
    Render a responsive dashboard that adapts to different screen sizes
    
    Args:
        video_component: Function to render video player
        player_cards: List of functions to render player cards
        win_probability: Function to render win probability bar
        chat_component: Function to render chat interface
        additional_components: Optional additional components to render
        layout_mode: Optional layout mode override
        container_width: Optional container width for layout detection
    """
    # Inject responsive CSS
    inject_responsive_css()
    
    # Detect layout mode if not provided
    if layout_mode is None:
        layout_mode = detect_layout_mode(container_width)
    
    # Add responsive container wrapper
    st.markdown('<div class="responsive-container">', unsafe_allow_html=True)
    
    # Render appropriate layout
    if layout_mode == LayoutMode.WIDE_DESKTOP:
        render_wide_desktop_layout(
            video_component, player_cards, win_probability, 
            chat_component, additional_components
        )
    elif layout_mode == LayoutMode.MEDIUM_TABLET:
        render_medium_tablet_layout(
            video_component, player_cards, win_probability, 
            chat_component, additional_components
        )
    else:  # MOBILE
        render_mobile_layout(
            video_component, player_cards, win_probability, 
            chat_component, additional_components
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add layout mode indicator (for debugging)
    if st.checkbox("Show Layout Debug Info", value=False):
        st.info(f"Current layout mode: {layout_mode.value}")
        st.info(f"Container width: {container_width or 'auto-detected'}")


def create_component_wrapper(component_func: Callable, **kwargs) -> Callable:
    """
    Create a wrapper function for components with pre-configured arguments
    
    Args:
        component_func: The component function to wrap
        **kwargs: Keyword arguments to pass to the component
        
    Returns:
        Wrapped function that can be called without arguments
    """
    def wrapper():
        return component_func(**kwargs)
    return wrapper


def get_layout_recommendations(layout_mode: LayoutMode) -> Dict[str, Any]:
    """
    Get layout recommendations for the given mode
    
    Args:
        layout_mode: The layout mode to get recommendations for
        
    Returns:
        Dictionary of layout recommendations
    """
    recommendations = {
        LayoutMode.WIDE_DESKTOP: {
            "max_columns": 4,
            "card_width": "auto",
            "sidebar_visible": True,
            "video_aspect_ratio": "16:9",
            "chat_height": "400px"
        },
        LayoutMode.MEDIUM_TABLET: {
            "max_columns": 2,
            "card_width": "300px",
            "sidebar_visible": True,
            "video_aspect_ratio": "16:9",
            "chat_height": "300px"
        },
        LayoutMode.MOBILE: {
            "max_columns": 1,
            "card_width": "100%",
            "sidebar_visible": False,
            "video_aspect_ratio": "16:9",
            "chat_height": "250px"
        }
    }
    
    return recommendations.get(layout_mode, recommendations[LayoutMode.MOBILE]) 