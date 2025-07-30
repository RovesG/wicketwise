# Purpose: Interactive GNN embedding visualization using UMAP and Streamlit
# Author: WicketWise Team, Last Modified: 2024-07-19

"""
This Streamlit application provides interactive visualization of GNN player embeddings.
It loads 128D player embeddings, projects them to 2D using UMAP, and provides
interactive exploration with filtering and hover information.

Features:
- Load GNN embeddings from saved files
- UMAP dimensionality reduction to 2D
- Interactive scatter plots with hover information
- Color coding by team, role, or batting average
- Filtering by team, role, and season
- Optional video links for players
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🏏 GNN Embedding Visualizer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .filter-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e6e6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class EmbeddingLoader:
    """Handles loading of GNN embeddings from various file formats."""
    
    @staticmethod
    def load_embeddings(file_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Load GNN embeddings from file.
        
        Args:
            file_path: Path to embedding file
            
        Returns:
            Tuple of (embeddings_dict, metadata_dict)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
        
        try:
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Extract embeddings and metadata
            if isinstance(data, dict):
                if 'embeddings' in data:
                    embeddings = data['embeddings']
                    metadata = {k: v for k, v in data.items() if k != 'embeddings'}
                else:
                    # Assume all values are embeddings
                    embeddings = data
                    metadata = {}
            else:
                raise ValueError("Invalid embedding file format")
            
            # Convert embeddings to numpy arrays if needed
            processed_embeddings = {}
            for player_id, embedding in embeddings.items():
                if isinstance(embedding, list):
                    processed_embeddings[player_id] = np.array(embedding, dtype=np.float32)
                elif isinstance(embedding, np.ndarray):
                    processed_embeddings[player_id] = embedding.astype(np.float32)
                else:
                    logger.warning(f"Skipping invalid embedding for {player_id}")
                    continue
            
            logger.info(f"Loaded {len(processed_embeddings)} player embeddings from {file_path}")
            return processed_embeddings, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings: {str(e)}")
    
    @staticmethod
    def create_sample_embeddings(num_players: int = 50, embedding_dim: int = 128) -> Dict[str, np.ndarray]:
        """Create sample embeddings for testing."""
        np.random.seed(42)
        
        # Sample player names (mix of real and fictional)
        player_names = [
            "kohli", "smith", "root", "williamson", "babar", "warner", "rohit", "stokes",
            "starc", "bumrah", "archer", "cummins", "rabada", "boult", "rashid", "lyon",
            "dhoni", "buttler", "pant", "de_kock", "rizwan", "carey", "foakes", "watling",
            "pandya", "stoinis", "russell", "narine", "jadeja", "maxwell", "shakib", "afridi"
        ]
        
        # Generate additional fictional players if needed
        while len(player_names) < num_players:
            player_names.append(f"player_{len(player_names) + 1}")
        
        embeddings = {}
        for i, player in enumerate(player_names[:num_players]):
            # Create somewhat structured embeddings
            embedding = np.random.normal(0, 0.5, embedding_dim).astype(np.float32)
            
            # Add some structure based on player type
            if any(name in player.lower() for name in ["starc", "bumrah", "archer", "cummins", "rabada"]):
                # Bowling-focused embeddings
                embedding[:32] *= 1.5  # Bowling features
            elif any(name in player.lower() for name in ["kohli", "smith", "root", "babar"]):
                # Batting-focused embeddings
                embedding[32:64] *= 1.5  # Batting features
            elif any(name in player.lower() for name in ["dhoni", "buttler", "pant", "de_kock"]):
                # Wicket-keeping features
                embedding[64:96] *= 1.2  # Keeping features
            
            embeddings[player] = embedding
        
        return embeddings


class PlayerMetadata:
    """Manages player metadata for visualization."""
    
    @staticmethod
    def create_sample_metadata(player_ids: List[str]) -> pd.DataFrame:
        """Create sample player metadata."""
        np.random.seed(42)
        
        # Sample teams
        teams = ["India", "Australia", "England", "New Zealand", "South Africa", 
                "Pakistan", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan"]
        
        # Sample roles
        roles = ["Top Order Batter", "Middle Order Batter", "Finisher", "All-rounder",
                "Fast Bowler", "Spin Bowler", "Wicket Keeper", "Opening Batter"]
        
        # Sample seasons
        seasons = ["2020", "2021", "2022", "2023", "2024"]
        
        metadata_list = []
        for player_id in player_ids:
            # Assign team based on player name patterns
            if any(name in player_id.lower() for name in ["kohli", "rohit", "bumrah", "pant", "jadeja", "dhoni"]):
                team = "India"
            elif any(name in player_id.lower() for name in ["smith", "warner", "starc", "cummins", "maxwell"]):
                team = "Australia"
            elif any(name in player_id.lower() for name in ["root", "stokes", "archer", "buttler"]):
                team = "England"
            elif any(name in player_id.lower() for name in ["williamson", "boult", "watling"]):
                team = "New Zealand"
            elif any(name in player_id.lower() for name in ["rabada", "de_kock"]):
                team = "South Africa"
            elif any(name in player_id.lower() for name in ["babar", "rizwan", "afridi"]):
                team = "Pakistan"
            else:
                team = np.random.choice(teams)
            
            # Assign role based on player name patterns
            if any(name in player_id.lower() for name in ["starc", "bumrah", "archer", "cummins", "rabada", "boult"]):
                role = "Fast Bowler"
            elif any(name in player_id.lower() for name in ["rashid", "lyon", "jadeja", "narine"]):
                role = "Spin Bowler"
            elif any(name in player_id.lower() for name in ["dhoni", "buttler", "pant", "de_kock", "rizwan", "carey"]):
                role = "Wicket Keeper"
            elif any(name in player_id.lower() for name in ["pandya", "stoinis", "russell", "maxwell", "shakib", "stokes"]):
                role = "All-rounder"
            elif any(name in player_id.lower() for name in ["kohli", "smith", "root", "williamson", "babar"]):
                role = "Top Order Batter"
            else:
                role = np.random.choice(roles)
            
            # Generate batting average (higher for batters)
            if "Batter" in role or "Keeper" in role:
                batting_avg = np.random.normal(35, 10)
            elif "All-rounder" in role:
                batting_avg = np.random.normal(25, 8)
            else:
                batting_avg = np.random.normal(15, 5)
            
            batting_avg = max(5.0, min(60.0, batting_avg))  # Clamp to realistic range
            
            metadata_list.append({
                'player_id': player_id,
                'team': team,
                'role': role,
                'batting_average': round(batting_avg, 1),
                'season': np.random.choice(seasons),
                'matches_played': np.random.randint(10, 100),
                'video_link': f"https://example.com/videos/{player_id.replace(' ', '_')}.mp4"
            })
        
        return pd.DataFrame(metadata_list)


class UMAPProjector:
    """Handles UMAP dimensionality reduction."""
    
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
        """Initialize UMAP projector."""
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.reducer = None
        self.fitted = False
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit UMAP and transform embeddings to 2D."""
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=2,
            random_state=self.random_state,
            metric='cosine'
        )
        
        projection = self.reducer.fit_transform(embeddings)
        self.fitted = True
        
        logger.info(f"UMAP projection completed: {embeddings.shape} -> {projection.shape}")
        return projection
    
    def get_params(self) -> Dict[str, Any]:
        """Get UMAP parameters."""
        return {
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'random_state': self.random_state
        }


def create_interactive_plot(df: pd.DataFrame, color_by: str, title: str) -> go.Figure:
    """Create interactive plotly scatter plot."""
    
    # Define color schemes
    color_schemes = {
        'team': px.colors.qualitative.Set3,
        'role': px.colors.qualitative.Pastel,
        'batting_average': px.colors.sequential.Viridis
    }
    
    if color_by in ['team', 'role']:
        # Categorical coloring
        fig = px.scatter(
            df, 
            x='umap_x', 
            y='umap_y',
            color=color_by,
            hover_data=['player_id', 'team', 'role', 'batting_average', 'matches_played'],
            title=title,
            color_discrete_sequence=color_schemes.get(color_by, px.colors.qualitative.Set1)
        )
    else:
        # Continuous coloring (batting_average)
        fig = px.scatter(
            df, 
            x='umap_x', 
            y='umap_y',
            color=color_by,
            hover_data=['player_id', 'team', 'role', 'batting_average', 'matches_played'],
            title=title,
            color_continuous_scale=color_schemes.get(color_by, 'Viridis')
        )
    
    # Customize layout
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
        hovermode='closest',
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        font=dict(size=12)
    )
    
    # Customize markers
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(width=1, color='white')
        )
    )
    
    return fig


def apply_filters(df: pd.DataFrame, team_filter: List[str], role_filter: List[str], season_filter: List[str]) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    filtered_df = df.copy()
    
    if team_filter:
        filtered_df = filtered_df[filtered_df['team'].isin(team_filter)]
    
    if role_filter:
        filtered_df = filtered_df[filtered_df['role'].isin(role_filter)]
    
    if season_filter:
        filtered_df = filtered_df[filtered_df['season'].isin(season_filter)]
    
    return filtered_df


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏏 GNN Embedding Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📊 Controls")
        
        # File upload or sample data
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Upload Embedding File", "Use Sample Data"]
        )
        
        embeddings = None
        metadata = None
        
        if data_source == "Upload Embedding File":
            uploaded_file = st.file_uploader(
                "Choose embedding file",
                type=['pkl', 'json'],
                help="Upload a pickle or JSON file containing player embeddings"
            )
            
            if uploaded_file is not None:
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_embeddings.{uploaded_file.name.split('.')[-1]}"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load embeddings
                    embeddings, file_metadata = EmbeddingLoader.load_embeddings(temp_path)
                    
                    # Clean up temp file
                    Path(temp_path).unlink()
                    
                    st.success(f"✅ Loaded {len(embeddings)} player embeddings")
                    
                except Exception as e:
                    st.error(f"❌ Error loading embeddings: {str(e)}")
                    return
        else:
            # Use sample data
            embeddings = EmbeddingLoader.create_sample_embeddings(num_players=50)
            st.success(f"✅ Generated {len(embeddings)} sample embeddings")
        
        if embeddings is None:
            st.info("👆 Please select a data source to continue")
            return
        
        # Create metadata
        player_ids = list(embeddings.keys())
        metadata = PlayerMetadata.create_sample_metadata(player_ids)
        
        # UMAP parameters
        st.subheader("🎯 UMAP Parameters")
        n_neighbors = st.slider("N Neighbors", 5, 50, 15, help="Controls local vs global structure")
        min_dist = st.slider("Min Distance", 0.01, 1.0, 0.1, help="Controls cluster tightness")
        
        # Color scheme
        st.subheader("🎨 Visualization")
        color_by = st.selectbox(
            "Color by:",
            ["team", "role", "batting_average"],
            help="Choose how to color the points"
        )
        
        # Filters
        st.subheader("🔍 Filters")
        
        available_teams = sorted(metadata['team'].unique())
        team_filter = st.multiselect(
            "Teams:",
            available_teams,
            default=[],
            help="Filter by teams (empty = all teams)"
        )
        
        available_roles = sorted(metadata['role'].unique())
        role_filter = st.multiselect(
            "Roles:",
            available_roles,
            default=[],
            help="Filter by player roles (empty = all roles)"
        )
        
        available_seasons = sorted(metadata['season'].unique())
        season_filter = st.multiselect(
            "Seasons:",
            available_seasons,
            default=[],
            help="Filter by seasons (empty = all seasons)"
        )
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📈 Embedding Projection")
        
        # Prepare data for UMAP
        embedding_matrix = np.vstack([embeddings[player_id] for player_id in player_ids])
        
        # Perform UMAP projection
        with st.spinner("🔄 Computing UMAP projection..."):
            projector = UMAPProjector(n_neighbors=n_neighbors, min_dist=min_dist)
            projection = projector.fit_transform(embedding_matrix)
        
        # Create dataframe with projections and metadata
        df = metadata.copy()
        df['umap_x'] = projection[:, 0]
        df['umap_y'] = projection[:, 1]
        
        # Apply filters
        filtered_df = apply_filters(df, team_filter, role_filter, season_filter)
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No players match the current filters")
            return
        
        # Create and display plot
        title = f"Player Embeddings (Colored by {color_by.title()})"
        fig = create_interactive_plot(filtered_df, color_by, title)
        
        # Display plot with click events
        selected_points = st.plotly_chart(
            fig, 
            use_container_width=True,
            selection_mode=['points'],
            key="embedding_plot"
        )
    
    with col2:
        st.subheader("📊 Statistics")
        
        # Display metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Players", len(df))
        st.metric("Filtered Players", len(filtered_df))
        st.metric("Embedding Dimension", embedding_matrix.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # UMAP parameters display
        st.subheader("🎯 UMAP Settings")
        umap_params = projector.get_params()
        for param, value in umap_params.items():
            st.text(f"{param}: {value}")
        
        # Filter statistics
        if team_filter or role_filter or season_filter:
            st.subheader("🔍 Active Filters")
            if team_filter:
                st.text(f"Teams: {', '.join(team_filter[:3])}{'...' if len(team_filter) > 3 else ''}")
            if role_filter:
                st.text(f"Roles: {', '.join(role_filter[:2])}{'...' if len(role_filter) > 2 else ''}")
            if season_filter:
                st.text(f"Seasons: {', '.join(season_filter)}")
    
    # Player details section
    st.subheader("👥 Player Details")
    
    # Show selected player info or top players
    if st.session_state.get('embedding_plot', {}).get('selection', {}).get('points'):
        # Show selected players
        selected_indices = [p['pointIndex'] for p in st.session_state['embedding_plot']['selection']['points']]
        selected_players = filtered_df.iloc[selected_indices]
        
        st.write(f"**Selected Players ({len(selected_players)}):**")
        
        for _, player in selected_players.iterrows():
            with st.expander(f"🏏 {player['player_id'].title()} ({player['team']})"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"**Team:** {player['team']}")
                    st.write(f"**Role:** {player['role']}")
                    st.write(f"**Batting Average:** {player['batting_average']}")
                
                with col_b:
                    st.write(f"**Season:** {player['season']}")
                    st.write(f"**Matches:** {player['matches_played']}")
                    
                    # Optional video link
                    if 'video_link' in player and player['video_link']:
                        st.markdown(f"[🎥 Watch Video]({player['video_link']})")
    else:
        # Show summary table
        st.write("**Click on points in the plot above to see player details**")
        
        # Display summary statistics
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            st.subheader("By Team")
            team_counts = filtered_df['team'].value_counts()
            st.dataframe(team_counts.head(5), use_container_width=True)
        
        with summary_cols[1]:
            st.subheader("By Role")
            role_counts = filtered_df['role'].value_counts()
            st.dataframe(role_counts, use_container_width=True)
        
        with summary_cols[2]:
            st.subheader("Top Averages")
            top_avg = filtered_df.nlargest(5, 'batting_average')[['player_id', 'batting_average']]
            st.dataframe(top_avg, use_container_width=True)


if __name__ == "__main__":
    main()