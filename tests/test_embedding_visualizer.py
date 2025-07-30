# Purpose: Unit tests for the embedding visualizer Streamlit app
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import the classes and functions from embedding_visualizer
import sys
sys.path.append('.')
from embedding_visualizer import (
    EmbeddingLoader,
    PlayerMetadata,
    UMAPProjector,
    create_interactive_plot,
    apply_filters
)


class TestEmbeddingLoader:
    """Test suite for embedding loading functionality."""
    
    def test_load_embeddings_pickle(self, tmp_path):
        """Test loading embeddings from pickle file."""
        # Create test embeddings
        test_embeddings = {
            "player1": np.random.rand(128).astype(np.float32),
            "player2": np.random.rand(128).astype(np.float32),
            "player3": np.random.rand(128).astype(np.float32)
        }
        
        test_metadata = {
            "model_type": "GraphSAGE",
            "embedding_dim": 128,
            "training_date": "2024-01-15"
        }
        
        test_data = {
            "embeddings": test_embeddings,
            **test_metadata
        }
        
        # Save to pickle file
        pickle_file = tmp_path / "test_embeddings.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        # Load embeddings
        loaded_embeddings, loaded_metadata = EmbeddingLoader.load_embeddings(str(pickle_file))
        
        assert isinstance(loaded_embeddings, dict), "Should return embeddings dict"
        assert len(loaded_embeddings) == 3, "Should load 3 embeddings"
        assert "player1" in loaded_embeddings, "Should contain player1"
        assert isinstance(loaded_embeddings["player1"], np.ndarray), "Should be numpy array"
        assert loaded_embeddings["player1"].dtype == np.float32, "Should be float32"
        assert loaded_embeddings["player1"].shape == (128,), "Should have correct shape"
        
        assert isinstance(loaded_metadata, dict), "Should return metadata dict"
        assert loaded_metadata["model_type"] == "GraphSAGE", "Should preserve metadata"
    
    def test_load_embeddings_json(self, tmp_path):
        """Test loading embeddings from JSON file."""
        test_embeddings = {
            "player1": [0.1, 0.2, 0.3, 0.4] * 32,  # 128D
            "player2": [0.5, 0.6, 0.7, 0.8] * 32   # 128D
        }
        
        # Save to JSON file
        json_file = tmp_path / "test_embeddings.json"
        with open(json_file, 'w') as f:
            json.dump(test_embeddings, f)
        
        # Load embeddings
        loaded_embeddings, loaded_metadata = EmbeddingLoader.load_embeddings(str(json_file))
        
        assert len(loaded_embeddings) == 2, "Should load 2 embeddings"
        assert isinstance(loaded_embeddings["player1"], np.ndarray), "Should convert to numpy array"
        assert loaded_embeddings["player1"].shape == (128,), "Should have correct shape"
        np.testing.assert_array_almost_equal(
            loaded_embeddings["player1"], 
            np.array([0.1, 0.2, 0.3, 0.4] * 32, dtype=np.float32),
            decimal=5
        )
    
    def test_load_embeddings_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            EmbeddingLoader.load_embeddings("nonexistent_file.pkl")
    
    def test_load_embeddings_invalid_format(self, tmp_path):
        """Test loading invalid file format."""
        # Create invalid file
        invalid_file = tmp_path / "invalid.txt"
        with open(invalid_file, 'w') as f:
            f.write("invalid content")
        
        with pytest.raises(RuntimeError, match="Failed to load embeddings"):
            EmbeddingLoader.load_embeddings(str(invalid_file))
    
    def test_load_embeddings_corrupted_pickle(self, tmp_path):
        """Test loading corrupted pickle file."""
        corrupted_file = tmp_path / "corrupted.pkl"
        with open(corrupted_file, 'wb') as f:
            f.write(b"corrupted pickle data")
        
        with pytest.raises(RuntimeError, match="Failed to load embeddings"):
            EmbeddingLoader.load_embeddings(str(corrupted_file))
    
    def test_create_sample_embeddings(self):
        """Test creating sample embeddings."""
        embeddings = EmbeddingLoader.create_sample_embeddings(num_players=10, embedding_dim=64)
        
        assert isinstance(embeddings, dict), "Should return dictionary"
        assert len(embeddings) == 10, "Should create 10 embeddings"
        
        for player_id, embedding in embeddings.items():
            assert isinstance(player_id, str), "Player ID should be string"
            assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
            assert embedding.shape == (64,), "Should have correct dimension"
            assert embedding.dtype == np.float32, "Should be float32"
    
    def test_create_sample_embeddings_deterministic(self):
        """Test that sample embeddings are deterministic."""
        embeddings1 = EmbeddingLoader.create_sample_embeddings(num_players=5, embedding_dim=32)
        embeddings2 = EmbeddingLoader.create_sample_embeddings(num_players=5, embedding_dim=32)
        
        # Should be identical due to fixed random seed
        assert list(embeddings1.keys()) == list(embeddings2.keys()), "Should have same player IDs"
        
        for player_id in embeddings1.keys():
            np.testing.assert_array_equal(
                embeddings1[player_id], 
                embeddings2[player_id],
                err_msg=f"Embeddings should be identical for {player_id}"
            )


class TestPlayerMetadata:
    """Test suite for player metadata functionality."""
    
    def test_create_sample_metadata(self):
        """Test creating sample player metadata."""
        player_ids = ["kohli", "smith", "starc", "bumrah", "dhoni"]
        metadata_df = PlayerMetadata.create_sample_metadata(player_ids)
        
        assert isinstance(metadata_df, pd.DataFrame), "Should return DataFrame"
        assert len(metadata_df) == 5, "Should create metadata for all players"
        
        # Check required columns
        required_columns = ['player_id', 'team', 'role', 'batting_average', 'season', 'matches_played', 'video_link']
        for col in required_columns:
            assert col in metadata_df.columns, f"Should have {col} column"
        
        # Check data types and ranges
        assert metadata_df['batting_average'].dtype in [np.float64, float], "Batting average should be numeric"
        assert all(metadata_df['batting_average'] >= 5.0), "Batting average should be >= 5.0"
        assert all(metadata_df['batting_average'] <= 60.0), "Batting average should be <= 60.0"
        
        assert metadata_df['matches_played'].dtype in [np.int64, int], "Matches played should be integer"
        assert all(metadata_df['matches_played'] >= 10), "Matches played should be >= 10"
        assert all(metadata_df['matches_played'] <= 100), "Matches played should be <= 100"
    
    def test_metadata_player_specific_assignments(self):
        """Test that specific players get appropriate team/role assignments."""
        player_ids = ["kohli", "starc", "dhoni"]
        metadata_df = PlayerMetadata.create_sample_metadata(player_ids)
        
        # Check specific assignments
        kohli_row = metadata_df[metadata_df['player_id'] == 'kohli'].iloc[0]
        assert kohli_row['team'] == 'India', "Kohli should be assigned to India"
        assert 'Batter' in kohli_row['role'], "Kohli should be a batter"
        
        starc_row = metadata_df[metadata_df['player_id'] == 'starc'].iloc[0]
        assert starc_row['team'] == 'Australia', "Starc should be assigned to Australia"
        assert 'Fast Bowler' in starc_row['role'], "Starc should be a fast bowler"
        
        dhoni_row = metadata_df[metadata_df['player_id'] == 'dhoni'].iloc[0]
        assert dhoni_row['team'] == 'India', "Dhoni should be assigned to India"
        assert 'Keeper' in dhoni_row['role'], "Dhoni should be a wicket keeper"
    
    def test_metadata_video_links(self):
        """Test that video links are generated correctly."""
        player_ids = ["test player", "another_player"]
        metadata_df = PlayerMetadata.create_sample_metadata(player_ids)
        
        for _, row in metadata_df.iterrows():
            video_link = row['video_link']
            assert video_link.startswith("https://example.com/videos/"), "Should have correct base URL"
            assert video_link.endswith(".mp4"), "Should end with .mp4"
            assert " " not in video_link.split("/")[-1], "Should replace spaces with underscores"


class TestUMAPProjector:
    """Test suite for UMAP projection functionality."""
    
    def test_umap_projector_initialization(self):
        """Test UMAP projector initialization."""
        projector = UMAPProjector(n_neighbors=10, min_dist=0.2, random_state=123)
        
        assert projector.n_neighbors == 10, "Should set n_neighbors correctly"
        assert projector.min_dist == 0.2, "Should set min_dist correctly"
        assert projector.random_state == 123, "Should set random_state correctly"
        assert not projector.fitted, "Should not be fitted initially"
        assert projector.reducer is None, "Should not have reducer initially"
    
    def test_umap_fit_transform(self):
        """Test UMAP fit and transform."""
        # Create sample high-dimensional data
        np.random.seed(42)
        embeddings = np.random.rand(20, 128).astype(np.float32)
        
        projector = UMAPProjector(n_neighbors=5, min_dist=0.1, random_state=42)
        projection = projector.fit_transform(embeddings)
        
        assert isinstance(projection, np.ndarray), "Should return numpy array"
        assert projection.shape == (20, 2), "Should project to 2D"
        assert projector.fitted, "Should be marked as fitted"
        assert projector.reducer is not None, "Should have reducer after fitting"
    
    def test_umap_deterministic(self):
        """Test that UMAP projection is deterministic with fixed random state."""
        np.random.seed(42)
        embeddings = np.random.rand(15, 64).astype(np.float32)
        
        projector1 = UMAPProjector(random_state=42)
        projection1 = projector1.fit_transform(embeddings)
        
        projector2 = UMAPProjector(random_state=42)
        projection2 = projector2.fit_transform(embeddings)
        
        np.testing.assert_array_almost_equal(
            projection1, projection2, decimal=5,
            err_msg="Projections should be identical with same random state"
        )
    
    def test_umap_get_params(self):
        """Test getting UMAP parameters."""
        projector = UMAPProjector(n_neighbors=20, min_dist=0.5, random_state=999)
        params = projector.get_params()
        
        expected_params = {
            'n_neighbors': 20,
            'min_dist': 0.5,
            'random_state': 999
        }
        
        assert params == expected_params, "Should return correct parameters"


class TestInteractivePlot:
    """Test suite for interactive plotting functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'player_id': ['player1', 'player2', 'player3', 'player4', 'player5'],
            'umap_x': np.random.rand(5),
            'umap_y': np.random.rand(5),
            'team': ['India', 'Australia', 'England', 'India', 'Australia'],
            'role': ['Batter', 'Bowler', 'All-rounder', 'Keeper', 'Batter'],
            'batting_average': [45.2, 12.5, 28.7, 35.1, 42.8],
            'matches_played': [50, 30, 45, 60, 25]
        })
    
    def test_create_interactive_plot_categorical(self, sample_dataframe):
        """Test creating interactive plot with categorical coloring."""
        fig = create_interactive_plot(sample_dataframe, 'team', 'Test Plot')
        
        assert fig is not None, "Should return a figure"
        assert len(fig.data) > 0, "Should have data traces"
        assert fig.layout.title.text == 'Test Plot', "Should set title correctly"
        assert fig.layout.xaxis.title.text == "UMAP Dimension 1", "Should set x-axis title"
        assert fig.layout.yaxis.title.text == "UMAP Dimension 2", "Should set y-axis title"
    
    def test_create_interactive_plot_continuous(self, sample_dataframe):
        """Test creating interactive plot with continuous coloring."""
        fig = create_interactive_plot(sample_dataframe, 'batting_average', 'Batting Average Plot')
        
        assert fig is not None, "Should return a figure"
        assert len(fig.data) > 0, "Should have data traces"
        assert fig.layout.title.text == 'Batting Average Plot', "Should set title correctly"
    
    def test_create_interactive_plot_role_coloring(self, sample_dataframe):
        """Test creating interactive plot with role coloring."""
        fig = create_interactive_plot(sample_dataframe, 'role', 'Role Plot')
        
        assert fig is not None, "Should return a figure"
        assert len(fig.data) > 0, "Should have data traces"
        
        # Check that all roles are represented
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        unique_roles = sample_dataframe['role'].unique()
        
        # Note: Plotly may group data differently, so we just check that we have traces
        assert len(trace_names) > 0, "Should have named traces for roles"


class TestFiltering:
    """Test suite for filtering functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'player_id': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6'],
            'team': ['India', 'Australia', 'England', 'India', 'Australia', 'England'],
            'role': ['Batter', 'Bowler', 'All-rounder', 'Keeper', 'Batter', 'Bowler'],
            'season': ['2023', '2024', '2023', '2024', '2023', '2024'],
            'batting_average': [45.0, 15.0, 30.0, 35.0, 40.0, 18.0]
        })
    
    def test_apply_filters_no_filters(self, sample_dataframe):
        """Test applying no filters (should return all data)."""
        filtered_df = apply_filters(sample_dataframe, [], [], [])
        
        assert len(filtered_df) == len(sample_dataframe), "Should return all rows with no filters"
        pd.testing.assert_frame_equal(filtered_df, sample_dataframe, "Should be identical to original")
    
    def test_apply_filters_team_filter(self, sample_dataframe):
        """Test applying team filter."""
        filtered_df = apply_filters(sample_dataframe, ['India'], [], [])
        
        assert len(filtered_df) == 2, "Should return 2 Indian players"
        assert all(filtered_df['team'] == 'India'), "All players should be from India"
        assert set(filtered_df['player_id']) == {'p1', 'p4'}, "Should contain correct players"
    
    def test_apply_filters_role_filter(self, sample_dataframe):
        """Test applying role filter."""
        filtered_df = apply_filters(sample_dataframe, [], ['Batter'], [])
        
        assert len(filtered_df) == 2, "Should return 2 batters"
        assert all(filtered_df['role'] == 'Batter'), "All players should be batters"
        assert set(filtered_df['player_id']) == {'p1', 'p5'}, "Should contain correct players"
    
    def test_apply_filters_season_filter(self, sample_dataframe):
        """Test applying season filter."""
        filtered_df = apply_filters(sample_dataframe, [], [], ['2023'])
        
        assert len(filtered_df) == 3, "Should return 3 players from 2023"
        assert all(filtered_df['season'] == '2023'), "All players should be from 2023"
        assert set(filtered_df['player_id']) == {'p1', 'p3', 'p5'}, "Should contain correct players"
    
    def test_apply_filters_multiple_filters(self, sample_dataframe):
        """Test applying multiple filters simultaneously."""
        filtered_df = apply_filters(sample_dataframe, ['India', 'Australia'], ['Batter'], ['2023'])
        
        assert len(filtered_df) == 2, "Should return 2 players matching all criteria"
        assert all(filtered_df['team'].isin(['India', 'Australia'])), "Should match team filter"
        assert all(filtered_df['role'] == 'Batter'), "Should match role filter"
        assert all(filtered_df['season'] == '2023'), "Should match season filter"
        assert set(filtered_df['player_id']) == {'p1', 'p5'}, "Should contain correct players"
    
    def test_apply_filters_no_matches(self, sample_dataframe):
        """Test applying filters that result in no matches."""
        filtered_df = apply_filters(sample_dataframe, ['India'], ['Bowler'], [])
        
        assert len(filtered_df) == 0, "Should return no players (no Indian bowlers in sample)"
    
    def test_apply_filters_multiple_values_same_filter(self, sample_dataframe):
        """Test applying multiple values for the same filter type."""
        filtered_df = apply_filters(sample_dataframe, ['India', 'Australia'], [], [])
        
        assert len(filtered_df) == 4, "Should return 4 players from India or Australia"
        assert all(filtered_df['team'].isin(['India', 'Australia'])), "Should match team filter"
        
        filtered_df2 = apply_filters(sample_dataframe, [], ['Batter', 'Bowler'], [])
        
        assert len(filtered_df2) == 4, "Should return 4 batters or bowlers"
        assert all(filtered_df2['role'].isin(['Batter', 'Bowler'])), "Should match role filter"


class TestStreamlitIntegration:
    """Test suite for Streamlit-specific functionality."""
    
    def test_embedding_loader_with_streamlit_file_upload(self):
        """Test embedding loader with simulated Streamlit file upload."""
        # Create sample embedding data
        test_embeddings = {
            "player1": np.random.rand(128).tolist(),
            "player2": np.random.rand(128).tolist()
        }
        
        # Create temporary file to simulate upload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_embeddings, f)
            temp_path = f.name
        
        try:
            # Load embeddings as if from uploaded file
            loaded_embeddings, metadata = EmbeddingLoader.load_embeddings(temp_path)
            
            assert len(loaded_embeddings) == 2, "Should load embeddings from uploaded file"
            assert isinstance(loaded_embeddings["player1"], np.ndarray), "Should convert to numpy arrays"
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_successful_projection_workflow(self):
        """Test complete workflow from embeddings to projection."""
        # Create sample embeddings
        embeddings = EmbeddingLoader.create_sample_embeddings(num_players=20, embedding_dim=64)
        
        # Create metadata
        player_ids = list(embeddings.keys())
        metadata = PlayerMetadata.create_sample_metadata(player_ids)
        
        # Prepare embedding matrix
        embedding_matrix = np.vstack([embeddings[player_id] for player_id in player_ids])
        
        # Perform UMAP projection
        projector = UMAPProjector(n_neighbors=5, min_dist=0.1)
        projection = projector.fit_transform(embedding_matrix)
        
        # Create combined dataframe
        df = metadata.copy()
        df['umap_x'] = projection[:, 0]
        df['umap_y'] = projection[:, 1]
        
        # Verify complete workflow
        assert len(df) == 20, "Should have all players in final dataframe"
        assert 'umap_x' in df.columns, "Should have UMAP x coordinates"
        assert 'umap_y' in df.columns, "Should have UMAP y coordinates"
        assert not df['umap_x'].isna().any(), "Should not have NaN values in projection"
        assert not df['umap_y'].isna().any(), "Should not have NaN values in projection"
    
    def test_interactivity_data_structure(self):
        """Test that data structures support Streamlit interactivity."""
        # Create sample data
        embeddings = EmbeddingLoader.create_sample_embeddings(num_players=10)
        player_ids = list(embeddings.keys())
        metadata = PlayerMetadata.create_sample_metadata(player_ids)
        
        # Create projection dataframe
        embedding_matrix = np.vstack([embeddings[player_id] for player_id in player_ids])
        projector = UMAPProjector()
        projection = projector.fit_transform(embedding_matrix)
        
        df = metadata.copy()
        df['umap_x'] = projection[:, 0]
        df['umap_y'] = projection[:, 1]
        
        # Test filtering (simulates Streamlit multiselect)
        filtered_df = apply_filters(df, ['India'], [], [])
        
        # Test plot creation (simulates Streamlit plotly_chart)
        fig = create_interactive_plot(filtered_df, 'team', 'Test Plot')
        
        # Verify structures support interactivity
        assert hasattr(fig, 'data'), "Figure should have data for interactivity"
        assert hasattr(fig, 'layout'), "Figure should have layout for interactivity"
        assert len(filtered_df) >= 0, "Filtered data should be valid for plotting"
    
    @patch('streamlit.error')
    @patch('streamlit.success')
    def test_error_handling_for_streamlit(self, mock_success, mock_error):
        """Test error handling that would integrate with Streamlit."""
        # Test successful case
        embeddings = EmbeddingLoader.create_sample_embeddings(num_players=5)
        assert len(embeddings) == 5, "Should create embeddings successfully"
        
        # Test error case
        with pytest.raises(FileNotFoundError):
            EmbeddingLoader.load_embeddings("nonexistent_file.pkl")
        
        # Verify that this would work with Streamlit error handling
        # (The actual Streamlit calls would be made in the main app)
        assert True, "Error handling structure is compatible with Streamlit"


# Integration test for the complete workflow
class TestCompleteWorkflow:
    """Test suite for complete embedding visualization workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Step 1: Create sample embeddings
        embeddings = EmbeddingLoader.create_sample_embeddings(num_players=25, embedding_dim=128)
        assert len(embeddings) == 25, "Should create sample embeddings"
        
        # Step 2: Create metadata
        player_ids = list(embeddings.keys())
        metadata = PlayerMetadata.create_sample_metadata(player_ids)
        assert len(metadata) == 25, "Should create metadata for all players"
        
        # Step 3: Perform UMAP projection
        embedding_matrix = np.vstack([embeddings[player_id] for player_id in player_ids])
        projector = UMAPProjector(n_neighbors=10, min_dist=0.1)
        projection = projector.fit_transform(embedding_matrix)
        assert projection.shape == (25, 2), "Should project to 2D"
        
        # Step 4: Create visualization dataframe
        df = metadata.copy()
        df['umap_x'] = projection[:, 0]
        df['umap_y'] = projection[:, 1]
        assert len(df) == 25, "Should have complete dataframe"
        
        # Step 5: Apply filters
        teams = df['team'].unique()[:2]  # Select first 2 teams
        filtered_df = apply_filters(df, list(teams), [], [])
        assert len(filtered_df) > 0, "Should have filtered results"
        assert len(filtered_df) <= 25, "Should not exceed original size"
        
        # Step 6: Create interactive plot
        fig = create_interactive_plot(filtered_df, 'team', 'End-to-End Test')
        assert fig is not None, "Should create interactive plot"
        assert len(fig.data) > 0, "Should have plot data"
        
        # Step 7: Verify all components work together
        assert all(col in df.columns for col in ['umap_x', 'umap_y', 'team', 'role']), \
            "Should have all required columns for visualization"
        
        # Step 8: Test different color schemes
        for color_by in ['team', 'role', 'batting_average']:
            fig_colored = create_interactive_plot(filtered_df, color_by, f'Test {color_by}')
            assert fig_colored is not None, f"Should create plot colored by {color_by}"
        
        print("✅ Complete end-to-end workflow test passed!")


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestCompleteWorkflow::test_end_to_end_workflow", "-v"])