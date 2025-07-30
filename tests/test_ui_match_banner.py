# Purpose: Test suite for match banner component
# Author: Claude, Last Modified: 2025-01-17

import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from ui_style import render_match_banner, CricketColors


class TestMatchBanner:
    """Test suite for render_match_banner function"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_match = {
            'team1_name': 'Mumbai Indians',
            'team1_score': 187,
            'team1_wickets': 5,
            'team2_name': 'Chennai Super Kings',
            'team2_score': 45,
            'team2_wickets': 2,
            'current_over': 7,
            'current_ball': 3,
            'current_innings': 2,
            'match_phase': 'Powerplay',
            'team1_color': '#004BA0',
            'team2_color': '#FFFF3C'
        }
        
        self.minimal_match = {
            'team1_name': 'Team A',
            'team2_name': 'Team B'
        }
    
    @patch('streamlit.markdown')
    def test_render_match_banner_basic(self, mock_markdown):
        """Test basic match banner rendering"""
        render_match_banner(self.sample_match)
        
        # Check that markdown was called twice (CSS and HTML)
        assert mock_markdown.call_count == 2
        
        # Check that both calls use unsafe_allow_html=True
        for call in mock_markdown.call_args_list:
            assert call[1]['unsafe_allow_html'] is True
    
    @patch('streamlit.markdown')
    def test_render_match_banner_content(self, mock_markdown):
        """Test match banner contains expected content"""
        render_match_banner(self.sample_match)
        
        # Get the HTML content (second call)
        html_content = mock_markdown.call_args_list[1][0][0]
        
        # Check that team names are present
        assert 'Mumbai Indians' in html_content
        assert 'Chennai Super Kings' in html_content
        
        # Check that scores are present
        assert '187/5' in html_content
        assert '45/2' in html_content
        
        # Check that over information is present
        assert 'Over 7.3' in html_content
        
        # Check that innings information is present
        assert 'Innings 2' in html_content
        
        # Check that match phase is present
        assert 'Powerplay' in html_content
        
        # Check that vs divider is present
        assert 'vs' in html_content
    
    @patch('streamlit.markdown')
    def test_render_match_banner_emojis(self, mock_markdown):
        """Test match banner contains expected emojis"""
        render_match_banner(self.sample_match)
        
        # Get the HTML content (second call)
        html_content = mock_markdown.call_args_list[1][0][0]
        
        # Check that emojis are present
        assert '🏏' in html_content  # Cricket emoji
        assert '🎯' in html_content  # Target emoji
        assert '⚡' in html_content  # Lightning emoji
    
    @patch('streamlit.markdown')
    def test_render_match_banner_element_order(self, mock_markdown):
        """Test that banner elements appear in expected order"""
        render_match_banner(self.sample_match)
        
        # Get the HTML content (second call)
        html_content = mock_markdown.call_args_list[1][0][0]
        
        # Find positions of key elements
        team1_pos = html_content.find('Mumbai Indians')
        vs_pos = html_content.find('vs')
        team2_pos = html_content.find('Chennai Super Kings')
        over_pos = html_content.find('Over 7.3')
        innings_pos = html_content.find('Innings 2')
        phase_pos = html_content.find('Powerplay')
        
        # Check that team names appear in correct order
        assert team1_pos < vs_pos < team2_pos
        
        # Check that all elements are present
        assert all(pos > 0 for pos in [team1_pos, vs_pos, team2_pos, over_pos, innings_pos, phase_pos])
    
    @patch('streamlit.markdown')
    def test_render_match_banner_css_styling(self, mock_markdown):
        """Test match banner CSS styling"""
        render_match_banner(self.sample_match)
        
        # Get the CSS content (first call)
        css_content = mock_markdown.call_args_list[0][0][0]
        
        # Check that key CSS classes are present
        assert '.match-banner' in css_content
        assert '.match-banner-content' in css_content
        assert '.team-score' in css_content
        assert '.team-name' in css_content
        assert '.score' in css_content
        assert '.match-status' in css_content
        assert '.status-item' in css_content
        assert '.emoji' in css_content
        assert '.vs-divider' in css_content
        
        # Check that responsive design is included
        assert '@media (max-width: 768px)' in css_content
        assert '@media (min-width: 769px)' in css_content
    
    @patch('streamlit.markdown')
    def test_render_match_banner_team_colors(self, mock_markdown):
        """Test match banner uses team colors correctly"""
        render_match_banner(self.sample_match)
        
        # Get the CSS content (first call)
        css_content = mock_markdown.call_args_list[0][0][0]
        
        # Since it's innings 2, should use team2_color
        assert '#FFFF3C' in css_content  # team2_color
        
        # Test first innings
        first_innings_match = self.sample_match.copy()
        first_innings_match['current_innings'] = 1
        
        render_match_banner(first_innings_match)
        
        # Get the CSS content from the new call
        css_content = mock_markdown.call_args_list[2][0][0]
        
        # Should use team1_color for first innings
        assert '#004BA0' in css_content  # team1_color
    
    @patch('streamlit.markdown')
    def test_render_match_banner_fallback_values(self, mock_markdown):
        """Test match banner with minimal data uses fallback values"""
        render_match_banner(self.minimal_match)
        
        # Get the HTML content (second call)
        html_content = mock_markdown.call_args_list[1][0][0]
        
        # Check that fallback values are used
        assert 'Team A' in html_content
        assert 'Team B' in html_content
        assert '0/0' in html_content  # Default scores
        assert 'Over 0.0' in html_content  # Default over
        assert 'Innings 1' in html_content  # Default innings
        assert 'In Progress' in html_content  # Default phase
    
    @patch('streamlit.markdown')
    def test_render_match_banner_empty_dict(self, mock_markdown):
        """Test match banner with empty dictionary"""
        render_match_banner({})
        
        # Get the HTML content (second call)
        html_content = mock_markdown.call_args_list[1][0][0]
        
        # Check that default values are used
        assert 'Team 1' in html_content
        assert 'Team 2' in html_content
        assert '0/0' in html_content
        assert 'Over 0.0' in html_content
        assert 'Innings 1' in html_content
        assert 'In Progress' in html_content
    
    @patch('streamlit.markdown')
    def test_render_match_banner_different_phases(self, mock_markdown):
        """Test match banner with different match phases"""
        phases = ['Powerplay', 'Middle Overs', 'Death Overs', 'Super Over']
        
        for phase in phases:
            test_match = self.sample_match.copy()
            test_match['match_phase'] = phase
            
            render_match_banner(test_match)
            
            # Get the HTML content (last call)
            html_content = mock_markdown.call_args_list[-1][0][0]
            
            # Check that the phase is present
            assert phase in html_content
    
    @patch('streamlit.markdown')
    def test_render_match_banner_different_innings(self, mock_markdown):
        """Test match banner with different innings"""
        for innings in [1, 2]:
            test_match = self.sample_match.copy()
            test_match['current_innings'] = innings
            
            render_match_banner(test_match)
            
            # Get the HTML content (last call)
            html_content = mock_markdown.call_args_list[-1][0][0]
            
            # Check that the innings is present
            assert f'Innings {innings}' in html_content
    
    @patch('streamlit.markdown')
    def test_render_match_banner_high_scores(self, mock_markdown):
        """Test match banner with high scores"""
        high_score_match = {
            'team1_name': 'Team A',
            'team1_score': 250,
            'team1_wickets': 3,
            'team2_name': 'Team B',
            'team2_score': 189,
            'team2_wickets': 7,
            'current_over': 19,
            'current_ball': 6,
            'current_innings': 2,
            'match_phase': 'Death Overs'
        }
        
        render_match_banner(high_score_match)
        
        # Get the HTML content (second call)
        html_content = mock_markdown.call_args_list[1][0][0]
        
        # Check that high scores are displayed correctly
        assert '250/3' in html_content
        assert '189/7' in html_content
        assert 'Over 19.6' in html_content
        assert 'Death Overs' in html_content
    
    @patch('streamlit.markdown')
    def test_render_match_banner_all_out(self, mock_markdown):
        """Test match banner when team is all out"""
        all_out_match = {
            'team1_name': 'Team A',
            'team1_score': 156,
            'team1_wickets': 10,
            'team2_name': 'Team B',
            'team2_score': 78,
            'team2_wickets': 4,
            'current_over': 15,
            'current_ball': 2,
            'current_innings': 2,
            'match_phase': 'Middle Overs'
        }
        
        render_match_banner(all_out_match)
        
        # Get the HTML content (second call)
        html_content = mock_markdown.call_args_list[1][0][0]
        
        # Check that all out is displayed correctly
        assert '156/10' in html_content
        assert '78/4' in html_content
    
    @patch('streamlit.markdown')
    def test_render_match_banner_responsive_css(self, mock_markdown):
        """Test match banner includes responsive CSS"""
        render_match_banner(self.sample_match)
        
        # Get the CSS content (first call)
        css_content = mock_markdown.call_args_list[0][0][0]
        
        # Check mobile responsive styles
        assert 'flex-direction: column' in css_content
        assert 'justify-content: center' in css_content
        assert 'font-size: 0.8em' in css_content
        
        # Check desktop responsive styles
        assert 'flex-direction: row' in css_content
        assert 'justify-content: flex-end' in css_content
        
        # Check that flexbox is used
        assert 'display: flex' in css_content
        assert 'flex-wrap: wrap' in css_content


class TestMatchBannerIntegration:
    """Integration tests for match banner component"""
    
    @patch('streamlit.markdown')
    def test_match_banner_with_real_team_data(self, mock_markdown):
        """Test match banner with realistic team data"""
        ipl_match = {
            'team1_name': 'Royal Challengers Bangalore',
            'team1_score': 173,
            'team1_wickets': 6,
            'team2_name': 'Mumbai Indians',
            'team2_score': 131,
            'team2_wickets': 4,
            'current_over': 16,
            'current_ball': 4,
            'current_innings': 2,
            'match_phase': 'Death Overs',
            'team1_color': '#d41e3a',
            'team2_color': '#004BA0'
        }
        
        render_match_banner(ipl_match)
        
        # Verify function executed without errors
        assert mock_markdown.call_count == 2
        
        # Get the HTML content
        html_content = mock_markdown.call_args_list[1][0][0]
        
        # Check that realistic team names are handled
        assert 'Royal Challengers Bangalore' in html_content
        assert 'Mumbai Indians' in html_content
        assert '173/6' in html_content
        assert '131/4' in html_content
        assert 'Over 16.4' in html_content
        assert 'Death Overs' in html_content
    
    @patch('streamlit.markdown')
    def test_match_banner_edge_cases(self, mock_markdown):
        """Test match banner with edge cases"""
        edge_cases = [
            # No wickets fallen
            {'team1_score': 45, 'team1_wickets': 0, 'team2_score': 0, 'team2_wickets': 0},
            # Very high over count
            {'current_over': 50, 'current_ball': 6},
            # Special characters in team names
            {'team1_name': 'Team A & B', 'team2_name': 'Team C/D'},
            # Large score difference
            {'team1_score': 300, 'team2_score': 50}
        ]
        
        for case in edge_cases:
            test_match = {'team1_name': 'Test A', 'team2_name': 'Test B'}
            test_match.update(case)
            
            # Should not raise any exceptions
            render_match_banner(test_match)
            
        # Verify all cases executed
        assert mock_markdown.call_count == len(edge_cases) * 2


if __name__ == "__main__":
    pytest.main([__file__]) 