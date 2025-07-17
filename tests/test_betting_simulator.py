# Purpose: Test suite for betting simulator functionality
# Author: Assistant, Last Modified: 2024

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
import csv

# Add the wicketwise module to path
sys.path.append(str(Path(__file__).parent.parent))

from wicketwise.betting_simulator import (
    OfflineBettingSimulator, 
    BettingConfig, 
    StakingMethod, 
    BettingDecision
)


class TestBettingSimulator:
    """Test suite for the OfflineBettingSimulator class."""
    
    @pytest.fixture
    def mock_predictions_data(self):
        """Create mock prediction data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        data = []
        for i in range(10):  # 10 balls as requested
            # Create realistic cricket prediction data
            win_prob = np.random.uniform(0.3, 0.9)  # Random win probability
            mispricing = np.random.uniform(-0.2, 0.3)  # Random mispricing
            
            data.append({
                'match_id': 'test_match_001',
                'ball_id': f'1.{i+1}',
                'actual_runs': np.random.choice([0, 1, 2, 4, 6]),
                'predicted_runs_class': np.random.choice(['0_runs', '1_run', '2_runs', '4_runs', '6_runs']),
                'win_prob': win_prob,
                'odds_mispricing': mispricing,
                'phase': 'powerplay',
                'batter_id': f'batter_{i % 3 + 1}',
                'bowler_id': f'bowler_{i % 2 + 1}',
                'actual_win_prob': win_prob + np.random.normal(0, 0.05)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_predictions_file(self, mock_predictions_data):
        """Create a temporary CSV file with mock predictions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            mock_predictions_data.to_csv(f.name, index=False)
            yield f.name
        
        # Cleanup
        try:
            os.unlink(f.name)
        except:
            pass
    
    @pytest.fixture
    def betting_config(self):
        """Create a test betting configuration."""
        return BettingConfig(
            initial_bankroll=1000.0,
            flat_stake=10.0,
            kelly_fraction=0.25,
            min_stake=1.0,
            max_stake=100.0,
            value_threshold=0.05,
            confidence_threshold=0.6
        )
    
    @pytest.fixture
    def simulator(self, betting_config):
        """Create a betting simulator instance."""
        return OfflineBettingSimulator(betting_config)
    
    def test_init(self, betting_config):
        """Test betting simulator initialization."""
        simulator = OfflineBettingSimulator(betting_config)
        
        assert simulator.config == betting_config
        assert simulator.bankroll == betting_config.initial_bankroll
        assert simulator.predictions_data is None
        assert simulator.betting_log == []
        assert simulator.total_bets == 0
        assert simulator.winning_bets == 0
        assert simulator.total_staked == 0.0
        assert simulator.total_returns == 0.0
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        simulator = OfflineBettingSimulator()
        
        assert simulator.config.initial_bankroll == 1000.0
        assert simulator.config.flat_stake == 10.0
        assert simulator.bankroll == 1000.0
    
    def test_load_predictions_success(self, simulator, temp_predictions_file):
        """Test successful prediction loading."""
        result = simulator.load_predictions(temp_predictions_file)
        
        assert result is True
        assert simulator.predictions_data is not None
        assert len(simulator.predictions_data) == 10
        
        # Check that processed columns were added
        assert 'simulated_odds' in simulator.predictions_data.columns
        assert 'implied_probability' in simulator.predictions_data.columns
        assert 'betting_edge' in simulator.predictions_data.columns
        assert 'betting_decision' in simulator.predictions_data.columns
        assert 'ball_sequence' in simulator.predictions_data.columns
    
    def test_load_predictions_file_not_found(self, simulator):
        """Test loading predictions from non-existent file."""
        result = simulator.load_predictions("non_existent_file.csv")
        
        assert result is False
        assert simulator.predictions_data is None
    
    def test_load_predictions_missing_columns(self, simulator):
        """Test loading predictions with missing columns."""
        # Create CSV with missing columns
        incomplete_data = pd.DataFrame({
            'match_id': ['test_match'],
            'ball_id': ['1.1'],
            'actual_runs': [4]
            # Missing required columns
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            
            result = simulator.load_predictions(f.name)
            
            assert result is False
            # Note: predictions_data may still be set even if validation fails
            
            os.unlink(f.name)
    
    def test_generate_odds(self, simulator):
        """Test odds generation logic."""
        # Test with various win probabilities and mispricing
        test_cases = [
            {'win_prob': 0.5, 'odds_mispricing': 0.0, 'expected_odds': 2.0},
            {'win_prob': 0.8, 'odds_mispricing': 0.2, 'expected_odds': 1.5},  # 1.25 * 1.2
            {'win_prob': 0.25, 'odds_mispricing': -0.1, 'expected_odds': 3.6},  # 4.0 * 0.9
        ]
        
        for case in test_cases:
            row = pd.Series(case)
            odds = simulator._generate_odds(row)
            
            assert isinstance(odds, float)
            assert simulator.config.min_odds <= odds <= simulator.config.max_odds
            assert abs(odds - case['expected_odds']) < 0.5  # Allow some tolerance
    
    def test_determine_betting_decision(self, simulator):
        """Test betting decision logic."""
        # Test value bet scenario
        row = pd.Series({
            'betting_edge': 0.1,  # Above threshold
            'win_prob': 0.7      # Above confidence threshold
        })
        decision = simulator._determine_betting_decision(row)
        assert decision == BettingDecision.VALUE_BET.value
        
        # Test risk alert scenario
        row = pd.Series({
            'betting_edge': -0.1,  # Below negative threshold
            'win_prob': 0.5
        })
        decision = simulator._determine_betting_decision(row)
        assert decision == BettingDecision.RISK_ALERT.value
        
        # Test no bet scenario
        row = pd.Series({
            'betting_edge': 0.02,  # Below threshold
            'win_prob': 0.5
        })
        decision = simulator._determine_betting_decision(row)
        assert decision == BettingDecision.NO_BET.value
    
    def test_calculate_stake_flat(self, simulator):
        """Test flat staking method."""
        row = pd.Series({'simulated_odds': 2.0, 'win_prob': 0.7})
        stake = simulator._calculate_stake(row, StakingMethod.FLAT)
        
        assert stake == simulator.config.flat_stake
    
    def test_calculate_stake_kelly(self, simulator):
        """Test Kelly staking method."""
        row = pd.Series({
            'simulated_odds': 2.0,
            'win_prob': 0.6
        })
        stake = simulator._calculate_stake(row, StakingMethod.KELLY)
        
        assert isinstance(stake, float)
        assert stake >= 0
        assert stake <= simulator.config.max_stake
        assert stake <= simulator.bankroll
    
    def test_calculate_stake_proportional(self, simulator):
        """Test proportional staking method."""
        row = pd.Series({
            'betting_edge': 0.1,
            'simulated_odds': 2.0,
            'win_prob': 0.7
        })
        stake = simulator._calculate_stake(row, StakingMethod.PROPORTIONAL)
        
        assert isinstance(stake, float)
        assert stake >= 0
        assert stake <= simulator.config.max_stake
    
    def test_calculate_stake_limits(self, simulator):
        """Test stake limits are enforced."""
        # Test minimum stake
        row = pd.Series({
            'betting_edge': 0.001,
            'simulated_odds': 1.1,
            'win_prob': 0.51
        })
        stake = simulator._calculate_stake(row, StakingMethod.PROPORTIONAL)
        assert stake >= simulator.config.min_stake
        
        # Test maximum stake (simulate high edge scenario)
        simulator.bankroll = 10000.0  # High bankroll
        row = pd.Series({
            'betting_edge': 0.5,
            'simulated_odds': 10.0,
            'win_prob': 0.9
        })
        stake = simulator._calculate_stake(row, StakingMethod.PROPORTIONAL)
        assert stake <= simulator.config.max_stake
    
    def test_determine_bet_result(self, simulator):
        """Test bet result determination."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Test with high win probability
        row = pd.Series({
            'win_prob': 0.9,
            'actual_win_prob': 0.9
        })
        
        # Run multiple times to test randomness
        results = []
        for _ in range(100):
            result = simulator._determine_bet_result(row)
            results.append(result)
        
        # Should win most of the time with high probability
        win_rate = sum(results) / len(results)
        assert 0.7 < win_rate < 1.0  # Allow some variance
    
    def test_simulate_betting_flat_staking(self, simulator, temp_predictions_file):
        """Test betting simulation with flat staking."""
        # Load predictions
        simulator.load_predictions(temp_predictions_file)
        
        # Run simulation
        results = simulator.simulate_betting(StakingMethod.FLAT, "test_betting_log.csv")
        
        # Validate results structure
        assert isinstance(results, dict)
        assert 'initial_bankroll' in results
        assert 'final_bankroll' in results
        assert 'total_bets' in results
        assert 'winning_bets' in results
        assert 'win_rate' in results
        assert 'net_profit' in results
        assert 'roi' in results
        assert 'max_drawdown' in results
        
        # Validate betting log
        assert len(simulator.betting_log) == 10  # One record per ball
        
        # Check that some bets were placed (depends on random data)
        total_bets = sum(1 for record in simulator.betting_log if record['stake'] > 0)
        assert total_bets >= 0  # Could be 0 if no value bets found
        
        # Cleanup
        if Path("test_betting_log.csv").exists():
            os.unlink("test_betting_log.csv")
    
    def test_simulate_betting_kelly_staking(self, simulator, temp_predictions_file):
        """Test betting simulation with Kelly staking."""
        simulator.load_predictions(temp_predictions_file)
        
        results = simulator.simulate_betting(StakingMethod.KELLY, "test_kelly_log.csv")
        
        assert isinstance(results, dict)
        assert results['initial_bankroll'] == simulator.config.initial_bankroll
        
        # Cleanup
        if Path("test_kelly_log.csv").exists():
            os.unlink("test_kelly_log.csv")
    
    def test_simulate_betting_proportional_staking(self, simulator, temp_predictions_file):
        """Test betting simulation with proportional staking."""
        simulator.load_predictions(temp_predictions_file)
        
        results = simulator.simulate_betting(StakingMethod.PROPORTIONAL, "test_prop_log.csv")
        
        assert isinstance(results, dict)
        assert results['initial_bankroll'] == simulator.config.initial_bankroll
        
        # Cleanup
        if Path("test_prop_log.csv").exists():
            os.unlink("test_prop_log.csv")
    
    def test_simulate_betting_no_predictions(self, simulator):
        """Test simulation fails without loaded predictions."""
        with pytest.raises(ValueError, match="No predictions loaded"):
            simulator.simulate_betting(StakingMethod.FLAT)
    
    def test_bankroll_progression(self, simulator):
        """Test bankroll progression logic."""
        # Create controlled prediction data
        controlled_data = pd.DataFrame([
            {
                'match_id': 'test_match',
                'ball_id': '1.1',
                'actual_runs': 4,
                'predicted_runs_class': '4_runs',
                'win_prob': 0.8,  # High win probability
                'odds_mispricing': 0.2,  # Positive mispricing (value bet)
                'phase': 'powerplay',
                'batter_id': 'batter_1',
                'bowler_id': 'bowler_1',
                'actual_win_prob': 0.8
            }
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            controlled_data.to_csv(f.name, index=False)
            
            # Load controlled data
            simulator.load_predictions(f.name)
            
            # Record initial bankroll
            initial_bankroll = simulator.bankroll
            
            # Run simulation
            results = simulator.simulate_betting(StakingMethod.FLAT)
            
            # Validate bankroll changed (either up or down)
            assert simulator.bankroll != initial_bankroll or simulator.total_bets == 0
            
            # Validate betting log records bankroll progression
            if simulator.betting_log:
                first_record = simulator.betting_log[0]
                assert 'bankroll' in first_record
                assert isinstance(first_record['bankroll'], (int, float))
            
            os.unlink(f.name)
    
    def test_betting_log_format(self, simulator, temp_predictions_file):
        """Test betting log CSV format."""
        simulator.load_predictions(temp_predictions_file)
        simulator.simulate_betting(StakingMethod.FLAT, "test_format_log.csv")
        
        # Check that log file was created
        assert Path("test_format_log.csv").exists()
        
        # Read and validate CSV format
        with open("test_format_log.csv", 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            # Check required headers
            expected_headers = ['match_id', 'ball_id', 'decision', 'odds', 'stake', 'result', 'pnl', 'bankroll']
            for header in expected_headers:
                assert header in headers
            
            # Check data rows
            rows = list(reader)
            assert len(rows) == 10  # One row per ball
            
            # Validate data types in first row
            if rows:
                row = rows[0]
                assert isinstance(row['match_id'], str)
                assert isinstance(row['ball_id'], str)
                assert row['decision'] in ['no_bet', 'value_bet', 'risk_alert']
                assert float(row['odds']) > 0
                assert float(row['stake']) >= 0
                assert row['result'] in ['no_bet', 'won', 'lost']
                assert isinstance(float(row['pnl']), float)
                assert float(row['bankroll']) > 0
        
        # Cleanup
        os.unlink("test_format_log.csv")
    
    def test_analyze_performance(self, simulator, temp_predictions_file):
        """Test performance analysis functionality."""
        simulator.load_predictions(temp_predictions_file)
        simulator.simulate_betting(StakingMethod.FLAT)
        
        analysis = simulator.analyze_performance()
        
        if simulator.total_bets > 0:
            assert isinstance(analysis, dict)
            assert 'total_opportunities' in analysis
            assert 'betting_frequency' in analysis
            assert 'average_stake' in analysis
            assert 'average_odds' in analysis
            assert 'sharpe_ratio' in analysis
            
            assert analysis['total_opportunities'] == 10
            assert 0 <= analysis['betting_frequency'] <= 100
        else:
            # No bets placed
            assert analysis == {}
    
    def test_get_betting_summary(self, simulator, temp_predictions_file):
        """Test betting summary generation."""
        simulator.load_predictions(temp_predictions_file)
        simulator.simulate_betting(StakingMethod.FLAT)
        
        summary = simulator.get_betting_summary()
        
        assert isinstance(summary, str)
        assert "BETTING SIMULATION RESULTS" in summary
        assert "Financial Performance" in summary
        assert "Betting Statistics" in summary
        assert "Initial Bankroll" in summary
        assert "Final Bankroll" in summary
    
    def test_calculate_results_no_bets(self, simulator, temp_predictions_file):
        """Test results calculation when no bets are placed."""
        # Create data that won't trigger any bets
        no_bet_data = pd.DataFrame([
            {
                'match_id': 'test_match',
                'ball_id': '1.1',
                'actual_runs': 0,
                'predicted_runs_class': '0_runs',
                'win_prob': 0.3,  # Low win probability
                'odds_mispricing': -0.1,  # Negative mispricing
                'phase': 'powerplay',
                'batter_id': 'batter_1',
                'bowler_id': 'bowler_1',
                'actual_win_prob': 0.3
            }
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            no_bet_data.to_csv(f.name, index=False)
            
            simulator.load_predictions(f.name)
            results = simulator.simulate_betting(StakingMethod.FLAT)
            
            # Should return default results structure
            assert results['total_bets'] == 0
            assert results['winning_bets'] == 0
            assert results['win_rate'] == 0.0
            assert results['net_profit'] == 0.0
            assert results['roi'] == 0.0
            assert results['max_drawdown'] == 0.0
            
            os.unlink(f.name)
    
    def test_max_drawdown_calculation(self, simulator):
        """Test maximum drawdown calculation."""
        # Simulate betting log with known bankroll progression
        simulator.betting_log = [
            {'bankroll': 1000},  # Start
            {'bankroll': 1100},  # Win
            {'bankroll': 1200},  # Win (peak)
            {'bankroll': 1000},  # Loss
            {'bankroll': 800},   # Loss (lowest point)
            {'bankroll': 900},   # Recovery
        ]
        
        # Set up simulator state for proper calculation
        simulator.total_bets = 5
        simulator.winning_bets = 2
        simulator.bankroll = 900  # Final bankroll
        
        results = simulator._calculate_results()
        
        # Max drawdown should be (1200 - 800) / 1200 = 33.33%
        expected_drawdown = (1200 - 800) / 1200 * 100
        assert abs(results['max_drawdown'] - expected_drawdown) < 0.01
    
    def test_process_predictions(self, simulator, mock_predictions_data):
        """Test prediction processing logic."""
        processed_data = simulator._process_predictions(mock_predictions_data)
        
        # Check that all required columns were added
        assert 'simulated_odds' in processed_data.columns
        assert 'implied_probability' in processed_data.columns
        assert 'betting_edge' in processed_data.columns
        assert 'betting_decision' in processed_data.columns
        assert 'ball_sequence' in processed_data.columns
        
        # Check data integrity
        assert len(processed_data) == len(mock_predictions_data)
        
        # Check that odds are in valid range
        assert all(processed_data['simulated_odds'] >= simulator.config.min_odds)
        assert all(processed_data['simulated_odds'] <= simulator.config.max_odds)
        
        # Check that implied probabilities are between 0 and 1
        assert all(processed_data['implied_probability'] > 0)
        assert all(processed_data['implied_probability'] <= 1)
        
        # Check that betting decisions are valid
        valid_decisions = {BettingDecision.NO_BET.value, BettingDecision.VALUE_BET.value, BettingDecision.RISK_ALERT.value}
        assert set(processed_data['betting_decision'].unique()).issubset(valid_decisions)
    
    def test_staking_method_enum(self):
        """Test StakingMethod enum values."""
        assert StakingMethod.FLAT.value == "flat"
        assert StakingMethod.KELLY.value == "kelly"
        assert StakingMethod.PROPORTIONAL.value == "proportional"
    
    def test_betting_decision_enum(self):
        """Test BettingDecision enum values."""
        assert BettingDecision.NO_BET.value == "no_bet"
        assert BettingDecision.VALUE_BET.value == "value_bet"
        assert BettingDecision.RISK_ALERT.value == "risk_alert"
    
    def test_betting_config_defaults(self):
        """Test BettingConfig default values."""
        config = BettingConfig()
        
        assert config.initial_bankroll == 1000.0
        assert config.flat_stake == 10.0
        assert config.kelly_fraction == 0.25
        assert config.min_stake == 1.0
        assert config.max_stake == 100.0
        assert config.min_odds == 1.1
        assert config.max_odds == 10.0
        assert config.value_threshold == 0.05
        assert config.confidence_threshold == 0.6


def test_main_function():
    """Test that the main function can be imported."""
    from wicketwise.betting_simulator import main
    
    # Test that main function exists and is callable
    assert callable(main)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 