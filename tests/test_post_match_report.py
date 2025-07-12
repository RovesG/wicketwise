# Purpose: Tests for post-match report generation functionality
# Author: Shamus Rae, Last Modified: 2024-07-30

import pytest
import json
import tempfile
import os
from crickformers.post_match_report import (
    generate_post_match_report,
    parse_prediction_log,
    calculate_prediction_accuracy,
    calculate_win_prob_performance,
    analyze_betting_performance,
    identify_tactical_patterns,
    BallPredictionLog,
    MatchSummary
)


@pytest.fixture
def sample_prediction_logs():
    """Provides sample prediction logs for testing."""
    return [
        {
            "ball_number": 1,
            "over": 1,
            "ball_in_over": 1,
            "actual_outcome": "single",
            "predicted_outcome_probs": [0.1, 0.4, 0.2, 0.1, 0.15, 0.05, 0.0],
            "predicted_outcome": "single",
            "win_probability": 0.55,
            "betting_decision": "no_bet",
            "betting_outcome": None,
            "context": {"phase": "Powerplay", "bowler_type": "fast", "batter_id": "player_1"}
        },
        {
            "ball_number": 2,
            "over": 1,
            "ball_in_over": 2,
            "actual_outcome": "dot",
            "predicted_outcome_probs": [0.3, 0.2, 0.15, 0.1, 0.2, 0.05, 0.0],
            "predicted_outcome": "dot",
            "win_probability": 0.52,
            "betting_decision": "value_bet",
            "betting_outcome": "win",
            "context": {"phase": "Powerplay", "bowler_type": "fast", "batter_id": "player_1"}
        },
        {
            "ball_number": 3,
            "over": 1,
            "ball_in_over": 3,
            "actual_outcome": "four",
            "predicted_outcome_probs": [0.2, 0.1, 0.05, 0.05, 0.3, 0.25, 0.05],
            "predicted_outcome": "four",
            "win_probability": 0.68,
            "betting_decision": "value_bet",
            "betting_outcome": "loss",
            "context": {"phase": "Powerplay", "bowler_type": "fast", "batter_id": "player_1"}
        },
        {
            "ball_number": 4,
            "over": 1,
            "ball_in_over": 4,
            "actual_outcome": "wicket",
            "predicted_outcome_probs": [0.15, 0.15, 0.1, 0.05, 0.2, 0.1, 0.25],
            "predicted_outcome": "single",
            "win_probability": 0.45,
            "betting_decision": "risk_alert",
            "betting_outcome": None,
            "context": {"phase": "Powerplay", "bowler_type": "fast", "batter_id": "player_1"}
        },
        {
            "ball_number": 5,
            "over": 1,
            "ball_in_over": 5,
            "actual_outcome": "single",
            "predicted_outcome_probs": [0.05, 0.35, 0.2, 0.15, 0.2, 0.05, 0.0],
            "predicted_outcome": "single",
            "win_probability": 0.48,
            "betting_decision": "no_bet",
            "betting_outcome": None,
            "context": {"phase": "Middle Overs", "bowler_type": "spin", "batter_id": "player_2"}
        },
        {
            "ball_number": 6,
            "over": 1,
            "ball_in_over": 6,
            "actual_outcome": "dot",
            "predicted_outcome_probs": [0.4, 0.15, 0.1, 0.1, 0.2, 0.05, 0.0],
            "predicted_outcome": "dot",
            "win_probability": 0.46,
            "betting_decision": "value_bet",
            "betting_outcome": "win",
            "context": {"phase": "Middle Overs", "bowler_type": "spin", "batter_id": "player_2"}
        }
    ]


@pytest.fixture
def temp_log_file(sample_prediction_logs):
    """Creates a temporary JSON file with sample logs."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_prediction_logs, f)
        temp_file_path = f.name
    
    yield temp_file_path
    
    # Cleanup
    os.unlink(temp_file_path)


def test_parse_prediction_log_from_list(sample_prediction_logs):
    """Test parsing prediction logs from list format."""
    parsed_logs = parse_prediction_log(sample_prediction_logs)
    
    assert len(parsed_logs) == 6
    assert isinstance(parsed_logs[0], BallPredictionLog)
    assert parsed_logs[0].actual_outcome == "single"
    assert parsed_logs[0].predicted_outcome == "single"
    assert parsed_logs[0].win_probability == 0.55
    assert parsed_logs[0].betting_decision == "no_bet"
    assert parsed_logs[0].context["phase"] == "Powerplay"


def test_parse_prediction_log_from_file(temp_log_file):
    """Test parsing prediction logs from JSON file."""
    parsed_logs = parse_prediction_log(temp_log_file)
    
    assert len(parsed_logs) == 6
    assert isinstance(parsed_logs[0], BallPredictionLog)
    assert parsed_logs[0].actual_outcome == "single"


def test_calculate_prediction_accuracy(sample_prediction_logs):
    """Test calculation of prediction accuracy."""
    logs = parse_prediction_log(sample_prediction_logs)
    accuracy = calculate_prediction_accuracy(logs)
    
    # Expected: 5 correct out of 6 predictions (83.33%)
    expected_accuracy = 5 / 6
    assert abs(accuracy - expected_accuracy) < 0.001


def test_calculate_prediction_accuracy_empty_logs():
    """Test prediction accuracy with empty logs."""
    accuracy = calculate_prediction_accuracy([])
    assert accuracy == 0.0


def test_calculate_win_prob_performance_win(sample_prediction_logs):
    """Test win probability performance calculation for a win."""
    logs = parse_prediction_log(sample_prediction_logs)
    stats = calculate_win_prob_performance(logs, "win")
    
    assert 'mae' in stats
    assert 'final_prob' in stats
    assert 'calibration_error' in stats
    
    # Final probability should be from last log
    assert stats['final_prob'] == 0.46
    
    # MAE should be reasonable (between 0 and 1)
    assert 0 <= stats['mae'] <= 1
    
    # Calibration error should be difference between final prob and actual (1.0 for win)
    expected_calibration_error = abs(0.46 - 1.0)
    assert abs(stats['calibration_error'] - expected_calibration_error) < 0.001


def test_calculate_win_prob_performance_loss(sample_prediction_logs):
    """Test win probability performance calculation for a loss."""
    logs = parse_prediction_log(sample_prediction_logs)
    stats = calculate_win_prob_performance(logs, "loss")
    
    # Calibration error should be difference between final prob and actual (0.0 for loss)
    expected_calibration_error = abs(0.46 - 0.0)
    assert abs(stats['calibration_error'] - expected_calibration_error) < 0.001


def test_analyze_betting_performance(sample_prediction_logs):
    """Test betting performance analysis."""
    logs = parse_prediction_log(sample_prediction_logs)
    betting_stats = analyze_betting_performance(logs)
    
    # Expected counts based on sample data
    assert betting_stats['value_bets'] == 3
    assert betting_stats['risk_alerts'] == 1
    assert betting_stats['no_bets'] == 2
    assert betting_stats['betting_wins'] == 2
    assert betting_stats['betting_losses'] == 1
    
    # Betting accuracy should be 2/3 = 66.67%
    expected_accuracy = 2 / 3
    assert abs(betting_stats['betting_accuracy'] - expected_accuracy) < 0.001


def test_identify_tactical_patterns(sample_prediction_logs):
    """Test tactical pattern identification."""
    logs = parse_prediction_log(sample_prediction_logs)
    insights = identify_tactical_patterns(logs)
    
    assert isinstance(insights, list)
    assert len(insights) > 0
    
    # Check that insights contain expected patterns
    insight_text = ' '.join(insights)
    assert 'phase' in insight_text.lower()
    assert 'outcome' in insight_text.lower()
    
    # Check specific insights based on our sample data
    assert any('Middle Overs' in insight for insight in insights)
    assert any('Powerplay' in insight for insight in insights)
    assert any('single' in insight for insight in insights)


def test_identify_tactical_patterns_with_bowler_insights():
    """Test tactical pattern identification with sufficient bowler data."""
    # Create logs with enough bowler data to trigger bowler insights
    extended_logs = []
    for i in range(12):
        bowler_type = "fast" if i < 6 else "spin"
        outcome = "single" if i % 2 == 0 else "dot"
        predicted = "single" if i % 3 == 0 else "dot"  # Mix correct/incorrect
        
        log_entry = {
            "ball_number": i + 1,
            "over": (i // 6) + 1,
            "ball_in_over": (i % 6) + 1,
            "actual_outcome": outcome,
            "predicted_outcome_probs": [0.5, 0.3, 0.1, 0.05, 0.05, 0.0, 0.0],
            "predicted_outcome": predicted,
            "win_probability": 0.5,
            "betting_decision": "no_bet",
            "betting_outcome": None,
            "context": {"phase": "Middle Overs", "bowler_type": bowler_type}
        }
        extended_logs.append(log_entry)
    
    logs = parse_prediction_log(extended_logs)
    insights = identify_tactical_patterns(logs)
    
    assert isinstance(insights, list)
    assert len(insights) > 0
    
    # Should now include bowler insights
    insight_text = ' '.join(insights)
    assert 'bowler' in insight_text.lower()
    assert 'predictable' in insight_text.lower()


def test_generate_post_match_report_dict_format(sample_prediction_logs):
    """Test generating post-match report in dictionary format."""
    report = generate_post_match_report(sample_prediction_logs, "win", "dict")
    
    assert isinstance(report, dict)
    assert 'match_summary' in report
    assert 'prediction_performance' in report
    assert 'betting_performance' in report
    assert 'tactical_insights' in report
    
    # Validate match summary
    match_summary = report['match_summary']
    assert match_summary['total_balls'] == 6
    assert match_summary['actual_result'] == "win"
    assert match_summary['final_win_probability'] == 0.46
    
    # Validate prediction performance
    pred_perf = report['prediction_performance']
    assert 'next_ball_accuracy' in pred_perf
    assert 'win_probability_mae' in pred_perf
    assert 'calibration_error' in pred_perf
    
    # Validate betting performance
    betting_perf = report['betting_performance']
    assert betting_perf['total_value_bets'] == 3
    assert betting_perf['total_risk_alerts'] == 1
    assert betting_perf['betting_wins'] == 2
    assert betting_perf['betting_losses'] == 1


def test_generate_post_match_report_markdown_format(sample_prediction_logs):
    """Test generating post-match report in markdown format."""
    report = generate_post_match_report(sample_prediction_logs, "loss", "markdown")
    
    assert isinstance(report, str)
    assert "# Post-Match Analysis Report" in report
    assert "## Match Summary" in report
    assert "## Prediction Performance" in report
    assert "## Betting Performance" in report
    assert "## Key Tactical Insights" in report
    
    # Check specific values are present
    assert "Total Balls Analyzed**: 6" in report
    assert "Actual Result**: Loss" in report
    assert "Value Bets Placed**: 3" in report
    assert "2W - 1L" in report  # Betting record format


def test_generate_post_match_report_from_file(temp_log_file):
    """Test generating report from JSON file."""
    report = generate_post_match_report(temp_log_file, "win", "dict")
    
    assert isinstance(report, dict)
    assert report['match_summary']['total_balls'] == 6


def test_generate_post_match_report_empty_logs():
    """Test generating report with empty logs."""
    report = generate_post_match_report([], "win", "dict")
    
    assert isinstance(report, dict)
    assert "error" in report
    assert report["error"] == "No prediction logs provided"


def test_win_loss_metrics_accuracy():
    """Test that win/loss metrics are calculated correctly."""
    # Create specific test data to validate calculations
    test_logs = [
        {
            "ball_number": 1,
            "over": 1,
            "ball_in_over": 1,
            "actual_outcome": "single",
            "predicted_outcome_probs": [0.1, 0.6, 0.1, 0.1, 0.1, 0.0, 0.0],
            "predicted_outcome": "single",
            "win_probability": 0.8,
            "betting_decision": "value_bet",
            "betting_outcome": "win",
            "context": {"phase": "Powerplay", "bowler_type": "fast"}
        },
        {
            "ball_number": 2,
            "over": 1,
            "ball_in_over": 2,
            "actual_outcome": "dot",
            "predicted_outcome_probs": [0.7, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0],
            "predicted_outcome": "dot",
            "win_probability": 0.75,
            "betting_decision": "value_bet",
            "betting_outcome": "loss",
            "context": {"phase": "Powerplay", "bowler_type": "fast"}
        }
    ]
    
    report = generate_post_match_report(test_logs, "win", "dict")
    
    # Validate specific calculations
    assert report['prediction_performance']['next_ball_accuracy'] == 1.0  # 2/2 correct
    assert report['betting_performance']['betting_accuracy'] == 0.5  # 1/2 correct
    assert report['betting_performance']['betting_wins'] == 1
    assert report['betting_performance']['betting_losses'] == 1
    
    # Validate win probability calibration for a win
    final_prob = 0.75
    expected_calibration_error = abs(final_prob - 1.0)  # 1.0 for actual win
    assert abs(report['prediction_performance']['calibration_error'] - expected_calibration_error) < 0.001


def test_different_match_outcomes():
    """Test report generation for different match outcomes."""
    simple_logs = [
        {
            "ball_number": 1,
            "over": 1,
            "ball_in_over": 1,
            "actual_outcome": "single",
            "predicted_outcome_probs": [0.1, 0.6, 0.1, 0.1, 0.1, 0.0, 0.0],
            "predicted_outcome": "single",
            "win_probability": 0.3,
            "betting_decision": "no_bet",
            "betting_outcome": None,
            "context": {"phase": "Death Overs", "bowler_type": "yorker"}
        }
    ]
    
    # Test win scenario
    win_report = generate_post_match_report(simple_logs, "win", "dict")
    assert win_report['match_summary']['actual_result'] == "win"
    
    # Test loss scenario
    loss_report = generate_post_match_report(simple_logs, "loss", "dict")
    assert loss_report['match_summary']['actual_result'] == "loss"
    
    # Calibration errors should be different
    win_calibration = win_report['prediction_performance']['calibration_error']
    loss_calibration = loss_report['prediction_performance']['calibration_error']
    
    # For win_probability = 0.3:
    # Win calibration error = |0.3 - 1.0| = 0.7
    # Loss calibration error = |0.3 - 0.0| = 0.3
    assert abs(win_calibration - 0.7) < 0.001
    assert abs(loss_calibration - 0.3) < 0.001 