# Purpose: Generates post-match analysis reports from prediction logs
# Author: Shamus Rae, Last Modified: 2024-07-30

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import statistics


@dataclass
class MatchSummary:
    """Summary statistics for a completed match."""
    total_balls: int
    prediction_accuracy: float
    win_prob_mae: float  # Mean Absolute Error for win probability
    final_win_prob: float
    actual_result: str  # "win" or "loss"
    betting_decisions: Dict[str, int]  # Count of each decision type
    betting_accuracy: float
    key_insights: List[str]


@dataclass
class BallPredictionLog:
    """Structure for a single ball's prediction log."""
    ball_number: int
    over: int
    ball_in_over: int
    actual_outcome: str
    predicted_outcome_probs: List[float]
    predicted_outcome: str
    win_probability: float
    betting_decision: str
    betting_outcome: Optional[str]  # "win", "loss", or None if no bet
    context: Dict[str, Any]


def parse_prediction_log(log_data: Union[List[Dict], str]) -> List[BallPredictionLog]:
    """Parse prediction log data into structured format."""
    if isinstance(log_data, str):
        with open(log_data, 'r') as f:
            log_data = json.load(f)
    
    parsed_logs = []
    for entry in log_data:
        parsed_logs.append(BallPredictionLog(
            ball_number=entry.get('ball_number', 0),
            over=entry.get('over', 0),
            ball_in_over=entry.get('ball_in_over', 0),
            actual_outcome=entry.get('actual_outcome', ''),
            predicted_outcome_probs=entry.get('predicted_outcome_probs', []),
            predicted_outcome=entry.get('predicted_outcome', ''),
            win_probability=entry.get('win_probability', 0.0),
            betting_decision=entry.get('betting_decision', 'no_bet'),
            betting_outcome=entry.get('betting_outcome'),
            context=entry.get('context', {})
        ))
    
    return parsed_logs


def calculate_prediction_accuracy(logs: List[BallPredictionLog]) -> float:
    """Calculate accuracy of next ball outcome predictions."""
    if not logs:
        return 0.0
    
    correct_predictions = sum(
        1 for log in logs 
        if log.predicted_outcome == log.actual_outcome
    )
    
    return correct_predictions / len(logs)


def calculate_win_prob_performance(logs: List[BallPredictionLog], actual_result: str) -> Dict[str, float]:
    """Calculate win probability prediction performance."""
    if not logs:
        return {'mae': 0.0, 'final_prob': 0.0, 'calibration_error': 0.0}
    
    # Convert actual result to probability (1.0 for win, 0.0 for loss)
    actual_prob = 1.0 if actual_result == "win" else 0.0
    
    # Calculate Mean Absolute Error
    mae = statistics.mean([
        abs(log.win_probability - actual_prob) for log in logs
    ])
    
    final_prob = logs[-1].win_probability if logs else 0.0
    
    # Simple calibration error (difference between final prediction and actual)
    calibration_error = abs(final_prob - actual_prob)
    
    return {
        'mae': mae,
        'final_prob': final_prob,
        'calibration_error': calibration_error
    }


def analyze_betting_performance(logs: List[BallPredictionLog]) -> Dict[str, Any]:
    """Analyze betting decision accuracy and outcomes."""
    betting_stats = {
        'total_bets': 0,
        'value_bets': 0,
        'risk_alerts': 0,
        'no_bets': 0,
        'betting_wins': 0,
        'betting_losses': 0,
        'betting_accuracy': 0.0
    }
    
    for log in logs:
        if log.betting_decision == 'value_bet':
            betting_stats['value_bets'] += 1
            betting_stats['total_bets'] += 1
            if log.betting_outcome == 'win':
                betting_stats['betting_wins'] += 1
            elif log.betting_outcome == 'loss':
                betting_stats['betting_losses'] += 1
        elif log.betting_decision == 'risk_alert':
            betting_stats['risk_alerts'] += 1
        else:
            betting_stats['no_bets'] += 1
    
    # Calculate betting accuracy
    total_decided_bets = betting_stats['betting_wins'] + betting_stats['betting_losses']
    if total_decided_bets > 0:
        betting_stats['betting_accuracy'] = betting_stats['betting_wins'] / total_decided_bets
    
    return betting_stats


def identify_tactical_patterns(logs: List[BallPredictionLog]) -> List[str]:
    """Identify key tactical patterns from prediction logs."""
    insights = []
    
    # Analyze phase-wise performance
    phase_accuracy = defaultdict(list)
    for log in logs:
        phase = log.context.get('phase', 'Unknown')
        is_correct = log.predicted_outcome == log.actual_outcome
        phase_accuracy[phase].append(is_correct)
    
    # Find best/worst performing phases
    phase_stats = {}
    for phase, results in phase_accuracy.items():
        if results:
            phase_stats[phase] = sum(results) / len(results)
    
    if phase_stats:
        best_phase = max(phase_stats, key=phase_stats.get)
        worst_phase = min(phase_stats, key=phase_stats.get)
        
        insights.append(f"Best prediction phase: {best_phase} ({phase_stats[best_phase]:.2%} accuracy)")
        insights.append(f"Worst prediction phase: {worst_phase} ({phase_stats[worst_phase]:.2%} accuracy)")
    
    # Analyze bowler type effectiveness
    bowler_performance = defaultdict(list)
    for log in logs:
        bowler_type = log.context.get('bowler_type', 'Unknown')
        is_correct = log.predicted_outcome == log.actual_outcome
        bowler_performance[bowler_type].append(is_correct)
    
    # Find bowler types with significant prediction differences
    bowler_stats = {}
    for bowler_type, results in bowler_performance.items():
        if len(results) >= 5:  # Only consider types with sufficient data
            bowler_stats[bowler_type] = sum(results) / len(results)
    
    if len(bowler_stats) > 1:
        best_bowler = max(bowler_stats, key=bowler_stats.get)
        worst_bowler = min(bowler_stats, key=bowler_stats.get)
        
        insights.append(f"Most predictable bowler type: {best_bowler} ({bowler_stats[best_bowler]:.2%} accuracy)")
        insights.append(f"Least predictable bowler type: {worst_bowler} ({bowler_stats[worst_bowler]:.2%} accuracy)")
    
    # Analyze outcome distribution
    outcome_counts = Counter(log.actual_outcome for log in logs)
    most_common_outcome = outcome_counts.most_common(1)[0] if outcome_counts else ("N/A", 0)
    insights.append(f"Most common outcome: {most_common_outcome[0]} ({most_common_outcome[1]} times)")
    
    return insights


def generate_post_match_report(
    prediction_logs: Union[List[Dict], str],
    actual_match_result: str,
    output_format: str = "dict"
) -> Union[Dict[str, Any], str]:
    """
    Generate comprehensive post-match analysis report.
    
    Args:
        prediction_logs: List of prediction log dictionaries or path to JSON file
        actual_match_result: "win" or "loss" for the team being analyzed
        output_format: "dict" for structured data or "markdown" for formatted report
    
    Returns:
        Dictionary with analysis results or markdown-formatted string
    """
    # Parse logs
    logs = parse_prediction_log(prediction_logs)
    
    if not logs:
        return {"error": "No prediction logs provided"}
    
    # Calculate core metrics
    prediction_accuracy = calculate_prediction_accuracy(logs)
    win_prob_stats = calculate_win_prob_performance(logs, actual_match_result)
    betting_stats = analyze_betting_performance(logs)
    tactical_insights = identify_tactical_patterns(logs)
    
    # Create summary
    summary = MatchSummary(
        total_balls=len(logs),
        prediction_accuracy=prediction_accuracy,
        win_prob_mae=win_prob_stats['mae'],
        final_win_prob=win_prob_stats['final_prob'],
        actual_result=actual_match_result,
        betting_decisions={
            'value_bet': betting_stats['value_bets'],
            'risk_alert': betting_stats['risk_alerts'],
            'no_bet': betting_stats['no_bets']
        },
        betting_accuracy=betting_stats['betting_accuracy'],
        key_insights=tactical_insights
    )
    
    # Prepare detailed results
    detailed_results = {
        'match_summary': {
            'total_balls': summary.total_balls,
            'actual_result': summary.actual_result,
            'final_win_probability': summary.final_win_prob
        },
        'prediction_performance': {
            'next_ball_accuracy': summary.prediction_accuracy,
            'win_probability_mae': summary.win_prob_mae,
            'calibration_error': win_prob_stats['calibration_error']
        },
        'betting_performance': {
            'total_value_bets': betting_stats['value_bets'],
            'total_risk_alerts': betting_stats['risk_alerts'],
            'betting_accuracy': summary.betting_accuracy,
            'betting_wins': betting_stats['betting_wins'],
            'betting_losses': betting_stats['betting_losses']
        },
        'tactical_insights': tactical_insights
    }
    
    if output_format == "markdown":
        return format_markdown_report(detailed_results)
    else:
        return detailed_results


def format_markdown_report(results: Dict[str, Any]) -> str:
    """Format analysis results as markdown report."""
    match_summary = results['match_summary']
    prediction_perf = results['prediction_performance']
    betting_perf = results['betting_performance']
    insights = results['tactical_insights']
    
    markdown = f"""# Post-Match Analysis Report

## Match Summary
- **Total Balls Analyzed**: {match_summary['total_balls']}
- **Actual Result**: {match_summary['actual_result'].title()}
- **Final Win Probability**: {match_summary['final_win_probability']:.1%}

## Prediction Performance
- **Next Ball Accuracy**: {prediction_perf['next_ball_accuracy']:.1%}
- **Win Probability MAE**: {prediction_perf['win_probability_mae']:.3f}
- **Calibration Error**: {prediction_perf['calibration_error']:.3f}

## Betting Performance
- **Value Bets Placed**: {betting_perf['total_value_bets']}
- **Risk Alerts Generated**: {betting_perf['total_risk_alerts']}
- **Betting Accuracy**: {betting_perf['betting_accuracy']:.1%}
- **Betting Record**: {betting_perf['betting_wins']}W - {betting_perf['betting_losses']}L

## Key Tactical Insights
"""
    
    for insight in insights:
        markdown += f"- {insight}\n"
    
    return markdown 