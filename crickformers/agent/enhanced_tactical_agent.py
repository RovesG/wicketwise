# Purpose: Enhanced tactical agent using OpenAI API for cricket analysis
# Author: WicketWise Team, Last Modified: 2024-12-07

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from openai import OpenAI
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TacticalInsight:
    """Structured tactical insight from AI analysis."""
    summary: str
    batting_advice: str
    bowling_advice: str
    field_placement: str
    risk_assessment: str
    confidence: float
    key_factors: List[str]
    recommended_actions: List[str]


@dataclass
class MatchContext:
    """Match context for tactical analysis."""
    match_id: str
    current_over: int
    current_ball: int
    innings: int
    batting_team: str
    bowling_team: str
    current_score: int
    wickets_lost: int
    target_score: Optional[int] = None
    required_run_rate: Optional[float] = None
    venue: Optional[str] = None
    competition: Optional[str] = None


class EnhancedTacticalAgent:
    """
    Enhanced tactical agent that uses OpenAI API for sophisticated
    cricket tactical analysis and recommendations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the tactical agent.
        
        Args:
            api_key: OpenAI API key (if not provided, will look for env var)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not found. Agent will run in mock mode.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        # Cricket domain knowledge
        self.cricket_knowledge = {
            'phases': {
                'powerplay': 'First 6 overs with fielding restrictions',
                'middle_overs': 'Overs 7-15, building phase',
                'death_overs': 'Overs 16-20, aggressive phase'
            },
            'bowling_types': {
                'fast': 'Pace bowlers (>130 kmph)',
                'medium': 'Medium pace (120-130 kmph)',
                'spin': 'Spin bowlers (<120 kmph)'
            },
            'field_positions': [
                'slip', 'gully', 'point', 'cover', 'mid-off', 'mid-on', 
                'midwicket', 'square leg', 'fine leg', 'third man',
                'long-on', 'long-off', 'deep midwicket', 'deep square leg'
            ]
        }
    
    def analyze_match_situation(self, 
                               match_context: MatchContext,
                               predictions: Dict[str, Any],
                               player_stats: Dict[str, Any],
                               recent_balls: List[Dict[str, Any]]) -> TacticalInsight:
        """
        Analyze the current match situation and provide tactical insights.
        
        Args:
            match_context: Current match context
            predictions: Model predictions (win prob, next ball outcome, etc.)
            player_stats: Current player statistics
            recent_balls: Recent ball-by-ball data
            
        Returns:
            Tactical insight with recommendations
        """
        logger.info(f"Analyzing tactical situation for {match_context.match_id}")
        
        if self.client is None:
            return self._generate_mock_insight(match_context, predictions)
        
        try:
            # Prepare context for AI analysis
            analysis_context = self._prepare_analysis_context(
                match_context, predictions, player_stats, recent_balls
            )
            
            # Generate tactical insight using OpenAI
            insight = self._generate_ai_insight(analysis_context)
            
            return insight
            
        except Exception as e:
            logger.error(f"Error in tactical analysis: {e}")
            return self._generate_mock_insight(match_context, predictions)
    
    def _prepare_analysis_context(self,
                                 match_context: MatchContext,
                                 predictions: Dict[str, Any],
                                 player_stats: Dict[str, Any],
                                 recent_balls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare comprehensive context for AI analysis."""
        
        # Determine match phase
        phase = 'powerplay' if match_context.current_over < 6 else \
                'middle_overs' if match_context.current_over < 16 else 'death_overs'
        
        # Calculate recent performance
        recent_runs = sum(ball.get('runs_scored', 0) for ball in recent_balls[-6:])
        recent_wickets = sum(1 for ball in recent_balls[-6:] if ball.get('is_wicket', False))
        recent_boundaries = sum(1 for ball in recent_balls[-6:] if ball.get('runs_scored', 0) >= 4)
        
        # Match situation assessment
        if match_context.innings == 1:
            situation = 'batting_first'
            pressure = 'building_total'
        else:
            runs_needed = match_context.target_score - match_context.current_score
            balls_remaining = (20 - match_context.current_over) * 6 - match_context.current_ball
            required_rr = (runs_needed / balls_remaining * 6) if balls_remaining > 0 else 0
            
            if required_rr < 6:
                pressure = 'comfortable_chase'
            elif required_rr < 10:
                pressure = 'moderate_pressure'
            else:
                pressure = 'high_pressure'
            
            situation = 'chasing'
        
        return {
            'match_context': {
                'phase': phase,
                'situation': situation,
                'pressure': pressure,
                'current_over': match_context.current_over,
                'current_ball': match_context.current_ball,
                'innings': match_context.innings,
                'score': f"{match_context.current_score}/{match_context.wickets_lost}",
                'target': match_context.target_score,
                'required_run_rate': match_context.required_run_rate,
                'venue': match_context.venue,
                'competition': match_context.competition
            },
            'predictions': predictions,
            'player_stats': player_stats,
            'recent_performance': {
                'runs_last_6_balls': recent_runs,
                'wickets_last_6_balls': recent_wickets,
                'boundaries_last_6_balls': recent_boundaries,
                'run_rate_last_6_balls': recent_runs
            },
            'cricket_knowledge': self.cricket_knowledge
        }
    
    def _generate_ai_insight(self, context: Dict[str, Any]) -> TacticalInsight:
        """Generate tactical insight using OpenAI API."""
        
        # Create system prompt
        system_prompt = """You are an expert cricket tactical analyst with deep knowledge of T20 cricket strategy. 
        Analyze the current match situation and provide specific, actionable tactical recommendations.
        
        Consider:
        - Match phase and situation
        - Player strengths and weaknesses
        - Recent performance trends
        - Win probability and risk assessment
        - Field placement strategies
        - Bowling changes and tactics
        - Batting approach recommendations
        
        Provide structured, practical advice that teams can implement immediately."""
        
        # Create user prompt with context
        user_prompt = f"""
        Analyze this T20 cricket match situation:
        
        **Match Context:**
        - Phase: {context['match_context']['phase']}
        - Situation: {context['match_context']['situation']}
        - Pressure: {context['match_context']['pressure']}
        - Current Over: {context['match_context']['current_over']}.{context['match_context']['current_ball']}
        - Score: {context['match_context']['score']}
        - Target: {context['match_context']['target']}
        - Required RR: {context['match_context']['required_run_rate']}
        
        **Model Predictions:**
        - Win Probability: {context['predictions'].get('win_probability', 'N/A')}
        - Next Ball Outcome Probabilities: {context['predictions'].get('next_ball_probabilities', {})}
        - Confidence: {context['predictions'].get('confidence', 'N/A')}
        
        **Recent Performance (Last 6 balls):**
        - Runs: {context['recent_performance']['runs_last_6_balls']}
        - Wickets: {context['recent_performance']['wickets_last_6_balls']}
        - Boundaries: {context['recent_performance']['boundaries_last_6_balls']}
        
        **Player Stats:**
        {json.dumps(context['player_stats'], indent=2)}
        
        Provide tactical recommendations in the following format:
        
        **SUMMARY:** [Brief situation summary]
        
        **BATTING ADVICE:** [Specific batting recommendations]
        
        **BOWLING ADVICE:** [Specific bowling recommendations]
        
        **FIELD PLACEMENT:** [Recommended field changes]
        
        **RISK ASSESSMENT:** [Risk analysis and mitigation]
        
        **KEY FACTORS:** [3-5 key factors influencing the situation]
        
        **RECOMMENDED ACTIONS:** [3-5 immediate actionable recommendations]
        
        **CONFIDENCE:** [Your confidence in these recommendations: 0.0-1.0]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse the AI response
            return self._parse_ai_response(ai_response)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _parse_ai_response(self, ai_response: str) -> TacticalInsight:
        """Parse the AI response into structured tactical insight."""
        
        # Initialize default values
        summary = ""
        batting_advice = ""
        bowling_advice = ""
        field_placement = ""
        risk_assessment = ""
        confidence = 0.7
        key_factors = []
        recommended_actions = []
        
        # Parse sections from AI response
        lines = ai_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('**SUMMARY:**'):
                current_section = 'summary'
                summary = line.replace('**SUMMARY:**', '').strip()
            elif line.startswith('**BATTING ADVICE:**'):
                current_section = 'batting_advice'
                batting_advice = line.replace('**BATTING ADVICE:**', '').strip()
            elif line.startswith('**BOWLING ADVICE:**'):
                current_section = 'bowling_advice'
                bowling_advice = line.replace('**BOWLING ADVICE:**', '').strip()
            elif line.startswith('**FIELD PLACEMENT:**'):
                current_section = 'field_placement'
                field_placement = line.replace('**FIELD PLACEMENT:**', '').strip()
            elif line.startswith('**RISK ASSESSMENT:**'):
                current_section = 'risk_assessment'
                risk_assessment = line.replace('**RISK ASSESSMENT:**', '').strip()
            elif line.startswith('**KEY FACTORS:**'):
                current_section = 'key_factors'
                continue
            elif line.startswith('**RECOMMENDED ACTIONS:**'):
                current_section = 'recommended_actions'
                continue
            elif line.startswith('**CONFIDENCE:**'):
                conf_text = line.replace('**CONFIDENCE:**', '').strip()
                try:
                    confidence = float(conf_text)
                except:
                    confidence = 0.7
            elif line and current_section:
                # Continue parsing current section
                if current_section == 'summary' and not summary:
                    summary = line
                elif current_section == 'batting_advice' and not batting_advice:
                    batting_advice = line
                elif current_section == 'bowling_advice' and not bowling_advice:
                    bowling_advice = line
                elif current_section == 'field_placement' and not field_placement:
                    field_placement = line
                elif current_section == 'risk_assessment' and not risk_assessment:
                    risk_assessment = line
                elif current_section == 'key_factors' and line.startswith('-'):
                    key_factors.append(line[1:].strip())
                elif current_section == 'recommended_actions' and line.startswith('-'):
                    recommended_actions.append(line[1:].strip())
        
        return TacticalInsight(
            summary=summary or "Tactical analysis completed",
            batting_advice=batting_advice or "Continue current approach",
            bowling_advice=bowling_advice or "Maintain current strategy",
            field_placement=field_placement or "No field changes recommended",
            risk_assessment=risk_assessment or "Moderate risk situation",
            confidence=confidence,
            key_factors=key_factors or ["Match situation", "Player form", "Venue conditions"],
            recommended_actions=recommended_actions or ["Monitor situation", "Adapt as needed"]
        )
    
    def _generate_mock_insight(self, 
                              match_context: MatchContext,
                              predictions: Dict[str, Any]) -> TacticalInsight:
        """Generate mock tactical insight when AI is not available."""
        
        phase = 'powerplay' if match_context.current_over < 6 else \
                'middle_overs' if match_context.current_over < 16 else 'death_overs'
        
        win_prob = predictions.get('win_probability', 0.5)
        
        if phase == 'powerplay':
            if win_prob > 0.6:
                summary = "Strong position in powerplay - capitalize on field restrictions"
                batting_advice = "Continue aggressive approach, target boundaries"
                bowling_advice = "Attack stumps, use variations sparingly"
            else:
                summary = "Challenging powerplay situation - need to build pressure"
                batting_advice = "Build partnerships, rotate strike"
                bowling_advice = "Tight lines, create pressure"
        
        elif phase == 'middle_overs':
            if win_prob > 0.6:
                summary = "Comfortable position in middle overs - maintain momentum"
                batting_advice = "Rotate strike, target loose deliveries"
                bowling_advice = "Contain runs, create pressure"
            else:
                summary = "Pressure building in middle overs - need acceleration"
                batting_advice = "Take calculated risks, find boundaries"
                bowling_advice = "Attack with variations, dry up runs"
        
        else:  # death_overs
            if win_prob > 0.6:
                summary = "Strong position in death overs - finish strongly"
                batting_advice = "Maximum aggression, target boundaries"
                bowling_advice = "Yorkers and slower balls, defend boundaries"
            else:
                summary = "High pressure death overs - need big shots"
                batting_advice = "All-out attack, take risks"
                bowling_advice = "Execute yorkers, pressure bowling"
        
        return TacticalInsight(
            summary=summary,
            batting_advice=batting_advice,
            bowling_advice=bowling_advice,
            field_placement="Adjust field based on batter tendencies",
            risk_assessment=f"{'Low' if win_prob > 0.7 else 'High' if win_prob < 0.3 else 'Medium'} risk situation",
            confidence=0.7,
            key_factors=[f"Match phase: {phase}", f"Win probability: {win_prob:.2f}", "Player form"],
            recommended_actions=["Monitor situation closely", "Adapt strategy as needed"]
        )
    
    def generate_match_report(self, 
                             match_data: pd.DataFrame,
                             predictions_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive post-match tactical report.
        
        Args:
            match_data: Complete match ball-by-ball data
            predictions_history: History of predictions throughout the match
            
        Returns:
            Comprehensive match report
        """
        logger.info("Generating post-match tactical report")
        
        # Calculate match statistics
        match_stats = self._calculate_match_statistics(match_data)
        
        # Identify key moments
        key_moments = self._identify_key_moments(match_data, predictions_history)
        
        # Generate tactical summary
        tactical_summary = self._generate_tactical_summary(match_stats, key_moments)
        
        return {
            'match_id': match_data['match_id'].iloc[0],
            'match_statistics': match_stats,
            'key_moments': key_moments,
            'tactical_summary': tactical_summary,
            'performance_analysis': self._analyze_performance(match_data),
            'recommendations': self._generate_recommendations(match_stats, key_moments)
        }
    
    def _calculate_match_statistics(self, match_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive match statistics."""
        
        # Overall statistics
        total_runs = match_data['runs_scored'].sum()
        total_wickets = match_data['is_wicket'].sum()
        total_balls = len(match_data)
        
        # Innings breakdown
        innings_stats = {}
        for innings in match_data['innings'].unique():
            innings_data = match_data[match_data['innings'] == innings]
            innings_stats[f'innings_{innings}'] = {
                'runs': innings_data['runs_scored'].sum(),
                'wickets': innings_data['is_wicket'].sum(),
                'balls': len(innings_data),
                'run_rate': (innings_data['runs_scored'].sum() / len(innings_data)) * 6
            }
        
        # Phase analysis
        phase_stats = {}
        for _, row in match_data.iterrows():
            over = row['over']
            phase = 'powerplay' if over < 6 else 'middle_overs' if over < 16 else 'death_overs'
            
            if phase not in phase_stats:
                phase_stats[phase] = {'runs': 0, 'wickets': 0, 'balls': 0}
            
            phase_stats[phase]['runs'] += row['runs_scored']
            phase_stats[phase]['wickets'] += 1 if row['is_wicket'] else 0
            phase_stats[phase]['balls'] += 1
        
        return {
            'total_runs': total_runs,
            'total_wickets': total_wickets,
            'total_balls': total_balls,
            'overall_run_rate': (total_runs / total_balls) * 6,
            'innings_breakdown': innings_stats,
            'phase_analysis': phase_stats
        }
    
    def _identify_key_moments(self, 
                             match_data: pd.DataFrame,
                             predictions_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key moments in the match."""
        
        key_moments = []
        
        # Find significant win probability changes
        if len(predictions_history) > 1:
            for i in range(1, len(predictions_history)):
                current_wp = predictions_history[i].get('win_probability', 0.5)
                previous_wp = predictions_history[i-1].get('win_probability', 0.5)
                
                change = abs(current_wp - previous_wp)
                
                if change > 0.15:  # Significant change
                    key_moments.append({
                        'ball_number': i,
                        'type': 'win_probability_swing',
                        'change': change,
                        'new_probability': current_wp,
                        'description': f"Win probability changed by {change:.2f}"
                    })
        
        # Find boundaries and wickets
        for idx, row in match_data.iterrows():
            if row['runs_scored'] >= 4:
                key_moments.append({
                    'ball_number': idx,
                    'type': 'boundary',
                    'runs': row['runs_scored'],
                    'over': row['over'],
                    'ball': row['ball'],
                    'description': f"{row['runs_scored']} runs scored"
                })
            
            if row['is_wicket']:
                key_moments.append({
                    'ball_number': idx,
                    'type': 'wicket',
                    'over': row['over'],
                    'ball': row['ball'],
                    'description': f"Wicket taken"
                })
        
        # Sort by impact and return top moments
        key_moments.sort(key=lambda x: x.get('change', 0.1), reverse=True)
        return key_moments[:10]
    
    def _generate_tactical_summary(self, 
                                  match_stats: Dict[str, Any],
                                  key_moments: List[Dict[str, Any]]) -> str:
        """Generate tactical summary of the match."""
        
        summary_parts = []
        
        # Overall match summary
        total_runs = match_stats['total_runs']
        total_wickets = match_stats['total_wickets']
        run_rate = match_stats['overall_run_rate']
        
        summary_parts.append(f"Match produced {total_runs} runs with {total_wickets} wickets at {run_rate:.2f} run rate.")
        
        # Phase analysis
        phase_analysis = match_stats['phase_analysis']
        for phase, stats in phase_analysis.items():
            phase_rr = (stats['runs'] / stats['balls']) * 6
            summary_parts.append(f"{phase.replace('_', ' ').title()}: {stats['runs']} runs, {stats['wickets']} wickets (RR: {phase_rr:.2f})")
        
        # Key moments
        if key_moments:
            summary_parts.append(f"Key moments included {len([m for m in key_moments if m['type'] == 'boundary'])} boundaries and {len([m for m in key_moments if m['type'] == 'wicket'])} wickets.")
        
        return " ".join(summary_parts)
    
    def _analyze_performance(self, match_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze team and player performance."""
        
        # Player performance
        batter_performance = match_data.groupby('batter').agg({
            'runs_scored': 'sum',
            'is_wicket': 'sum'
        }).reset_index()
        
        bowler_performance = match_data.groupby('bowler').agg({
            'runs_scored': 'sum',
            'is_wicket': 'sum'
        }).reset_index()
        
        return {
            'top_batters': batter_performance.nlargest(5, 'runs_scored').to_dict('records'),
            'top_bowlers': bowler_performance.nlargest(5, 'is_wicket').to_dict('records')
        }
    
    def _generate_recommendations(self, 
                                 match_stats: Dict[str, Any],
                                 key_moments: List[Dict[str, Any]]) -> List[str]:
        """Generate tactical recommendations based on match analysis."""
        
        recommendations = []
        
        # Analyze run rate by phase
        phase_analysis = match_stats['phase_analysis']
        
        for phase, stats in phase_analysis.items():
            phase_rr = (stats['runs'] / stats['balls']) * 6
            
            if phase == 'powerplay' and phase_rr < 7:
                recommendations.append("Consider more aggressive approach in powerplay to capitalize on field restrictions")
            elif phase == 'middle_overs' and phase_rr < 6:
                recommendations.append("Focus on rotating strike and building partnerships in middle overs")
            elif phase == 'death_overs' and phase_rr < 9:
                recommendations.append("Need more aggressive batting in death overs to maximize total")
        
        # Wicket analysis
        total_wickets = match_stats['total_wickets']
        if total_wickets < 10:
            recommendations.append("Batting team could have been more aggressive with wickets in hand")
        elif total_wickets > 15:
            recommendations.append("Too many wickets lost - focus on building partnerships")
        
        return recommendations 