# Purpose: Streamlit UI for Match Simulator with counterfactual analysis
# Author: Shamus Rae, Last Modified: 2024-01-15

"""
This module provides a Streamlit-based user interface for the Match Simulator,
allowing users to explore counterfactual scenarios and compare outcomes
side-by-side.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import json

from crickformers.match_simulator import (
    MatchSimulator,
    CounterfactualEvent,
    CounterfactualEventType,
    SimulationContext,
    SimulationComparison,
    create_simulation_context_from_match_state,
    CounterfactualEventGenerator
)


class MatchSimulatorUI:
    """Streamlit UI for Match Simulator."""
    
    def __init__(self):
        """Initialize the simulator UI."""
        self.simulator = MatchSimulator()
        self.event_generator = CounterfactualEventGenerator()
    
    def render(self):
        """Render the complete match simulator interface."""
        st.title("🏏 Match Simulator - Counterfactual Analysis")
        st.markdown("Explore 'what-if' scenarios and see how different outcomes affect match predictions")
        
        # Sidebar for controls
        with st.sidebar:
            st.header("🎮 Simulation Controls")
            self._render_sidebar_controls()
        
        # Main content area
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_scenario_selection()
        
        with col2:
            self._render_simulation_results()
    
    def _render_sidebar_controls(self):
        """Render sidebar controls."""
        # Match data upload
        st.subheader("📊 Match Data")
        
        uploaded_file = st.file_uploader(
            "Upload match data (CSV)",
            type=['csv'],
            help="Upload ball-by-ball match data for simulation"
        )
        
        if uploaded_file is not None:
            try:
                match_data = pd.read_csv(uploaded_file)
                st.session_state['match_data'] = match_data
                st.success(f"Loaded {len(match_data)} balls of data")
                
                # Show basic match info
                if len(match_data) > 0:
                    st.write("**Match Summary:**")
                    st.write(f"- Matches: {match_data['match_id'].nunique()}")
                    st.write(f"- Innings: {len(match_data.groupby(['match_id', 'innings']))}")
                    st.write(f"- Total balls: {len(match_data)}")
            
            except Exception as e:
                st.error(f"Error loading match data: {e}")
        
        # Use sample data if no upload
        if 'match_data' not in st.session_state:
            if st.button("🎲 Use Sample Data"):
                st.session_state['match_data'] = self._generate_sample_data()
                st.success("Sample data loaded!")
        
        # Simulation settings
        st.subheader("⚙️ Settings")
        
        st.session_state['auto_suggest'] = st.checkbox(
            "Auto-suggest scenarios",
            value=True,
            help="Automatically suggest relevant counterfactual scenarios"
        )
        
        st.session_state['num_suggestions'] = st.slider(
            "Number of suggestions",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of scenarios to auto-suggest"
        )
        
        st.session_state['show_confidence'] = st.checkbox(
            "Show confidence intervals",
            value=True,
            help="Display confidence intervals for predictions"
        )
    
    def _render_scenario_selection(self):
        """Render scenario selection interface."""
        st.header("📝 Scenario Selection")
        
        if 'match_data' not in st.session_state:
            st.info("Please upload match data or use sample data to begin simulation.")
            return
        
        match_data = st.session_state['match_data']
        
        # Add ball_id if not present
        if 'ball_id' not in match_data.columns:
            match_data['ball_id'] = [f"ball_{i}" for i in range(len(match_data))]
            st.session_state['match_data'] = match_data
        
        # Select target ball for simulation
        st.subheader("🎯 Target Ball")
        
        # Show recent balls
        recent_balls = match_data.tail(10)
        
        ball_options = []
        for _, row in recent_balls.iterrows():
            ball_desc = f"Over {row.get('over', 0)}.{row.get('ball', 0)} - "
            ball_desc += f"{row.get('batter', 'Unknown')} vs {row.get('bowler', 'Unknown')} - "
            ball_desc += f"{row.get('runs_scored', 0)} runs"
            if row.get('wicket_type'):
                ball_desc += f" (WICKET: {row.get('wicket_type')})"
            
            ball_options.append((row.get('ball_id', f"ball_{len(ball_options)}"), ball_desc))
        
        selected_ball_id = st.selectbox(
            "Select ball to simulate from:",
            options=[ball_id for ball_id, _ in ball_options],
            format_func=lambda x: next(desc for ball_id, desc in ball_options if ball_id == x),
            help="Choose the ball from which to run counterfactual simulations"
        )
        
        if selected_ball_id:
            st.session_state['selected_ball_id'] = selected_ball_id
            
            # Create simulation context
            try:
                context = create_simulation_context_from_match_state(
                    match_data, selected_ball_id
                )
                st.session_state['simulation_context'] = context
                
                # Show current match state
                self._render_match_state(context)
                
            except Exception as e:
                st.error(f"Error creating simulation context: {e}")
                return
        
        # Scenario creation
        st.subheader("🎭 Create Scenarios")
        
        # Auto-suggested scenarios
        if st.session_state.get('auto_suggest', True):
            if st.button("🔮 Generate Suggested Scenarios"):
                self._generate_suggested_scenarios()
        
        # Manual scenario creation
        with st.expander("➕ Create Custom Scenario"):
            self._render_custom_scenario_creator()
        
        # Display created scenarios
        self._render_scenario_list()
    
    def _render_match_state(self, context: SimulationContext):
        """Render current match state."""
        st.subheader("📊 Current Match State")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score", f"{context.current_score}/{context.wickets_fallen}")
            st.metric("Over", f"{context.over}")
        
        with col2:
            st.metric("Balls Remaining", context.balls_remaining)
            if context.target_score:
                st.metric("Target", context.target_score)
        
        with col3:
            if context.required_run_rate:
                st.metric("Required RR", f"{context.required_run_rate:.1f}")
            st.metric("Innings", context.innings)
    
    def _render_custom_scenario_creator(self):
        """Render custom scenario creation interface."""
        scenario_type = st.selectbox(
            "Scenario Type",
            options=[
                CounterfactualEventType.CATCH_TAKEN,
                CounterfactualEventType.CATCH_DROPPED,
                CounterfactualEventType.BOUNDARY_HIT,
                CounterfactualEventType.BOUNDARY_STOPPED,
                CounterfactualEventType.RUN_OUT_SUCCESS,
                CounterfactualEventType.RUN_OUT_FAILED
            ],
            format_func=lambda x: x.value.replace('_', ' ').title()
        )
        
        description = st.text_input(
            "Scenario Description",
            placeholder="e.g., 'Kohli's catch taken at deep cover'"
        )
        
        # Get original outcome for selected ball
        if 'selected_ball_id' in st.session_state and 'match_data' in st.session_state:
            match_data = st.session_state['match_data']
            ball_row = match_data[match_data['ball_id'] == st.session_state['selected_ball_id']]
            
            if len(ball_row) > 0:
                original_outcome = ball_row.iloc[0].to_dict()
                
                # Modify outcome based on scenario type
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Outcome:**")
                    st.write(f"Runs: {original_outcome.get('runs_scored', 0)}")
                    st.write(f"Wicket: {original_outcome.get('wicket_type', 'None')}")
                
                with col2:
                    st.write("**Modified Outcome:**")
                    new_runs = st.number_input("Runs", min_value=0, max_value=6, value=0)
                    new_wicket = st.selectbox(
                        "Wicket Type",
                        options=['None', 'caught', 'bowled', 'lbw', 'run_out', 'stumped'],
                        index=0
                    )
                
                if st.button("Create Scenario"):
                    modified_outcome = original_outcome.copy()
                    modified_outcome['runs_scored'] = new_runs
                    modified_outcome['wicket_type'] = new_wicket if new_wicket != 'None' else None
                    
                    event = CounterfactualEvent(
                        event_type=scenario_type,
                        ball_id=st.session_state['selected_ball_id'],
                        original_outcome=original_outcome,
                        modified_outcome=modified_outcome,
                        description=description or f"Custom {scenario_type.value}"
                    )
                    
                    if 'scenarios' not in st.session_state:
                        st.session_state['scenarios'] = []
                    
                    st.session_state['scenarios'].append(event)
                    st.success("Scenario created!")
                    st.rerun()
    
    def _generate_suggested_scenarios(self):
        """Generate and store suggested scenarios."""
        if 'simulation_context' not in st.session_state or 'match_data' not in st.session_state:
            st.error("Please select a target ball first")
            return
        
        context = st.session_state['simulation_context']
        match_data = st.session_state['match_data']
        num_suggestions = st.session_state.get('num_suggestions', 5)
        
        try:
            suggested_scenarios = self.simulator.generate_suggested_scenarios(
                context, match_data, num_suggestions
            )
            
            st.session_state['scenarios'] = suggested_scenarios
            st.success(f"Generated {len(suggested_scenarios)} suggested scenarios!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating scenarios: {e}")
    
    def _render_scenario_list(self):
        """Render list of created scenarios."""
        if 'scenarios' not in st.session_state or not st.session_state['scenarios']:
            st.info("No scenarios created yet. Use auto-suggest or create custom scenarios.")
            return
        
        st.subheader("📋 Created Scenarios")
        
        scenarios = st.session_state['scenarios']
        
        for i, scenario in enumerate(scenarios):
            with st.expander(f"Scenario {i+1}: {scenario.description}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original:**")
                    st.json(scenario.original_outcome, expanded=False)
                
                with col2:
                    st.write("**Modified:**")
                    st.json(scenario.modified_outcome, expanded=False)
                
                if st.button(f"🚀 Simulate Scenario {i+1}", key=f"sim_{i}"):
                    self._run_simulation(scenario)
        
        # Bulk simulation
        if len(scenarios) > 1:
            if st.button("🎯 Simulate All Scenarios"):
                self._run_bulk_simulation()
    
    def _render_simulation_results(self):
        """Render simulation results."""
        st.header("📈 Simulation Results")
        
        if 'simulation_results' not in st.session_state:
            st.info("Select and simulate scenarios to see results here.")
            return
        
        results = st.session_state['simulation_results']
        
        if isinstance(results, list):
            # Multiple scenarios
            self._render_bulk_results(results)
        else:
            # Single scenario
            self._render_single_result(results)
    
    def _render_single_result(self, comparison: SimulationComparison):
        """Render results for a single scenario."""
        st.subheader(f"🎭 {comparison.event.description}")
        
        # Impact summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_color = "normal" if abs(comparison.win_probability_impact) < 0.05 else "inverse"
            st.metric(
                "Win Probability Impact",
                f"{comparison.win_probability_impact:+.1%}",
                delta=f"{comparison.impact_magnitude.title()} impact",
                delta_color=delta_color
            )
        
        with col2:
            st.metric(
                "Score Impact",
                f"{comparison.score_impact:+.1f} runs",
                delta="per over"
            )
        
        with col3:
            st.metric(
                "Wicket Impact",
                f"{comparison.wicket_impact:+.1%}",
                delta="probability"
            )
        
        # Side-by-side comparison
        st.subheader("📊 Detailed Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Original Prediction**")
            self._render_prediction_card(comparison.original_prediction, "original")
        
        with col2:
            st.write("**🔮 Counterfactual Prediction**")
            self._render_prediction_card(comparison.counterfactual_prediction, "counterfactual")
        
        # Visualization
        self._render_comparison_chart(comparison)
    
    def _render_prediction_card(self, prediction, prefix):
        """Render a prediction card."""
        # Win probability
        win_prob_color = "green" if prediction.win_probability > 0.5 else "red"
        st.markdown(
            f"<div style='padding: 10px; border-radius: 5px; background-color: #f0f2f6;'>"
            f"<h4 style='color: {win_prob_color}; margin: 0;'>Win Probability: {prediction.win_probability:.1%}</h4>"
            f"<p><strong>Over Runs:</strong> {prediction.over_runs_prediction:.1f}</p>"
            f"<p><strong>Next Ball:</strong> {prediction.next_ball_runs} runs</p>"
            f"<p><strong>Wicket Chance:</strong> {prediction.wicket_probability:.1%}</p>"
            f"<p><strong>Boundary Chance:</strong> {prediction.boundary_probability:.1%}</p>"
            f"<p><strong>Confidence:</strong> {prediction.prediction_confidence:.1%}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Runs distribution
        runs_dist = prediction.runs_distribution
        if runs_dist:
            st.write("**Runs Distribution:**")
            
            fig = px.bar(
                x=list(runs_dist.keys()),
                y=list(runs_dist.values()),
                title="Probability by Runs",
                labels={'x': 'Runs', 'y': 'Probability'}
            )
            fig.update_layout(height=200, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_comparison_chart(self, comparison: SimulationComparison):
        """Render comparison visualization."""
        st.subheader("📊 Impact Visualization")
        
        # Create comparison metrics
        metrics = ['Win Probability', 'Over Runs', 'Wicket Probability', 'Boundary Probability']
        original_values = [
            comparison.original_prediction.win_probability,
            comparison.original_prediction.over_runs_prediction / 10,  # Normalize
            comparison.original_prediction.wicket_probability,
            comparison.original_prediction.boundary_probability
        ]
        counterfactual_values = [
            comparison.counterfactual_prediction.win_probability,
            comparison.counterfactual_prediction.over_runs_prediction / 10,  # Normalize
            comparison.counterfactual_prediction.wicket_probability,
            comparison.counterfactual_prediction.boundary_probability
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=original_values,
            theta=metrics,
            fill='toself',
            name='Original',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=counterfactual_values,
            theta=metrics,
            fill='toself',
            name='Counterfactual',
            line_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Prediction Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_bulk_results(self, comparisons: List[SimulationComparison]):
        """Render results for multiple scenarios."""
        st.subheader(f"🎯 Bulk Simulation Results ({len(comparisons)} scenarios)")
        
        # Summary statistics
        impact_analysis = self.simulator.analyze_scenario_impact(comparisons)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Win Impact", f"{impact_analysis['avg_win_prob_impact']:+.1%}")
        
        with col2:
            st.metric("Max Win Impact", f"{impact_analysis['max_win_prob_impact']:+.1%}")
        
        with col3:
            st.metric("Avg Score Impact", f"{impact_analysis['avg_score_impact']:+.1f}")
        
        with col4:
            st.metric("High Impact", f"{impact_analysis['high_impact_scenarios']} scenarios")
        
        # Impact distribution
        st.subheader("📊 Impact Distribution")
        
        impact_dist = impact_analysis['impact_distribution']
        
        fig = px.pie(
            values=list(impact_dist.values()),
            names=list(impact_dist.keys()),
            title="Impact Magnitude Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual scenario results
        st.subheader("📋 Individual Scenarios")
        
        for i, comparison in enumerate(comparisons):
            with st.expander(f"Scenario {i+1}: {comparison.event.description}"):
                self._render_single_result(comparison)
    
    def _run_simulation(self, scenario: CounterfactualEvent):
        """Run simulation for a single scenario."""
        if 'simulation_context' not in st.session_state or 'match_data' not in st.session_state:
            st.error("Missing simulation context or match data")
            return
        
        context = st.session_state['simulation_context']
        match_data = st.session_state['match_data']
        
        try:
            with st.spinner("Running simulation..."):
                comparison = self.simulator.simulate_counterfactual(
                    context, match_data, scenario
                )
                st.session_state['simulation_results'] = comparison
                st.success("Simulation completed!")
                st.rerun()
        
        except Exception as e:
            st.error(f"Simulation failed: {e}")
    
    def _run_bulk_simulation(self):
        """Run simulation for all scenarios."""
        if 'simulation_context' not in st.session_state or 'match_data' not in st.session_state:
            st.error("Missing simulation context or match data")
            return
        
        context = st.session_state['simulation_context']
        match_data = st.session_state['match_data']
        scenarios = st.session_state.get('scenarios', [])
        
        if not scenarios:
            st.error("No scenarios to simulate")
            return
        
        try:
            with st.spinner(f"Running {len(scenarios)} simulations..."):
                comparisons = self.simulator.simulate_multiple_scenarios(
                    context, match_data, scenarios
                )
                st.session_state['simulation_results'] = comparisons
                st.success(f"Completed {len(comparisons)} simulations!")
                st.rerun()
        
        except Exception as e:
            st.error(f"Bulk simulation failed: {e}")
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample match data for demonstration."""
        np.random.seed(42)  # For reproducible results
        
        data = []
        
        # Generate a T20 match
        for over in range(1, 21):  # 20 overs
            for ball in range(1, 7):  # 6 balls per over
                ball_id = f"match1_1_{over}_{ball}"
                
                # Generate realistic outcomes
                runs = np.random.choice([0, 1, 2, 3, 4, 6], p=[0.35, 0.3, 0.15, 0.05, 0.1, 0.05])
                
                # Wickets more likely in certain situations
                wicket_prob = 0.03
                if over >= 16:  # Death overs
                    wicket_prob = 0.05
                
                wicket = np.random.choice([None, 'caught', 'bowled', 'lbw'], 
                                        p=[1-wicket_prob, wicket_prob*0.5, wicket_prob*0.3, wicket_prob*0.2])
                
                data.append({
                    'ball_id': ball_id,
                    'match_id': 'sample_match',
                    'innings': 1,
                    'over': over,
                    'ball': ball,
                    'batter': 'Kohli' if over <= 10 else 'Dhoni',
                    'bowler': 'Starc' if over % 2 == 1 else 'Cummins',
                    'runs_scored': runs,
                    'wicket_type': wicket
                })
        
        return pd.DataFrame(data)


def render_match_simulator():
    """Main function to render the match simulator UI."""
    simulator_ui = MatchSimulatorUI()
    simulator_ui.render()