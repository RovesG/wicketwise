# Purpose: Streamlit-based match replay visualizer for cricket ball-by-ball analysis
# Author: Assistant, Last Modified: 2024

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Set up matplotlib and seaborn styling
plt.style.use('default')
sns.set_palette("husl")

class ReplayVisualizer:
    """
    Cricket match replay visualizer with predictions and betting insights.
    
    Loads eval_predictions.csv and provides interactive ball-by-ball analysis
    with actual vs predicted outcomes, win probability progression, and
    shadow betting decisions.
    """
    
    def __init__(self):
        """Initialize the replay visualizer."""
        self.data = None
        self.matches = []
        self.current_match = None
        self.current_ball_idx = 0
        
        # Outcome color mapping
        self.outcome_colors = {
            "0_runs": "#E8E8E8",      # Light gray
            "1_run": "#B3D9FF",       # Light blue  
            "2_runs": "#80C7FF",      # Blue
            "3_runs": "#4DB5FF",      # Medium blue
            "4_runs": "#1A9AFF",      # Bright blue
            "6_runs": "#0066CC",      # Dark blue
            "wicket": "#FF4444"       # Red
        }
        
        # Betting decision colors
        self.betting_colors = {
            "value_bet": "#28A745",    # Green
            "no_bet": "#6C757D",       # Gray
            "risk_alert": "#DC3545"    # Red
        }
        
    def load_data(self, csv_path: str) -> bool:
        """
        Load evaluation predictions from CSV file.
        
        Args:
            csv_path: Path to eval_predictions.csv file
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            if not Path(csv_path).exists():
                st.error(f"File not found: {csv_path}")
                return False
            
            # Load CSV data
            self.data = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = [
                "match_id", "ball_id", "actual_runs", "predicted_runs_class",
                "win_prob", "odds_mispricing", "phase", "batter_id", "bowler_id"
            ]
            
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Process data
            self.data = self._process_data(self.data)
            
            # Extract unique matches
            self.matches = sorted(self.data['match_id'].unique())
            
            logger.info(f"Loaded {len(self.data)} predictions from {len(self.matches)} matches")
            st.success(f"✅ Loaded {len(self.data):,} predictions from {len(self.matches)} matches")
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enhance the loaded data.
        
        Args:
            df: Raw dataframe from CSV
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Make a copy to avoid pandas warnings
        df = df.copy()
        
        # Convert actual_runs to string for consistent comparison
        df['actual_runs_str'] = df['actual_runs'].apply(self._runs_to_outcome_class)
        
        # Calculate prediction accuracy
        df['prediction_correct'] = (df['actual_runs_str'] == df['predicted_runs_class'])
        
        # Generate betting decisions based on mispricing and confidence
        df['betting_decision'] = df.apply(self._generate_betting_decision, axis=1)
        
        # Add ball sequence number within each match
        df['ball_sequence'] = df.groupby('match_id').cumcount() + 1
        
        # Calculate cumulative win probability change
        df['win_prob_change'] = df.groupby('match_id')['win_prob'].diff().fillna(0)
        
        # Add over and ball number
        df['over_ball'] = df.apply(self._extract_over_ball, axis=1)
        
        return df
    
    def _runs_to_outcome_class(self, runs: int) -> str:
        """Convert runs scored to outcome class string."""
        if runs == 0:
            return "0_runs"
        elif runs == 1:
            return "1_run"
        elif runs == 2:
            return "2_runs"
        elif runs == 3:
            return "3_runs"
        elif runs == 4:
            return "4_runs"
        elif runs == 6:
            return "6_runs"
        else:
            return "wicket"  # Assume other values are wickets
    
    def _generate_betting_decision(self, row: pd.Series) -> str:
        """
        Generate shadow betting decision based on mispricing and confidence.
        
        Args:
            row: DataFrame row with prediction data
            
        Returns:
            str: Betting decision (value_bet, no_bet, risk_alert)
        """
        mispricing = row['odds_mispricing']
        win_prob = row['win_prob']
        
        # Simple betting logic
        if mispricing > 0.15 and win_prob > 0.6:
            return "value_bet"
        elif mispricing < -0.15 or win_prob < 0.2:
            return "risk_alert"
        else:
            return "no_bet"
    
    def _extract_over_ball(self, row: pd.Series) -> str:
        """Extract over.ball format from ball_id if possible."""
        try:
            # Try to extract over.ball from ball_id
            ball_id = str(row['ball_id'])
            if '.' in ball_id:
                return ball_id
            else:
                # Generate synthetic over.ball based on sequence
                ball_seq = row.get('ball_sequence', 1)
                over_num = ((ball_seq - 1) // 6) + 1
                ball_num = ((ball_seq - 1) % 6) + 1
                return f"{over_num}.{ball_num}"
        except:
            return "1.1"
    
    def render_match_selector(self) -> Optional[str]:
        """
        Render match selection interface.
        
        Returns:
            str: Selected match ID or None
        """
        if not self.matches:
            st.warning("No matches available. Please load data first.")
            return None
        
        # Match selector
        selected_match = st.selectbox(
            "Select Match",
            options=self.matches,
            index=0 if self.current_match is None else self.matches.index(self.current_match),
            help="Choose a match to replay"
        )
        
        return selected_match
    
    def render_ball_navigation(self, match_data: pd.DataFrame) -> int:
        """
        Render ball navigation interface.
        
        Args:
            match_data: DataFrame containing match data
            
        Returns:
            int: Selected ball index
        """
        total_balls = len(match_data)
        
        # Ball slider
        ball_idx = st.slider(
            "Ball Number",
            min_value=0,
            max_value=total_balls - 1,
            value=min(self.current_ball_idx, total_balls - 1),
            help=f"Navigate through {total_balls} balls in this match"
        )
        
        # Previous/Next buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⏮️ Previous", disabled=(ball_idx <= 0)):
                ball_idx = max(0, ball_idx - 1)
        
        with col2:
            st.write(f"Ball {ball_idx + 1} of {total_balls}")
        
        with col3:
            if st.button("Next ⏭️", disabled=(ball_idx >= total_balls - 1)):
                ball_idx = min(total_balls - 1, ball_idx + 1)
        
        return ball_idx
    
    def render_ball_details(self, ball_data: pd.Series):
        """
        Render detailed information for a specific ball.
        
        Args:
            ball_data: Series containing ball data
        """
        # Ball header
        st.subheader(f"⚾ Ball {ball_data['over_ball']} - {ball_data['phase'].title()}")
        
        # Main metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Actual outcome
            actual_color = self.outcome_colors.get(ball_data['actual_runs_str'], "#E8E8E8")
            st.markdown(f"""
            <div style="background-color: {actual_color}; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>Actual</strong><br>
                {ball_data['actual_runs_str'].replace('_', ' ').title()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Predicted outcome
            pred_color = self.outcome_colors.get(ball_data['predicted_runs_class'], "#E8E8E8")
            is_correct = ball_data['prediction_correct']
            border_color = "#28A745" if is_correct else "#DC3545"
            
            st.markdown(f"""
            <div style="background-color: {pred_color}; padding: 10px; border-radius: 5px; 
                        text-align: center; border: 2px solid {border_color};">
                <strong>Predicted</strong><br>
                {ball_data['predicted_runs_class'].replace('_', ' ').title()}
                <br>{'✅' if is_correct else '❌'}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Win probability
            win_prob = ball_data['win_prob']
            win_color = "#28A745" if win_prob > 0.6 else "#DC3545" if win_prob < 0.4 else "#FFC107"
            
            st.markdown(f"""
            <div style="background-color: {win_color}; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>Win Probability</strong><br>
                {win_prob:.1%}
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Betting decision
            betting_decision = ball_data['betting_decision']
            betting_color = self.betting_colors.get(betting_decision, "#6C757D")
            
            st.markdown(f"""
            <div style="background-color: {betting_color}; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>Betting Signal</strong><br>
                {betting_decision.replace('_', ' ').title()}
            </div>
            """, unsafe_allow_html=True)
        
        # Additional details
        st.markdown("---")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write("**Players:**")
            st.write(f"🏏 Batter: {ball_data['batter_id']}")
            st.write(f"⚾ Bowler: {ball_data['bowler_id']}")
            st.write(f"📊 Odds Mispricing: {ball_data['odds_mispricing']:.3f}")
        
        with detail_col2:
            st.write("**Context:**")
            st.write(f"🎯 Ball ID: {ball_data['ball_id']}")
            st.write(f"🏟️ Phase: {ball_data['phase'].title()}")
            if 'win_prob_change' in ball_data:
                change = ball_data['win_prob_change']
                arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                st.write(f"{arrow} Win Prob Change: {change:+.1%}")
    
    def render_win_probability_chart(self, match_data: pd.DataFrame, current_ball: int):
        """
        Render win probability progression chart.
        
        Args:
            match_data: DataFrame containing match data
            current_ball: Current ball index
        """
        st.subheader("📈 Win Probability Progression")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot win probability line
        x_values = range(len(match_data))
        y_values = match_data['win_prob'].values
        
        ax.plot(x_values, y_values, linewidth=2, color='#1f77b4', alpha=0.8)
        ax.fill_between(x_values, y_values, alpha=0.3, color='#1f77b4')
        
        # Highlight current ball
        if current_ball < len(match_data):
            ax.axvline(x=current_ball, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.scatter([current_ball], [y_values[current_ball]], 
                      color='red', s=100, zorder=5)
        
        # Styling
        ax.set_xlabel('Ball Number')
        ax.set_ylabel('Win Probability')
        ax.set_title('Win Probability Throughout Match')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add horizontal reference lines
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0.75, color='gray', linestyle=':', alpha=0.5)
        
        st.pyplot(fig)
        plt.close()
    
    def render_prediction_accuracy_chart(self, match_data: pd.DataFrame):
        """
        Render prediction accuracy visualization.
        
        Args:
            match_data: DataFrame containing match data
        """
        st.subheader("🎯 Prediction Accuracy Analysis")
        
        # Calculate accuracy by phase
        accuracy_by_phase = match_data.groupby('phase')['prediction_correct'].mean()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy by phase
        accuracy_by_phase.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Prediction Accuracy by Phase')
        ax1.set_ylabel('Accuracy Rate')
        ax1.set_xlabel('Match Phase')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add percentage labels on bars
        for i, v in enumerate(accuracy_by_phase.values):
            ax1.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # Outcome distribution
        outcome_counts = match_data['actual_runs_str'].value_counts()
        colors = [self.outcome_colors.get(outcome, '#E8E8E8') for outcome in outcome_counts.index]
        
        ax2.pie(outcome_counts.values, labels=outcome_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Actual Outcome Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def render_betting_analysis(self, match_data: pd.DataFrame):
        """
        Render betting decision analysis.
        
        Args:
            match_data: DataFrame containing match data
        """
        st.subheader("💰 Betting Decision Analysis")
        
        # Betting decision distribution
        betting_dist = match_data['betting_decision'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Betting decision pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = [self.betting_colors.get(decision, '#6C757D') for decision in betting_dist.index]
            
            ax.pie(betting_dist.values, labels=betting_dist.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax.set_title('Betting Decision Distribution')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Betting metrics
            st.write("**Betting Metrics:**")
            
            total_balls = len(match_data)
            value_bets = (match_data['betting_decision'] == 'value_bet').sum()
            risk_alerts = (match_data['betting_decision'] == 'risk_alert').sum()
            
            st.metric("Total Balls", total_balls)
            st.metric("Value Bets", value_bets, f"{value_bets/total_balls:.1%}")
            st.metric("Risk Alerts", risk_alerts, f"{risk_alerts/total_balls:.1%}")
            
            # Average mispricing by decision
            avg_mispricing = match_data.groupby('betting_decision')['odds_mispricing'].mean()
            st.write("**Average Mispricing by Decision:**")
            for decision, mispricing in avg_mispricing.items():
                st.write(f"• {decision.replace('_', ' ').title()}: {mispricing:.3f}")
    
    def render_filters(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """
        Render sidebar filters and return filtered data.
        
        Args:
            match_data: DataFrame containing match data
            
        Returns:
            pd.DataFrame: Filtered match data
        """
        st.sidebar.header("🔍 Filters")
        
        # Batter filter
        batters = ['All'] + sorted(match_data['batter_id'].unique())
        selected_batter = st.sidebar.selectbox("Filter by Batter", batters)
        
        # Bowler filter
        bowlers = ['All'] + sorted(match_data['bowler_id'].unique())
        selected_bowler = st.sidebar.selectbox("Filter by Bowler", bowlers)
        
        # Phase filter
        phases = ['All'] + sorted(match_data['phase'].unique())
        selected_phase = st.sidebar.selectbox("Filter by Phase", phases)
        
        # Over range filter
        max_over = match_data['ball_sequence'].max() // 6 + 1
        over_range = st.sidebar.slider(
            "Over Range",
            min_value=1,
            max_value=int(max_over),
            value=(1, int(max_over)),
            help="Filter balls by over range"
        )
        
        # Apply filters
        filtered_data = match_data.copy()
        
        if selected_batter != 'All':
            filtered_data = filtered_data[filtered_data['batter_id'] == selected_batter]
        
        if selected_bowler != 'All':
            filtered_data = filtered_data[filtered_data['bowler_id'] == selected_bowler]
        
        if selected_phase != 'All':
            filtered_data = filtered_data[filtered_data['phase'] == selected_phase]
        
        # Over range filter
        over_min, over_max = over_range
        ball_min = (over_min - 1) * 6 + 1
        ball_max = over_max * 6
        filtered_data = filtered_data[
            (filtered_data['ball_sequence'] >= ball_min) & 
            (filtered_data['ball_sequence'] <= ball_max)
        ]
        
        # Show filter summary
        st.sidebar.write(f"**Filtered Results:** {len(filtered_data)} balls")
        
        return filtered_data


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="WicketWise Match Replay",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏏 WicketWise Match Replay Visualizer")
    st.markdown("Interactive ball-by-ball match analysis with predictions and betting insights")
    
    # Initialize visualizer
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = ReplayVisualizer()
    
    visualizer = st.session_state.visualizer
    
    # File upload/selection
    st.sidebar.header("📁 Data Source")
    
    # Option to upload file or specify path
    upload_option = st.sidebar.radio(
        "Data Source",
        ["Upload File", "File Path"],
        help="Choose how to load eval_predictions.csv"
    )
    
    csv_path = None
    
    if upload_option == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload eval_predictions.csv",
            type=['csv'],
            help="Upload the CSV file generated by eval.py"
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path("temp_eval_predictions.csv")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            csv_path = str(temp_path)
    else:
        csv_path = st.sidebar.text_input(
            "File Path",
            value="eval_predictions.csv",
            help="Enter the path to eval_predictions.csv"
        )
    
    # Load data button
    if st.sidebar.button("Load Data", disabled=(csv_path is None)):
        if visualizer.load_data(csv_path):
            st.session_state.data_loaded = True
        else:
            st.session_state.data_loaded = False
    
    # Main interface
    if getattr(st.session_state, 'data_loaded', False) and visualizer.data is not None:
        # Match selection
        selected_match = visualizer.render_match_selector()
        
        if selected_match:
            # Get match data
            match_data = visualizer.data[visualizer.data['match_id'] == selected_match].copy()
            match_data = match_data.sort_values('ball_sequence').reset_index(drop=True)
            
            # Apply filters
            filtered_data = visualizer.render_filters(match_data)
            
            if len(filtered_data) > 0:
                # Ball navigation
                ball_idx = visualizer.render_ball_navigation(filtered_data)
                
                # Update session state
                visualizer.current_match = selected_match
                visualizer.current_ball_idx = ball_idx
                
                # Display ball details
                current_ball = filtered_data.iloc[ball_idx]
                visualizer.render_ball_details(current_ball)
                
                # Charts and analysis
                st.markdown("---")
                
                # Win probability chart
                visualizer.render_win_probability_chart(filtered_data, ball_idx)
                
                # Analysis tabs
                tab1, tab2 = st.tabs(["📊 Prediction Analysis", "💰 Betting Analysis"])
                
                with tab1:
                    visualizer.render_prediction_accuracy_chart(filtered_data)
                
                with tab2:
                    visualizer.render_betting_analysis(filtered_data)
                
            else:
                st.warning("No balls match the current filter criteria.")
        
        # Match summary
        if st.sidebar.checkbox("Show Match Summary"):
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Match Summary")
            
            if selected_match and visualizer.data is not None:
                match_data = visualizer.data[visualizer.data['match_id'] == selected_match]
                
                total_balls = len(match_data)
                accuracy = match_data['prediction_correct'].mean()
                value_bets = (match_data['betting_decision'] == 'value_bet').sum()
                
                st.sidebar.metric("Total Balls", total_balls)
                st.sidebar.metric("Prediction Accuracy", f"{accuracy:.1%}")
                st.sidebar.metric("Value Bets", value_bets)
    
    else:
        # Instructions
        st.info("👆 Please load an eval_predictions.csv file to begin match replay analysis.")
        
        # Show expected format
        with st.expander("📋 Expected CSV Format"):
            st.markdown("""
            The CSV file should contain the following columns:
            - `match_id`: Unique identifier for each match
            - `ball_id`: Unique identifier for each ball
            - `actual_runs`: Actual runs scored (0-6 or wicket indicator)
            - `predicted_runs_class`: Predicted outcome class (e.g., '2_runs', 'wicket')
            - `win_prob`: Win probability (0.0 to 1.0)
            - `odds_mispricing`: Betting odds mispricing value
            - `phase`: Match phase (e.g., 'powerplay', 'middle_overs')
            - `batter_id`: Batter identifier
            - `bowler_id`: Bowler identifier
            
            This file is typically generated by running `eval.py` on a trained model.
            """)
    
    # Cleanup temporary file
    if upload_option == "Upload File":
        temp_path = Path("temp_eval_predictions.csv")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass


if __name__ == "__main__":
    main() 