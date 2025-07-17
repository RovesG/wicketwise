# Purpose: Offline betting simulator for cricket match predictions with bankroll management
# Author: Assistant, Last Modified: 2024

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import csv
from dataclasses import dataclass

# Configure logger
logger = logging.getLogger(__name__)

class StakingMethod(Enum):
    """Staking methods for betting simulation."""
    FLAT = "flat"
    KELLY = "kelly"
    PROPORTIONAL = "proportional"

class BettingDecision(Enum):
    """Betting decision types."""
    NO_BET = "no_bet"
    VALUE_BET = "value_bet"
    RISK_ALERT = "risk_alert"

@dataclass
class BettingConfig:
    """Configuration for betting simulation."""
    initial_bankroll: float = 1000.0
    flat_stake: float = 10.0
    kelly_fraction: float = 0.25  # Fractional Kelly for safety
    min_stake: float = 1.0
    max_stake: float = 100.0
    min_odds: float = 1.1
    max_odds: float = 10.0
    value_threshold: float = 0.05  # Minimum edge to place bet
    confidence_threshold: float = 0.6  # Minimum win probability to bet


class OfflineBettingSimulator:
    """
    Offline betting simulator for cricket match predictions.
    
    Simulates betting strategies based on model predictions and mispricing
    analysis without connecting to any real betting platforms.
    """
    
    def __init__(self, config: Optional[BettingConfig] = None):
        """
        Initialize the betting simulator.
        
        Args:
            config: Betting configuration parameters
        """
        self.config = config or BettingConfig()
        self.bankroll = self.config.initial_bankroll
        self.predictions_data = None
        self.betting_log = []
        
        # Performance tracking
        self.total_bets = 0
        self.winning_bets = 0
        self.total_staked = 0.0
        self.total_returns = 0.0
        
        logger.info(f"Initialized betting simulator with £{self.config.initial_bankroll} bankroll")
    
    def load_predictions(self, csv_path: str) -> bool:
        """
        Load predictions from eval_predictions.csv.
        
        Args:
            csv_path: Path to eval_predictions.csv file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if not Path(csv_path).exists():
                logger.error(f"Predictions file not found: {csv_path}")
                return False
            
            # Load predictions data
            self.predictions_data = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = [
                "match_id", "ball_id", "actual_runs", "predicted_runs_class",
                "win_prob", "odds_mispricing", "phase"
            ]
            
            missing_columns = [col for col in required_columns if col not in self.predictions_data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Process predictions for betting
            self.predictions_data = self._process_predictions(self.predictions_data)
            
            logger.info(f"Loaded {len(self.predictions_data)} predictions from {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            return False
    
    def _process_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process predictions data for betting simulation.
        
        Args:
            df: Raw predictions dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe with betting fields
        """
        df = df.copy()
        
        # Generate simulated odds based on win probability and mispricing
        df['simulated_odds'] = df.apply(self._generate_odds, axis=1)
        
        # Calculate implied probability from odds
        df['implied_probability'] = 1 / df['simulated_odds']
        
        # Calculate betting edge (value)
        df['betting_edge'] = df['win_prob'] - df['implied_probability']
        
        # Determine betting decision
        df['betting_decision'] = df.apply(self._determine_betting_decision, axis=1)
        
        # Add ball sequence for ordering
        df['ball_sequence'] = df.groupby('match_id').cumcount() + 1
        
        return df
    
    def _generate_odds(self, row: pd.Series) -> float:
        """
        Generate simulated odds based on win probability and mispricing.
        
        Args:
            row: Row from predictions dataframe
            
        Returns:
            float: Simulated odds
        """
        win_prob = row['win_prob']
        mispricing = row['odds_mispricing']
        
        # Convert win probability to fair odds
        fair_odds = 1 / win_prob if win_prob > 0 else 10.0
        
        # Apply mispricing to create market odds
        # Positive mispricing = odds too high (value bet)
        # Negative mispricing = odds too low (avoid)
        market_odds = fair_odds * (1 + mispricing)
        
        # Clamp odds to reasonable range
        market_odds = np.clip(market_odds, self.config.min_odds, self.config.max_odds)
        
        return round(market_odds, 2)
    
    def _determine_betting_decision(self, row: pd.Series) -> str:
        """
        Determine betting decision based on edge and confidence.
        
        Args:
            row: Row from predictions dataframe
            
        Returns:
            str: Betting decision
        """
        betting_edge = row['betting_edge']
        win_prob = row['win_prob']
        
        # Check if bet meets criteria
        if (betting_edge > self.config.value_threshold and 
            win_prob > self.config.confidence_threshold):
            return BettingDecision.VALUE_BET.value
        elif betting_edge < -self.config.value_threshold:
            return BettingDecision.RISK_ALERT.value
        else:
            return BettingDecision.NO_BET.value
    
    def _calculate_stake(self, row: pd.Series, staking_method: StakingMethod) -> float:
        """
        Calculate stake size based on staking method.
        
        Args:
            row: Row from predictions dataframe
            staking_method: Staking method to use
            
        Returns:
            float: Stake amount
        """
        if staking_method == StakingMethod.FLAT:
            return self.config.flat_stake
        
        elif staking_method == StakingMethod.KELLY:
            # Kelly Criterion: f = (bp - q) / b
            # where b = odds - 1, p = win probability, q = 1 - p
            odds = row['simulated_odds']
            win_prob = row['win_prob']
            
            b = odds - 1
            p = win_prob
            q = 1 - p
            
            if b > 0 and p > 0:
                kelly_fraction = (b * p - q) / b
                # Apply fractional Kelly for safety
                kelly_fraction *= self.config.kelly_fraction
                
                # Calculate stake as fraction of bankroll
                stake = max(0, kelly_fraction * self.bankroll)
            else:
                stake = 0
        
        elif staking_method == StakingMethod.PROPORTIONAL:
            # Stake proportional to betting edge
            betting_edge = row['betting_edge']
            stake = max(0, betting_edge * self.bankroll * 0.1)  # 10% of edge
        
        else:
            stake = self.config.flat_stake
        
        # Apply stake limits
        stake = np.clip(stake, self.config.min_stake, self.config.max_stake)
        
        # Don't stake more than available bankroll
        stake = min(stake, self.bankroll)
        
        return round(stake, 2)
    
    def _determine_bet_result(self, row: pd.Series) -> bool:
        """
        Determine if bet won based on actual outcome.
        
        Args:
            row: Row from predictions dataframe
            
        Returns:
            bool: True if bet won, False otherwise
        """
        # For simplicity, assume we're betting on team win probability
        # In a real scenario, this would depend on the specific bet type
        
        # Use a random outcome based on the true win probability
        # This simulates the uncertainty in actual match outcomes
        actual_win_prob = row.get('actual_win_prob', row['win_prob'])
        
        # Add some noise to simulate real-world uncertainty
        noise = np.random.normal(0, 0.1)
        adjusted_prob = np.clip(actual_win_prob + noise, 0, 1)
        
        # Determine outcome
        return np.random.random() < adjusted_prob
    
    def simulate_betting(self, staking_method: StakingMethod = StakingMethod.FLAT, 
                        output_file: str = "betting_log.csv") -> Dict:
        """
        Simulate betting strategy on loaded predictions.
        
        Args:
            staking_method: Staking method to use
            output_file: Output file for betting log
            
        Returns:
            Dict: Simulation results summary
        """
        if self.predictions_data is None:
            raise ValueError("No predictions loaded. Call load_predictions() first.")
        
        logger.info(f"Starting betting simulation with {staking_method.value} staking")
        
        # Reset simulation state
        self.bankroll = self.config.initial_bankroll
        self.betting_log = []
        self.total_bets = 0
        self.winning_bets = 0
        self.total_staked = 0.0
        self.total_returns = 0.0
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Process each prediction
        for idx, row in self.predictions_data.iterrows():
            self._process_betting_opportunity(row, staking_method)
        
        # Save betting log
        self._save_betting_log(output_file)
        
        # Calculate final results
        results = self._calculate_results()
        
        logger.info(f"Simulation completed. Final bankroll: £{self.bankroll:.2f}")
        logger.info(f"Total bets: {self.total_bets}, Win rate: {self.winning_bets/max(1,self.total_bets):.2%}")
        
        return results
    
    def _process_betting_opportunity(self, row: pd.Series, staking_method: StakingMethod):
        """
        Process a single betting opportunity.
        
        Args:
            row: Row from predictions dataframe
            staking_method: Staking method to use
        """
        match_id = row['match_id']
        ball_id = row['ball_id']
        decision = row['betting_decision']
        odds = row['simulated_odds']
        
        # Initialize betting record
        bet_record = {
            'match_id': match_id,
            'ball_id': ball_id,
            'decision': decision,
            'odds': odds,
            'stake': 0.0,
            'result': 'no_bet',
            'pnl': 0.0,
            'bankroll': self.bankroll
        }
        
        # Only place bet if decision is value bet
        if decision == BettingDecision.VALUE_BET.value and self.bankroll > 0:
            # Calculate stake
            stake = self._calculate_stake(row, staking_method)
            
            if stake > 0:
                # Place bet
                bet_record['stake'] = stake
                self.total_bets += 1
                self.total_staked += stake
                
                # Determine bet result
                bet_won = self._determine_bet_result(row)
                
                if bet_won:
                    # Calculate winnings
                    winnings = stake * odds
                    pnl = winnings - stake
                    self.bankroll += pnl
                    self.winning_bets += 1
                    self.total_returns += winnings
                    
                    bet_record['result'] = 'won'
                    bet_record['pnl'] = pnl
                else:
                    # Lose stake
                    pnl = -stake
                    self.bankroll += pnl
                    
                    bet_record['result'] = 'lost'
                    bet_record['pnl'] = pnl
                
                bet_record['bankroll'] = self.bankroll
        
        # Add to betting log
        self.betting_log.append(bet_record)
    
    def _save_betting_log(self, output_file: str):
        """
        Save betting log to CSV file.
        
        Args:
            output_file: Output file path
        """
        if not self.betting_log:
            logger.warning("No betting log to save")
            return
        
        # Define CSV headers
        headers = ['match_id', 'ball_id', 'decision', 'odds', 'stake', 'result', 'pnl', 'bankroll']
        
        try:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.betting_log)
            
            logger.info(f"Betting log saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving betting log: {str(e)}")
    
    def _calculate_results(self) -> Dict:
        """
        Calculate simulation results summary.
        
        Returns:
            Dict: Results summary
        """
        if self.total_bets == 0:
            return {
                'initial_bankroll': self.config.initial_bankroll,
                'final_bankroll': self.bankroll,
                'total_bets': 0,
                'winning_bets': 0,
                'win_rate': 0.0,
                'total_staked': 0.0,
                'total_returns': 0.0,
                'net_profit': 0.0,
                'roi': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate basic metrics
        net_profit = self.bankroll - self.config.initial_bankroll
        roi = net_profit / self.config.initial_bankroll * 100
        win_rate = self.winning_bets / self.total_bets * 100
        
        # Calculate maximum drawdown
        bankroll_history = [record['bankroll'] for record in self.betting_log]
        peak_bankroll = self.config.initial_bankroll
        max_drawdown = 0.0
        
        for bankroll in bankroll_history:
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            drawdown = (peak_bankroll - bankroll) / peak_bankroll * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'initial_bankroll': self.config.initial_bankroll,
            'final_bankroll': self.bankroll,
            'total_bets': self.total_bets,
            'winning_bets': self.winning_bets,
            'win_rate': win_rate,
            'total_staked': self.total_staked,
            'total_returns': self.total_returns,
            'net_profit': net_profit,
            'roi': roi,
            'max_drawdown': max_drawdown
        }
    
    def analyze_performance(self) -> Dict:
        """
        Analyze betting performance with detailed metrics.
        
        Returns:
            Dict: Performance analysis
        """
        if not self.betting_log:
            return {}
        
        # Convert betting log to DataFrame for analysis
        log_df = pd.DataFrame(self.betting_log)
        
        # Filter only actual bets
        bets_df = log_df[log_df['stake'] > 0].copy()
        
        if len(bets_df) == 0:
            return {}
        
        # Calculate performance metrics
        analysis = {
            'total_opportunities': len(log_df),
            'betting_frequency': len(bets_df) / len(log_df) * 100,
            'average_stake': bets_df['stake'].mean(),
            'average_odds': bets_df['odds'].mean(),
            'largest_win': bets_df['pnl'].max(),
            'largest_loss': bets_df['pnl'].min(),
            'average_win': bets_df[bets_df['result'] == 'won']['pnl'].mean() if len(bets_df[bets_df['result'] == 'won']) > 0 else 0,
            'average_loss': bets_df[bets_df['result'] == 'lost']['pnl'].mean() if len(bets_df[bets_df['result'] == 'lost']) > 0 else 0,
        }
        
        # Calculate Sharpe ratio (simplified)
        returns = bets_df['pnl'] / bets_df['stake']
        if len(returns) > 1:
            analysis['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            analysis['sharpe_ratio'] = 0
        
        return analysis
    
    def get_betting_summary(self) -> str:
        """
        Get a formatted summary of betting simulation results.
        
        Returns:
            str: Formatted summary
        """
        results = self._calculate_results()
        analysis = self.analyze_performance()
        
        summary = f"""
🏏 BETTING SIMULATION RESULTS
{'=' * 50}
💰 Financial Performance:
   Initial Bankroll: £{results['initial_bankroll']:.2f}
   Final Bankroll:   £{results['final_bankroll']:.2f}
   Net Profit:       £{results['net_profit']:.2f}
   ROI:              {results['roi']:.2f}%
   Max Drawdown:     {results['max_drawdown']:.2f}%

📊 Betting Statistics:
   Total Bets:       {results['total_bets']}
   Winning Bets:     {results['winning_bets']}
   Win Rate:         {results['win_rate']:.2f}%
   Total Staked:     £{results['total_staked']:.2f}
   Total Returns:    £{results['total_returns']:.2f}
"""
        
        if analysis:
            summary += f"""
🎯 Performance Analysis:
   Betting Frequency: {analysis['betting_frequency']:.2f}%
   Average Stake:     £{analysis['average_stake']:.2f}
   Average Odds:      {analysis['average_odds']:.2f}
   Largest Win:       £{analysis['largest_win']:.2f}
   Largest Loss:      £{analysis['largest_loss']:.2f}
   Sharpe Ratio:      {analysis['sharpe_ratio']:.3f}
"""
        
        return summary


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Offline Betting Simulator")
    parser.add_argument(
        "--predictions",
        type=str,
        default="eval_predictions.csv",
        help="Path to predictions CSV file"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll amount"
    )
    parser.add_argument(
        "--staking",
        type=str,
        choices=["flat", "kelly", "proportional"],
        default="flat",
        help="Staking method"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="betting_log.csv",
        help="Output file for betting log"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Create betting config
    config = BettingConfig(initial_bankroll=args.bankroll)
    
    # Initialize simulator
    simulator = OfflineBettingSimulator(config)
    
    # Load predictions
    if not simulator.load_predictions(args.predictions):
        print("❌ Failed to load predictions")
        return
    
    # Run simulation
    staking_method = StakingMethod(args.staking)
    results = simulator.simulate_betting(staking_method, args.output)
    
    # Print results
    print(simulator.get_betting_summary())


if __name__ == "__main__":
    main() 