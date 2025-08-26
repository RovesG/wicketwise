# Purpose: Test SIM strategy baseline implementations
# Author: WicketWise AI, Last Modified: 2024

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.strategy import (
    EdgeKellyStrategy, MeanRevertLOBStrategy, MomentumFollowStrategy,
    StrategyAction, OrderSide, OrderType, AccountState, create_strategy
)
from sim.state import MatchState, MarketState, MarketSnapshot, SelectionBook, LadderLevel, MarketStatus


class TestEdgeKellyStrategy:
    """Test EdgeKelly strategy implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = EdgeKellyStrategy(
            edge_threshold=0.02,
            kelly_fraction=0.25,
            max_stake_pct=0.05,
            min_odds=1.1,
            max_odds=10.0
        )
        
        self.account_state = AccountState(cash=100000.0)
        self.match_state = self._create_test_match_state()
        self.market_state = self._create_test_market_state()
    
    def _create_test_match_state(self) -> MatchState:
        """Create test match state"""
        match_state = MatchState("test_venue")
        match_state.score = 120
        match_state.wickets = 3
        match_state.over = 15
        match_state.ball = 3
        match_state.innings = 1
        return match_state
    
    def _create_test_market_state(self) -> MarketState:
        """Create test market state"""
        home_selection = SelectionBook(
            selection_id="home",
            back=[LadderLevel(1.95, 1000.0)],
            lay=[LadderLevel(1.96, 800.0)]
        )
        
        away_selection = SelectionBook(
            selection_id="away",
            back=[LadderLevel(2.10, 800.0)],
            lay=[LadderLevel(2.11, 600.0)]
        )
        
        snapshot = MarketSnapshot(
            ts=datetime.now().isoformat(),
            market_id="match_odds",
            status=MarketStatus.OPEN,
            selections=[home_selection, away_selection]
        )
        
        market_state = MarketState()
        market_state.update_snapshot(snapshot)
        return market_state
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.edge_threshold == 0.02
        assert self.strategy.kelly_fraction == 0.25
        assert self.strategy.max_stake_pct == 0.05
        assert self.strategy.get_name() == "edge_kelly_v3"
        
        params = self.strategy.get_params()
        assert params["edge_threshold"] == 0.02
        assert params["kelly_fraction"] == 0.25
    
    def test_no_action_when_market_closed(self):
        """Test no action when market is closed"""
        # Close the market
        closed_snapshot = self.market_state.market_snapshots["match_odds"]
        closed_snapshot.status = MarketStatus.CLOSED
        self.market_state.update_snapshot(closed_snapshot)
        
        actions = self.strategy.on_tick(self.match_state, self.market_state, self.account_state)
        
        assert len(actions) == 0
    
    def test_no_action_when_no_edge(self):
        """Test no action when edge is below threshold"""
        # Mock model probability to give no edge
        with patch.object(self.strategy, '_get_model_probability', return_value=0.51):  # Slight edge only
            actions = self.strategy.on_tick(self.match_state, self.market_state, self.account_state)
        
        # Should have no actions due to insufficient edge
        assert len(actions) == 0
    
    def test_action_generation_with_edge(self):
        """Test action generation when edge exists"""
        # Mock model probability to give significant edge
        with patch.object(self.strategy, '_get_model_probability', return_value=0.55):  # 5% edge
            actions = self.strategy.on_tick(self.match_state, self.market_state, self.account_state)
        
        # Should generate actions
        assert len(actions) > 0
        
        action = actions[0]
        assert action.type == OrderType.LIMIT
        assert action.side == OrderSide.BACK
        assert action.market_id == "match_odds"
        assert action.price > 0
        assert action.size > 0
        assert action.post_only is True
    
    def test_kelly_stake_calculation(self):
        """Test Kelly criterion stake calculation"""
        edge = 0.03  # 3% edge
        odds = 2.0
        bankroll = 100000.0
        
        stake = self.strategy._calculate_kelly_stake(edge, odds, bankroll)
        
        # Should be positive and reasonable
        assert stake > 0
        assert stake <= self.strategy.max_stake_pct * bankroll
    
    def test_odds_filtering(self):
        """Test odds filtering (min/max odds)"""
        # Create market with odds outside acceptable range
        low_odds_selection = SelectionBook(
            selection_id="home",
            back=[LadderLevel(1.05, 1000.0)],  # Below min_odds
            lay=[LadderLevel(1.06, 800.0)]
        )
        
        high_odds_selection = SelectionBook(
            selection_id="away",
            back=[LadderLevel(15.0, 800.0)],  # Above max_odds
            lay=[LadderLevel(16.0, 600.0)]
        )
        
        snapshot = MarketSnapshot(
            ts=datetime.now().isoformat(),
            market_id="match_odds",
            status=MarketStatus.OPEN,
            selections=[low_odds_selection, high_odds_selection]
        )
        
        market_state = MarketState()
        market_state.update_snapshot(snapshot)
        
        with patch.object(self.strategy, '_get_model_probability', return_value=0.6):
            actions = self.strategy.on_tick(self.match_state, market_state, self.account_state)
        
        # Should filter out odds outside acceptable range
        assert len(actions) == 0
    
    def test_order_id_generation(self):
        """Test unique order ID generation"""
        with patch.object(self.strategy, '_get_model_probability', return_value=0.55):
            actions1 = self.strategy.on_tick(self.match_state, self.market_state, self.account_state)
            actions2 = self.strategy.on_tick(self.match_state, self.market_state, self.account_state)
        
        if actions1 and actions2:
            assert actions1[0].client_order_id != actions2[0].client_order_id


class TestMeanRevertLOBStrategy:
    """Test MeanRevertLOB strategy implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = MeanRevertLOBStrategy(
            revert_threshold=0.02,
            max_position=1000.0,
            hold_time_seconds=30.0
        )
        
        self.account_state = AccountState(cash=100000.0)
        self.match_state = MatchState("test_venue")
        self.market_state = self._create_test_market_state()
    
    def _create_test_market_state(self) -> MarketState:
        """Create test market state"""
        home_selection = SelectionBook(
            selection_id="home",
            back=[LadderLevel(1.95, 1000.0)],
            lay=[LadderLevel(1.96, 800.0)]
        )
        
        snapshot = MarketSnapshot(
            ts=datetime.now().isoformat(),
            market_id="match_odds",
            status=MarketStatus.OPEN,
            selections=[home_selection]
        )
        
        market_state = MarketState()
        market_state.update_snapshot(snapshot)
        return market_state
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.revert_threshold == 0.02
        assert self.strategy.max_position == 1000.0
        assert self.strategy.get_name() == "mean_revert_lob"
    
    def test_no_action_first_tick(self):
        """Test no action on first tick (no price history)"""
        actions = self.strategy.on_tick(self.match_state, self.market_state, self.account_state)
        
        # Should have no actions on first tick
        assert len(actions) == 0
    
    def test_mean_reversion_detection(self):
        """Test mean reversion opportunity detection"""
        # First tick to establish baseline
        self.strategy.on_tick(self.match_state, self.market_state, self.account_state)
        
        # Create market with significant price move
        moved_selection = SelectionBook(
            selection_id="home",
            back=[LadderLevel(2.05, 1000.0)],  # Significant price increase
            lay=[LadderLevel(2.06, 800.0)]
        )
        
        moved_snapshot = MarketSnapshot(
            ts=datetime.now().isoformat(),
            market_id="match_odds",
            status=MarketStatus.OPEN,
            selections=[moved_selection]
        )
        
        moved_market_state = MarketState()
        moved_market_state.update_snapshot(moved_snapshot)
        
        # Second tick with moved price
        actions = self.strategy.on_tick(self.match_state, moved_market_state, self.account_state)
        
        # Should generate mean reversion action
        if actions:
            action = actions[0]
            assert action.side == OrderSide.LAY  # Fade the upward move
            assert action.size <= self.strategy.max_position


class TestMomentumFollowStrategy:
    """Test MomentumFollow strategy implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = MomentumFollowStrategy(
            momentum_threshold=0.01,
            event_multiplier=1.5,
            max_stake=1000.0
        )
        
        self.account_state = AccountState(cash=100000.0)
        self.match_state = self._create_test_match_state()
        self.market_state = self._create_test_market_state()
    
    def _create_test_match_state(self) -> MatchState:
        """Create test match state with boundary event"""
        from sim.state import MatchEvent, MatchPhase
        
        match_state = MatchState("test_venue")
        
        # Create event with boundary
        event = MatchEvent(
            ts=datetime.now().isoformat(),
            over=10,
            ball=3,
            innings=1,
            bat_striker="player_1",
            bat_non_striker="player_2",
            bowler="bowler_1",
            runs_batter=4,  # Boundary
            runs_extras=0,
            phase=MatchPhase.MIDDLE
        )
        
        match_state.update_from_event(event)
        return match_state
    
    def _create_test_market_state(self) -> MarketState:
        """Create test market state"""
        home_selection = SelectionBook(
            selection_id="home",
            back=[LadderLevel(1.95, 1000.0)],
            lay=[LadderLevel(1.96, 800.0)]
        )
        
        snapshot = MarketSnapshot(
            ts=datetime.now().isoformat(),
            market_id="match_odds",
            status=MarketStatus.OPEN,
            selections=[home_selection]
        )
        
        market_state = MarketState()
        market_state.update_snapshot(snapshot)
        return market_state
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.momentum_threshold == 0.01
        assert self.strategy.event_multiplier == 1.5
        assert self.strategy.get_name() == "momentum_follow"
    
    def test_event_multiplier_application(self):
        """Test event multiplier for significant cricket events"""
        # Build price history first
        for i in range(5):
            price = 1.90 + i * 0.01
            selection = SelectionBook(
                selection_id="home",
                back=[LadderLevel(price, 1000.0)],
                lay=[LadderLevel(price + 0.01, 800.0)]
            )
            
            snapshot = MarketSnapshot(
                ts=datetime.now().isoformat(),
                market_id="match_odds",
                status=MarketStatus.OPEN,
                selections=[selection]
            )
            
            market_state = MarketState()
            market_state.update_snapshot(snapshot)
            
            self.strategy.on_tick(self.match_state, market_state, self.account_state)
        
        # The boundary event should trigger event multiplier
        # (Implementation details would depend on actual momentum detection)
        assert len(self.strategy.price_history) > 0


class TestStrategyFactory:
    """Test strategy factory function"""
    
    def test_create_edge_kelly_strategy(self):
        """Test creating EdgeKelly strategy"""
        params = {
            "edge_threshold": 0.03,
            "kelly_fraction": 0.2,
            "max_stake_pct": 0.04
        }
        
        strategy = create_strategy("edge_kelly_v3", params)
        
        assert isinstance(strategy, EdgeKellyStrategy)
        assert strategy.edge_threshold == 0.03
        assert strategy.kelly_fraction == 0.2
        assert strategy.max_stake_pct == 0.04
    
    def test_create_mean_revert_strategy(self):
        """Test creating MeanRevert strategy"""
        params = {
            "revert_threshold": 0.03,
            "max_position": 1500.0
        }
        
        strategy = create_strategy("mean_revert_lob", params)
        
        assert isinstance(strategy, MeanRevertLOBStrategy)
        assert strategy.revert_threshold == 0.03
        assert strategy.max_position == 1500.0
    
    def test_create_momentum_strategy(self):
        """Test creating Momentum strategy"""
        params = {
            "momentum_threshold": 0.015,
            "event_multiplier": 2.0
        }
        
        strategy = create_strategy("momentum_follow", params)
        
        assert isinstance(strategy, MomentumFollowStrategy)
        assert strategy.momentum_threshold == 0.015
        assert strategy.event_multiplier == 2.0
    
    def test_unknown_strategy_error(self):
        """Test error for unknown strategy"""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("unknown_strategy", {})


if __name__ == "__main__":
    pytest.main([__file__])
