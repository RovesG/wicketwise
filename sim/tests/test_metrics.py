# Purpose: Test SIM metrics calculation functionality
# Author: WicketWise AI, Last Modified: 2024

import pytest
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.metrics import SimMetrics, KPICalculator, TradeRecord, PerformanceSnapshot
from sim.strategy import StrategyAction, FillEvent, OrderSide, OrderType, AccountState
from sim.state import MatchState, MarketState


class TestKPICalculator:
    """Test KPI calculation functions"""
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        # Test with positive returns
        returns = [0.01, 0.02, -0.005, 0.015, 0.008]
        sharpe = KPICalculator.calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Should be positive for profitable returns
        
        # Test with empty returns
        sharpe_empty = KPICalculator.calculate_sharpe_ratio([])
        assert sharpe_empty == 0.0
        
        # Test with single return
        sharpe_single = KPICalculator.calculate_sharpe_ratio([0.01])
        assert sharpe_single == 0.0
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation"""
        returns = [0.02, -0.01, 0.015, -0.008, 0.01]
        sortino = KPICalculator.calculate_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        assert sortino > 0  # Should be positive for profitable returns
        
        # Test with no negative returns
        positive_returns = [0.01, 0.02, 0.015, 0.008]
        sortino_positive = KPICalculator.calculate_sortino_ratio(positive_returns)
        assert sortino_positive == float('inf')
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Simulate balance history with drawdown
        balance_history = [100000, 105000, 102000, 108000, 95000, 110000]
        max_dd = KPICalculator.calculate_max_drawdown(balance_history)
        
        assert isinstance(max_dd, float)
        assert max_dd >= 0  # Drawdown should be positive percentage
        
        # Test with monotonically increasing balance (no drawdown)
        increasing_balance = [100000, 105000, 110000, 115000]
        max_dd_none = KPICalculator.calculate_max_drawdown(increasing_balance)
        assert max_dd_none == 0.0
        
        # Test with empty history
        max_dd_empty = KPICalculator.calculate_max_drawdown([])
        assert max_dd_empty == 0.0
    
    def test_calmar_ratio_calculation(self):
        """Test Calmar ratio calculation"""
        total_return = 15.0  # 15% return
        max_drawdown = 5.0   # 5% max drawdown
        
        calmar = KPICalculator.calculate_calmar_ratio(total_return, max_drawdown)
        assert calmar == 3.0  # 15% / 5%
        
        # Test with zero drawdown
        calmar_no_dd = KPICalculator.calculate_calmar_ratio(10.0, 0.0)
        assert calmar_no_dd == float('inf')
        
        # Test with negative return
        calmar_negative = KPICalculator.calculate_calmar_ratio(-5.0, 10.0)
        assert calmar_negative == -0.5
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        trades = [
            TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, 50.0, 0.0, None),   # Win
            TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, -30.0, 0.0, None),  # Loss
            TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, 25.0, 0.0, None),   # Win
            TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, -10.0, 0.0, None),  # Loss
        ]
        
        hit_rate = KPICalculator.calculate_hit_rate(trades)
        assert hit_rate == 0.5  # 2 wins out of 4 trades
        
        # Test with empty trades
        hit_rate_empty = KPICalculator.calculate_hit_rate([])
        assert hit_rate_empty == 0.0
    
    def test_average_edge_calculation(self):
        """Test average edge calculation"""
        trades = [
            TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, 50.0, 0.0, 0.02),
            TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, -30.0, 0.0, 0.015),
            TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, 25.0, 0.0, 0.025),
        ]
        
        avg_edge = KPICalculator.calculate_average_edge(trades)
        expected_avg = (0.02 + 0.015 + 0.025) / 3
        assert abs(avg_edge - expected_avg) < 0.001
        
        # Test with no edges
        trades_no_edge = [
            TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, 50.0, 0.0, None),
        ]
        avg_edge_none = KPICalculator.calculate_average_edge(trades_no_edge)
        assert avg_edge_none == 0.0
    
    def test_slippage_calculation(self):
        """Test slippage calculation"""
        fills_and_actions = [
            (FillEvent("order1", 100.0, 2.0, 5.0, 10.0, "filled", ""), 
             StrategyAction("", OrderType.LIMIT, OrderSide.BACK, "", "", 2.0, 100.0, "order1")),
            (FillEvent("order2", 200.0, 1.95, 8.0, 5.0, "filled", ""), 
             StrategyAction("", OrderType.LIMIT, OrderSide.BACK, "", "", 1.95, 200.0, "order2")),
        ]
        
        avg_slippage = KPICalculator.calculate_slippage_bps(fills_and_actions)
        expected_avg = (10.0 + 5.0) / 2
        assert avg_slippage == expected_avg
        
        # Test with empty fills
        avg_slippage_empty = KPICalculator.calculate_slippage_bps([])
        assert avg_slippage_empty == 0.0
    
    def test_fill_rate_calculation(self):
        """Test fill rate calculation"""
        actions = [
            StrategyAction("", OrderType.LIMIT, OrderSide.BACK, "", "", 2.0, 100.0, "order1"),
            StrategyAction("", OrderType.LIMIT, OrderSide.BACK, "", "", 2.0, 100.0, "order2"),
            StrategyAction("", OrderType.LIMIT, OrderSide.BACK, "", "", 2.0, 100.0, "order3"),
            StrategyAction("", OrderType.LIMIT, OrderSide.BACK, "", "", 2.0, 100.0, "order4"),
        ]
        
        fills = [
            FillEvent("order1", 100.0, 2.0, 5.0, 0.0, "filled", ""),
            FillEvent("order2", 50.0, 2.0, 2.5, 0.0, "partial", ""),  # Partial fill
            FillEvent("order4", 0.0, 0.0, 0.0, 0.0, "no_fill", ""),   # No fill
        ]
        
        fill_rate = KPICalculator.calculate_fill_rate(actions, fills)
        assert fill_rate == 0.5  # 2 filled orders out of 4 actions
        
        # Test with empty actions
        fill_rate_empty = KPICalculator.calculate_fill_rate([], fills)
        assert fill_rate_empty == 0.0


class TestSimMetrics:
    """Test SimMetrics functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.metrics = SimMetrics()
        self.initial_balance = 100000.0
        self.metrics.initialize(self.initial_balance)
        
        self.account_state = AccountState(cash=self.initial_balance)
        self.match_state = MatchState("test_venue")
        self.market_state = MarketState()
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        assert self.metrics.initial_balance == self.initial_balance
        assert len(self.metrics.balance_history) == 1
        assert self.metrics.balance_history[0] == self.initial_balance
        assert len(self.metrics.pnl_history) == 1
        assert self.metrics.pnl_history[0] == 0.0
    
    def test_tick_updates(self):
        """Test metrics updates on tick"""
        # Update account state
        self.account_state.cash = 95000.0
        self.account_state.realized_pnl = -5000.0
        
        self.metrics.update_tick(self.match_state, self.market_state, self.account_state)
        
        # Should have updated history
        assert len(self.metrics.balance_history) == 2
        assert self.metrics.balance_history[-1] == 90000.0  # 95000 + (-5000)
        assert self.metrics.pnl_history[-1] == -10000.0  # 90000 - 100000
    
    def test_action_recording(self):
        """Test action recording"""
        action = StrategyAction(
            ts=datetime.now().isoformat(),
            type=OrderType.LIMIT,
            side=OrderSide.BACK,
            market_id="match_odds",
            selection_id="home",
            price=2.0,
            size=100.0,
            client_order_id="test_order"
        )
        
        self.metrics.update_action(action)
        
        assert len(self.metrics.actions) == 1
        assert self.metrics.actions[0] == action
    
    def test_fill_recording(self):
        """Test fill recording and trade tracking"""
        action = StrategyAction(
            ts=datetime.now().isoformat(),
            type=OrderType.LIMIT,
            side=OrderSide.BACK,
            market_id="match_odds",
            selection_id="home",
            price=2.0,
            size=100.0,
            client_order_id="test_order"
        )
        
        fill = FillEvent(
            client_order_id="test_order",
            fill_qty=100.0,
            avg_price=2.0,
            fees=5.0,
            slippage=0.0,
            reason="filled",
            ts=datetime.now().isoformat()
        )
        
        self.metrics.update_fill(fill, action, self.account_state)
        
        assert len(self.metrics.fills) == 1
        assert self.metrics.fills[0][0] == fill
        assert self.metrics.fills[0][1] == action
    
    def test_trade_tracking(self):
        """Test individual trade tracking"""
        # Open position
        action_open = StrategyAction(
            ts=datetime.now().isoformat(),
            type=OrderType.LIMIT,
            side=OrderSide.BACK,
            market_id="match_odds",
            selection_id="home",
            price=2.0,
            size=100.0,
            client_order_id="open_order"
        )
        
        fill_open = FillEvent(
            client_order_id="open_order",
            fill_qty=100.0,
            avg_price=2.0,
            fees=5.0,
            slippage=0.0,
            reason="filled",
            ts=datetime.now().isoformat()
        )
        
        # Update account state to reflect position
        self.account_state.update_from_fill(fill_open, action_open)
        self.metrics.update_fill(fill_open, action_open, self.account_state)
        
        # Should have open position
        position_key = "match_odds:home"
        assert position_key in self.metrics.open_positions
    
    def test_performance_snapshots(self):
        """Test performance snapshot creation"""
        # Update multiple times to trigger snapshot
        for i in range(101):  # Should trigger snapshot at 100th update
            self.account_state.cash = self.initial_balance - i * 100
            self.metrics.update_tick(self.match_state, self.market_state, self.account_state)
        
        # Should have created at least one snapshot
        assert len(self.metrics.snapshots) > 0
        
        snapshot = self.metrics.snapshots[0]
        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.balance > 0
        assert snapshot.pnl <= 0  # Should be negative due to decreasing cash
    
    def test_final_kpis_calculation(self):
        """Test final KPIs calculation"""
        # Simulate some trading activity
        for i in range(10):
            self.account_state.cash = self.initial_balance + i * 1000  # Increasing balance
            self.account_state.realized_pnl = i * 1000
            self.metrics.update_tick(self.match_state, self.market_state, self.account_state)
        
        # Add some trades
        trade1 = TradeRecord(
            entry_time=datetime.now().isoformat(),
            exit_time=(datetime.now() + timedelta(minutes=5)).isoformat(),
            market_id="match_odds",
            selection_id="home",
            side=OrderSide.BACK,
            entry_price=2.0,
            exit_price=2.1,
            size=100.0,
            pnl=500.0,
            duration_seconds=300.0,
            edge=0.02
        )
        self.metrics.trades.append(trade1)
        
        kpis = self.metrics.calculate_final_kpis(self.account_state, self.initial_balance)
        
        # Check KPI structure
        assert kpis.pnl_total > 0  # Should be profitable
        assert kpis.sharpe >= 0
        assert kpis.max_drawdown >= 0
        assert kpis.hit_rate >= 0
        assert kpis.num_trades == 1
    
    def test_phase_metrics_tracking(self):
        """Test phase-specific metrics tracking"""
        phase_metrics = self.metrics.get_phase_breakdown()
        
        # Should have all phases initialized
        assert "powerplay" in phase_metrics
        assert "middle" in phase_metrics
        assert "death" in phase_metrics
        
        for phase in phase_metrics.values():
            assert "pnl" in phase
            assert "trades" in phase
    
    def test_trade_analysis(self):
        """Test detailed trade analysis"""
        # Add some trades
        winning_trade = TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, 50.0, 0.0, None)
        losing_trade = TradeRecord("", None, "", "", OrderSide.BACK, 2.0, None, 100.0, -30.0, 0.0, None)
        
        self.metrics.trades.extend([winning_trade, losing_trade])
        
        analysis = self.metrics.get_trade_analysis()
        
        assert analysis["total_trades"] == 2
        assert analysis["winning_trades"] == 1
        assert analysis["losing_trades"] == 1
        assert analysis["hit_rate"] == 0.5
        assert analysis["avg_win"] == 50.0
        assert analysis["avg_loss"] == -30.0
        assert analysis["largest_win"] == 50.0
        assert analysis["largest_loss"] == -30.0
        assert analysis["profit_factor"] > 0
    
    def test_time_series_export(self):
        """Test time series data export"""
        # Add some data points
        for i in range(5):
            self.account_state.cash = self.initial_balance + i * 1000
            self.metrics.update_tick(self.match_state, self.market_state, self.account_state)
        
        time_series = self.metrics.export_time_series()
        
        assert "timestamps" in time_series
        assert "balance" in time_series
        assert "pnl" in time_series
        assert "exposure" in time_series
        
        # Should have 6 data points (initial + 5 updates)
        assert len(time_series["timestamps"]) == 6
        assert len(time_series["balance"]) == 6
        assert len(time_series["pnl"]) == 6
        assert len(time_series["exposure"]) == 6
    
    def test_edge_estimation(self):
        """Test edge estimation functionality"""
        action = StrategyAction(
            ts=datetime.now().isoformat(),
            type=OrderType.LIMIT,
            side=OrderSide.BACK,
            market_id="match_odds",
            selection_id="home",
            price=2.5,  # Longer odds
            size=100.0,
            client_order_id="test_order"
        )
        
        edge = self.metrics._estimate_edge(action)
        
        assert edge is not None
        assert edge > 0
        assert edge <= 0.1  # Should be reasonable percentage


if __name__ == "__main__":
    pytest.main([__file__])
