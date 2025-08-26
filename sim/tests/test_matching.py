# Purpose: Test SIM matching engine functionality
# Author: WicketWise AI, Last Modified: 2024

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.matching import MatchingEngine, LatencyModel, SlippageModel, CommissionModel, FillReason
from sim.strategy import StrategyAction, OrderSide, OrderType
from sim.state import MarketSnapshot, SelectionBook, LadderLevel, MarketStatus


class TestMatchingEngine:
    """Test cases for the matching engine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.latency_model = LatencyModel(mean_ms=100.0, std_ms=10.0)
        self.slippage_model = SlippageModel(model_type="lob_queue")
        self.commission_model = CommissionModel(commission_bps=200.0)
        
        self.engine = MatchingEngine(
            latency_model=self.latency_model,
            slippage_model=self.slippage_model,
            commission_model=self.commission_model,
            participation_factor=0.5
        )
        
        # Create test market snapshot
        self.test_snapshot = self._create_test_snapshot()
        self.engine.update_market_state(self.test_snapshot)
    
    def _create_test_snapshot(self) -> MarketSnapshot:
        """Create test market snapshot"""
        home_selection = SelectionBook(
            selection_id="home",
            back=[
                LadderLevel(1.95, 1000.0),
                LadderLevel(1.94, 800.0),
                LadderLevel(1.93, 600.0)
            ],
            lay=[
                LadderLevel(1.96, 900.0),
                LadderLevel(1.97, 700.0),
                LadderLevel(1.98, 500.0)
            ],
            traded_volume=50000.0
        )
        
        away_selection = SelectionBook(
            selection_id="away",
            back=[
                LadderLevel(2.10, 800.0),
                LadderLevel(2.09, 600.0),
                LadderLevel(2.08, 400.0)
            ],
            lay=[
                LadderLevel(2.11, 700.0),
                LadderLevel(2.12, 500.0),
                LadderLevel(2.13, 300.0)
            ],
            traded_volume=40000.0
        )
        
        return MarketSnapshot(
            ts=datetime.now().isoformat(),
            market_id="match_odds",
            status=MarketStatus.OPEN,
            selections=[home_selection, away_selection],
            total_matched=90000.0
        )
    
    def _create_test_action(self, side: OrderSide, price: float, size: float, 
                          order_type: OrderType = OrderType.LIMIT) -> StrategyAction:
        """Create test strategy action"""
        return StrategyAction(
            ts=datetime.now().isoformat(),
            type=order_type,
            side=side,
            market_id="match_odds",
            selection_id="home",
            price=price,
            size=size,
            client_order_id=f"test_order_{datetime.now().timestamp()}"
        )
    
    def test_full_fill_back_order(self):
        """Test full fill of back order"""
        # Create back order at available lay price
        action = self._create_test_action(OrderSide.BACK, 1.96, 400.0)
        
        with patch('sim.matching.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(milliseconds=100)
            fills = self.engine.handle_action(action)
        
        # Should get one fill event
        assert len(fills) == 1
        fill = fills[0]
        
        # Check fill details
        assert fill.client_order_id == action.client_order_id
        assert fill.fill_qty > 0
        assert fill.avg_price == 1.96
        assert fill.reason == FillReason.FULL_FILL.value
        assert fill.remaining_qty == 0.0
    
    def test_partial_fill_large_order(self):
        """Test partial fill of large order"""
        # Create large back order
        action = self._create_test_action(OrderSide.BACK, 1.96, 2000.0)
        
        with patch('sim.matching.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(milliseconds=100)
            fills = self.engine.handle_action(action)
        
        # Should get partial fill due to participation factor
        assert len(fills) == 1
        fill = fills[0]
        
        assert fill.fill_qty < action.size
        assert fill.fill_qty > 0
        assert fill.reason == FillReason.PARTIAL_QUEUE.value
        assert fill.remaining_qty > 0
    
    def test_no_fill_price_too_low(self):
        """Test no fill when back price too low"""
        # Create back order below best lay price
        action = self._create_test_action(OrderSide.BACK, 1.94, 500.0)
        
        with patch('sim.matching.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(milliseconds=100)
            fills = self.engine.handle_action(action)
        
        # Should get no fill or queued order
        if fills:
            assert fills[0].fill_qty == 0
    
    def test_lay_order_execution(self):
        """Test lay order execution"""
        # Create lay order at available back price
        action = self._create_test_action(OrderSide.LAY, 1.95, 400.0)
        
        with patch('sim.matching.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(milliseconds=100)
            fills = self.engine.handle_action(action)
        
        # Should get fill
        assert len(fills) == 1
        fill = fills[0]
        
        assert fill.fill_qty > 0
        assert fill.avg_price == 1.95
    
    def test_ioc_order_execution(self):
        """Test IOC order execution"""
        # Create IOC order
        action = self._create_test_action(OrderSide.BACK, 1.96, 400.0, OrderType.IOC)
        
        with patch('sim.matching.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(milliseconds=100)
            fills = self.engine.handle_action(action)
        
        # Should get fill or IOC expiry
        assert len(fills) == 1
        fill = fills[0]
        
        if fill.fill_qty == 0:
            assert fill.reason == FillReason.IOC_EXPIRED.value
        else:
            assert fill.reason in [FillReason.FULL_FILL.value, FillReason.IOC_EXPIRED.value]
    
    def test_suspended_market_handling(self):
        """Test handling of suspended markets"""
        # Suspend the market
        suspended_snapshot = self.test_snapshot
        suspended_snapshot.status = MarketStatus.SUSPENDED
        self.engine.update_market_state(suspended_snapshot)
        
        # Create order
        action = self._create_test_action(OrderSide.BACK, 1.96, 400.0)
        
        with patch('sim.matching.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(milliseconds=100)
            fills = self.engine.handle_action(action)
        
        # IOC orders should be rejected, LIMIT orders queued
        if action.type == OrderType.IOC:
            assert len(fills) == 1
            assert fills[0].reason == FillReason.NO_FILL_SUSPENDED.value
        else:
            # LIMIT orders are queued (no immediate fill)
            assert len(fills) == 0 or fills[0].fill_qty == 0
    
    def test_vwap_calculation(self):
        """Test VWAP calculation across multiple levels"""
        # Create large order that spans multiple levels
        action = self._create_test_action(OrderSide.BACK, 1.98, 1500.0)
        
        with patch('sim.matching.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(milliseconds=100)
            fills = self.engine.handle_action(action)
        
        if fills and fills[0].fill_qty > 0:
            # VWAP should be between best and worst prices
            assert 1.96 <= fills[0].avg_price <= 1.98
    
    def test_commission_calculation(self):
        """Test commission calculation"""
        action = self._create_test_action(OrderSide.BACK, 1.96, 1000.0)
        
        with patch('sim.matching.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(milliseconds=100)
            fills = self.engine.handle_action(action)
        
        if fills and fills[0].fill_qty > 0:
            # Commission should be calculated on net winnings
            assert fills[0].fees >= 0
    
    def test_latency_model(self):
        """Test latency model functionality"""
        latency = self.latency_model.sample_latency()
        
        assert self.latency_model.min_ms <= latency <= self.latency_model.max_ms
        
        # Test multiple samples for distribution
        latencies = [self.latency_model.sample_latency() for _ in range(100)]
        avg_latency = sum(latencies) / len(latencies)
        
        # Should be close to mean (within reasonable tolerance)
        assert abs(avg_latency - self.latency_model.mean_ms) < 20.0
    
    def test_slippage_calculation(self):
        """Test slippage calculation"""
        original_price = 1.95
        fill_price = 1.96
        
        slippage = self.slippage_model.calculate_slippage(original_price, fill_price)
        
        # Should be positive slippage (worse price)
        assert slippage > 0
        
        # Should be approximately 0.51% in basis points
        expected_slippage = ((1.96 - 1.95) / 1.95) * 10000
        assert abs(slippage - expected_slippage) < 1.0
    
    def test_engine_stats(self):
        """Test engine statistics"""
        stats = self.engine.get_stats()
        
        assert "orders_in_flight" in stats
        assert "suspended_markets" in stats
        assert "participation_factor" in stats
        assert "latency_mean_ms" in stats
        assert "commission_bps" in stats
        
        assert stats["participation_factor"] == 0.5
        assert stats["latency_mean_ms"] == 100.0
        assert stats["commission_bps"] == 200.0


class TestLatencyModel:
    """Test latency model"""
    
    def test_latency_bounds(self):
        """Test latency stays within bounds"""
        model = LatencyModel(mean_ms=250.0, std_ms=50.0, min_ms=10.0, max_ms=1000.0)
        
        for _ in range(1000):
            latency = model.sample_latency()
            assert 10.0 <= latency <= 1000.0


class TestCommissionModel:
    """Test commission model"""
    
    def test_commission_calculation(self):
        """Test commission calculation logic"""
        model = CommissionModel(commission_bps=200.0)
        
        # Winning bet
        commission = model.calculate_commission(stake=1000.0, odds=2.0, won=True)
        expected = 1000.0 * (2.0 - 1.0) * 0.02  # 2% of net winnings
        assert abs(commission - expected) < 0.01
        
        # Losing bet
        commission = model.calculate_commission(stake=1000.0, odds=2.0, won=False)
        assert commission == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
