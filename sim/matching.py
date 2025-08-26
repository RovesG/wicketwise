# Purpose: SIM LOB matching engine with price-time priority
# Author: WicketWise AI, Last Modified: 2024

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import random
import math
from enum import Enum

try:
    from .strategy import StrategyAction, FillEvent, OrderSide, OrderType
    from .state import MarketSnapshot, SelectionBook, LadderLevel, MarketStatus
except ImportError:
    from strategy import StrategyAction, FillEvent, OrderSide, OrderType
    from state import MarketSnapshot, SelectionBook, LadderLevel, MarketStatus


class FillReason(Enum):
    """Reasons for order fills"""
    FULL_FILL = "full_fill"
    PARTIAL_QUEUE = "partial_queue"
    PARTIAL_SIZE = "partial_size"
    NO_FILL_SUSPENDED = "no_fill_suspended"
    NO_FILL_NO_LIQUIDITY = "no_fill_no_liquidity"
    NO_FILL_PRICE_MOVED = "no_fill_price_moved"
    CANCELLED = "cancelled"
    IOC_EXPIRED = "ioc_expired"


@dataclass
class OrderInFlight:
    """Order that is in flight (subject to latency)"""
    action: StrategyAction
    submit_time: datetime
    effective_time: datetime
    original_market_state: MarketSnapshot


@dataclass
class LatencyModel:
    """Latency model configuration"""
    mean_ms: float = 250.0
    std_ms: float = 50.0
    min_ms: float = 10.0
    max_ms: float = 2000.0
    
    def sample_latency(self) -> float:
        """Sample latency in milliseconds"""
        latency = random.gauss(self.mean_ms, self.std_ms)
        return max(self.min_ms, min(self.max_ms, latency))


@dataclass
class SlippageModel:
    """Slippage model configuration"""
    model_type: str = "lob_queue"  # "lob_queue", "linear", "none"
    price_impact_bps: float = 5.0  # Basis points per unit size
    
    def calculate_slippage(self, original_price: float, fill_price: float) -> float:
        """Calculate slippage in basis points"""
        if original_price <= 0:
            return 0.0
        return ((fill_price - original_price) / original_price) * 10000


@dataclass
class CommissionModel:
    """Commission model configuration"""
    commission_bps: float = 200.0  # 2% commission
    discount_rate: float = 0.0  # Volume discount
    
    def calculate_commission(self, stake: float, odds: float, won: bool) -> float:
        """Calculate commission on net winnings"""
        if not won:
            return 0.0  # No commission on losing bets
        
        net_winnings = stake * (odds - 1.0)
        commission_rate = self.commission_bps / 10000.0
        return net_winnings * commission_rate * (1.0 - self.discount_rate)


class MatchingEngine:
    """
    LOB matching engine with price-time priority, partial fills, and latency simulation
    """
    
    def __init__(self, latency_model: Optional[LatencyModel] = None,
                 slippage_model: Optional[SlippageModel] = None,
                 commission_model: Optional[CommissionModel] = None,
                 participation_factor: float = 0.1):
        self.latency_model = latency_model or LatencyModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_model = commission_model or CommissionModel()
        self.participation_factor = participation_factor
        
        # State
        self.orders_in_flight: List[OrderInFlight] = []
        self.current_market_state: Dict[str, MarketSnapshot] = {}
        self.suspended_markets: Dict[str, datetime] = {}
        self.order_book_queue: Dict[str, List[StrategyAction]] = {}  # Queued orders by market
        
    def update_market_state(self, snapshot: MarketSnapshot):
        """Update current market state"""
        self.current_market_state[snapshot.market_id] = snapshot
        
        # Handle suspension state changes
        if snapshot.status == MarketStatus.SUSPENDED:
            if snapshot.market_id not in self.suspended_markets:
                self.suspended_markets[snapshot.market_id] = datetime.now()
        elif snapshot.status == MarketStatus.OPEN:
            self.suspended_markets.pop(snapshot.market_id, None)
    
    def handle_action(self, action: StrategyAction) -> List[FillEvent]:
        """
        Handle strategy action with latency and matching logic
        
        Args:
            action: Strategy action to process
            
        Returns:
            List of fill events (may be empty if no immediate fill)
        """
        current_time = datetime.now()
        
        # Apply latency
        latency_ms = self.latency_model.sample_latency()
        effective_time = current_time + timedelta(milliseconds=latency_ms)
        
        # Store current market state for slippage calculation
        current_snapshot = self.current_market_state.get(action.market_id)
        if not current_snapshot:
            return []
        
        # Add to in-flight orders
        order_in_flight = OrderInFlight(
            action=action,
            submit_time=current_time,
            effective_time=effective_time,
            original_market_state=current_snapshot
        )
        self.orders_in_flight.append(order_in_flight)
        
        # Process any orders that have reached their effective time
        return self._process_ready_orders()
    
    def _process_ready_orders(self) -> List[FillEvent]:
        """Process orders that have reached their effective time"""
        fills = []
        current_time = datetime.now()
        
        # Find orders ready to process
        ready_orders = []
        remaining_orders = []
        
        for order in self.orders_in_flight:
            if current_time >= order.effective_time:
                ready_orders.append(order)
            else:
                remaining_orders.append(order)
        
        self.orders_in_flight = remaining_orders
        
        # Process each ready order
        for order in ready_orders:
            fill_events = self._execute_order(order)
            fills.extend(fill_events)
        
        return fills
    
    def _execute_order(self, order: OrderInFlight) -> List[FillEvent]:
        """Execute a single order against current market state"""
        action = order.action
        
        # Check if market is suspended
        if action.market_id in self.suspended_markets:
            if action.type == OrderType.IOC:
                # IOC orders are rejected if market suspended
                return [FillEvent(
                    client_order_id=action.client_order_id,
                    fill_qty=0.0,
                    avg_price=0.0,
                    fees=0.0,
                    slippage=0.0,
                    reason=FillReason.NO_FILL_SUSPENDED.value,
                    ts=datetime.now().isoformat(),
                    remaining_qty=action.size
                )]
            else:
                # LIMIT orders are queued
                self._queue_order(action)
                return []
        
        # Get current market snapshot
        current_snapshot = self.current_market_state.get(action.market_id)
        if not current_snapshot:
            return [FillEvent(
                client_order_id=action.client_order_id,
                fill_qty=0.0,
                avg_price=0.0,
                fees=0.0,
                slippage=0.0,
                reason=FillReason.NO_FILL_NO_LIQUIDITY.value,
                ts=datetime.now().isoformat(),
                remaining_qty=action.size
            )]
        
        # Find selection
        selection = current_snapshot.get_selection(action.selection_id)
        if not selection:
            return [FillEvent(
                client_order_id=action.client_order_id,
                fill_qty=0.0,
                avg_price=0.0,
                fees=0.0,
                slippage=0.0,
                reason=FillReason.NO_FILL_NO_LIQUIDITY.value,
                ts=datetime.now().isoformat(),
                remaining_qty=action.size
            )]
        
        # Execute based on order type
        if action.type == OrderType.LIMIT:
            return self._execute_limit_order(action, selection, order.original_market_state)
        elif action.type == OrderType.IOC:
            return self._execute_ioc_order(action, selection, order.original_market_state)
        elif action.type == OrderType.CANCEL:
            return self._execute_cancel_order(action)
        else:
            return []
    
    def _execute_limit_order(self, action: StrategyAction, selection: SelectionBook, 
                           original_snapshot: MarketSnapshot) -> List[FillEvent]:
        """Execute limit order with price-time priority"""
        
        # Determine which side of the book to match against
        if action.side == OrderSide.BACK:
            # Backing - match against lay side
            opposite_levels = selection.lay
        else:
            # Laying - match against back side  
            opposite_levels = selection.back
        
        if not opposite_levels:
            # No liquidity on opposite side
            if action.post_only:
                # Add to book (simulated)
                return []
            else:
                return [FillEvent(
                    client_order_id=action.client_order_id,
                    fill_qty=0.0,
                    avg_price=0.0,
                    fees=0.0,
                    slippage=0.0,
                    reason=FillReason.NO_FILL_NO_LIQUIDITY.value,
                    ts=datetime.now().isoformat(),
                    remaining_qty=action.size
                )]
        
        # Check if order can match at desired price
        best_opposite_price = opposite_levels[0].price
        
        if action.side == OrderSide.BACK and action.price < best_opposite_price:
            # Back order price too low
            return []
        elif action.side == OrderSide.LAY and action.price > best_opposite_price:
            # Lay order price too high
            return []
        
        # Calculate fill based on available liquidity and participation factor
        available_size = sum(level.size for level in opposite_levels 
                           if (action.side == OrderSide.BACK and level.price <= action.price) or
                              (action.side == OrderSide.LAY and level.price >= action.price))
        
        max_fill_size = available_size * self.participation_factor
        fill_size = min(action.size, max_fill_size)
        
        if fill_size <= 0:
            return [FillEvent(
                client_order_id=action.client_order_id,
                fill_qty=0.0,
                avg_price=0.0,
                fees=0.0,
                slippage=0.0,
                reason=FillReason.NO_FILL_NO_LIQUIDITY.value,
                ts=datetime.now().isoformat(),
                remaining_qty=action.size
            )]
        
        # Calculate average fill price (VWAP across levels)
        fill_price = self._calculate_vwap_price(opposite_levels, fill_size, action.side, action.price)
        
        # Calculate slippage
        original_selection = original_snapshot.get_selection(action.selection_id)
        original_price = action.price
        if original_selection:
            if action.side == OrderSide.BACK:
                original_price = original_selection.best_back_price() or action.price
            else:
                original_price = original_selection.best_lay_price() or action.price
        
        slippage = self.slippage_model.calculate_slippage(original_price, fill_price)
        
        # Calculate commission (simplified - assume win for now)
        commission = self.commission_model.calculate_commission(
            fill_size * fill_price, fill_price, True
        )
        
        # Determine fill reason
        if fill_size >= action.size:
            reason = FillReason.FULL_FILL.value
            remaining_qty = 0.0
        else:
            reason = FillReason.PARTIAL_QUEUE.value
            remaining_qty = action.size - fill_size
        
        return [FillEvent(
            client_order_id=action.client_order_id,
            fill_qty=fill_size,
            avg_price=fill_price,
            fees=commission,
            slippage=slippage,
            reason=reason,
            ts=datetime.now().isoformat(),
            remaining_qty=remaining_qty
        )]
    
    def _execute_ioc_order(self, action: StrategyAction, selection: SelectionBook,
                          original_snapshot: MarketSnapshot) -> List[FillEvent]:
        """Execute immediate-or-cancel order"""
        # IOC orders are executed immediately or cancelled
        fills = self._execute_limit_order(action, selection, original_snapshot)
        
        # If no fill or partial fill, mark remainder as expired
        if not fills or fills[0].remaining_qty > 0:
            if fills:
                # Update reason for partial fill
                fills[0].reason = FillReason.IOC_EXPIRED.value
            else:
                # No fill at all
                fills = [FillEvent(
                    client_order_id=action.client_order_id,
                    fill_qty=0.0,
                    avg_price=0.0,
                    fees=0.0,
                    slippage=0.0,
                    reason=FillReason.IOC_EXPIRED.value,
                    ts=datetime.now().isoformat(),
                    remaining_qty=action.size
                )]
        
        return fills
    
    def _execute_cancel_order(self, action: StrategyAction) -> List[FillEvent]:
        """Execute cancel order"""
        # Remove from queued orders (simplified)
        return [FillEvent(
            client_order_id=action.client_order_id,
            fill_qty=0.0,
            avg_price=0.0,
            fees=0.0,
            slippage=0.0,
            reason=FillReason.CANCELLED.value,
            ts=datetime.now().isoformat(),
            remaining_qty=0.0
        )]
    
    def _calculate_vwap_price(self, levels: List[LadderLevel], fill_size: float,
                             side: OrderSide, limit_price: float) -> float:
        """Calculate volume-weighted average price across ladder levels"""
        total_value = 0.0
        total_size = 0.0
        remaining_size = fill_size
        
        for level in levels:
            # Check price constraint
            if side == OrderSide.BACK and level.price > limit_price:
                break
            elif side == OrderSide.LAY and level.price < limit_price:
                break
            
            # Calculate size to take from this level
            available_at_level = level.size * self.participation_factor
            size_from_level = min(remaining_size, available_at_level)
            
            if size_from_level <= 0:
                break
            
            total_value += size_from_level * level.price
            total_size += size_from_level
            remaining_size -= size_from_level
            
            if remaining_size <= 0:
                break
        
        return total_value / total_size if total_size > 0 else limit_price
    
    def _queue_order(self, action: StrategyAction):
        """Queue order for when market reopens"""
        if action.market_id not in self.order_book_queue:
            self.order_book_queue[action.market_id] = []
        self.order_book_queue[action.market_id].append(action)
    
    def process_market_reopen(self, market_id: str) -> List[FillEvent]:
        """Process queued orders when market reopens"""
        fills = []
        
        if market_id in self.order_book_queue:
            queued_orders = self.order_book_queue[market_id]
            self.order_book_queue[market_id] = []
            
            # Process queued orders in FIFO order
            for action in queued_orders:
                current_snapshot = self.current_market_state.get(market_id)
                if current_snapshot:
                    selection = current_snapshot.get_selection(action.selection_id)
                    if selection:
                        order_fills = self._execute_limit_order(action, selection, current_snapshot)
                        fills.extend(order_fills)
        
        return fills
    
    def get_stats(self) -> Dict[str, Any]:
        """Get matching engine statistics"""
        return {
            "orders_in_flight": len(self.orders_in_flight),
            "suspended_markets": len(self.suspended_markets),
            "queued_orders": sum(len(orders) for orders in self.order_book_queue.values()),
            "participation_factor": self.participation_factor,
            "latency_mean_ms": self.latency_model.mean_ms,
            "commission_bps": self.commission_model.commission_bps
        }
