# Purpose: SIM strategy protocol and built-in strategies
# Author: WicketWise AI, Last Modified: 2024

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
from enum import Enum
import json
import math
import random


class OrderSide(Enum):
    """Order side enumeration"""
    BACK = "back"
    LAY = "lay"


class OrderType(Enum):
    """Order type enumeration"""
    LIMIT = "limit"
    IOC = "ioc"  # Immediate or Cancel
    CANCEL = "cancel"
    REPLACE = "replace"


@dataclass
class StrategyAction:
    """Strategy action/order request"""
    ts: str
    type: OrderType
    side: OrderSide
    market_id: str
    selection_id: str
    price: float
    size: float
    client_order_id: str
    reference_order_id: Optional[str] = None  # For cancel/replace
    post_only: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "type": self.type.value,
            "side": self.side.value,
            "market_id": self.market_id,
            "selection_id": self.selection_id,
            "price": self.price,
            "size": self.size,
            "client_order_id": self.client_order_id,
            "reference_order_id": self.reference_order_id,
            "post_only": self.post_only
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyAction':
        data = data.copy()
        data["type"] = OrderType(data["type"])
        data["side"] = OrderSide(data["side"])
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyAction':
        return cls.from_dict(json.loads(json_str))


@dataclass
class FillEvent:
    """Order fill event"""
    client_order_id: str
    fill_qty: float
    avg_price: float
    fees: float
    slippage: float
    reason: str
    ts: str
    remaining_qty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "fill_qty": self.fill_qty,
            "avg_price": self.avg_price,
            "fees": self.fees,
            "slippage": self.slippage,
            "reason": self.reason,
            "ts": self.ts,
            "remaining_qty": self.remaining_qty
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FillEvent':
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class Position:
    """Position in a market selection"""
    market_id: str
    selection_id: str
    back_stake: float = 0.0
    lay_liability: float = 0.0
    matched_back: float = 0.0
    matched_lay: float = 0.0
    avg_back_price: float = 0.0
    avg_lay_price: float = 0.0
    
    def net_position(self) -> float:
        """Calculate net position (positive = long, negative = short)"""
        return self.matched_back - self.matched_lay
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price"""
        net_pos = self.net_position()
        if net_pos == 0:
            return 0.0
        
        if net_pos > 0:  # Long position
            return net_pos * (current_price - self.avg_back_price)
        else:  # Short position
            return abs(net_pos) * (self.avg_lay_price - current_price)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "selection_id": self.selection_id,
            "back_stake": self.back_stake,
            "lay_liability": self.lay_liability,
            "matched_back": self.matched_back,
            "matched_lay": self.matched_lay,
            "avg_back_price": self.avg_back_price,
            "avg_lay_price": self.avg_lay_price
        }


@dataclass
class AccountState:
    """Account state for P&L tracking"""
    cash: float = 100000.0
    exposure: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    margin_util: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)  # key: market_id:selection_id
    
    def get_position(self, market_id: str, selection_id: str) -> Position:
        """Get or create position for market/selection"""
        key = f"{market_id}:{selection_id}"
        if key not in self.positions:
            self.positions[key] = Position(market_id, selection_id)
        return self.positions[key]
    
    def update_from_fill(self, fill: FillEvent, action: StrategyAction):
        """Update account state from fill event"""
        position = self.get_position(action.market_id, action.selection_id)
        
        if action.side == OrderSide.BACK:
            # Update back position
            total_stake = position.matched_back * position.avg_back_price + fill.fill_qty * fill.avg_price
            position.matched_back += fill.fill_qty
            if position.matched_back > 0:
                position.avg_back_price = total_stake / position.matched_back
            position.back_stake += fill.fill_qty * fill.avg_price
        else:  # LAY
            # Update lay position  
            total_liability = position.matched_lay * position.avg_lay_price + fill.fill_qty * fill.avg_price
            position.matched_lay += fill.fill_qty
            if position.matched_lay > 0:
                position.avg_lay_price = total_liability / position.matched_lay
            position.lay_liability += fill.fill_qty * (fill.avg_price - 1.0)
        
        # Update cash (reduce by stake/liability)
        if action.side == OrderSide.BACK:
            self.cash -= fill.fill_qty * fill.avg_price
        else:
            self.cash -= fill.fill_qty * (fill.avg_price - 1.0)
        
        # Update realized P&L (subtract fees)
        self.realized_pnl -= fill.fees
        
        # Recalculate exposure and unrealized P&L
        self._recalculate_exposure()
    
    def _recalculate_exposure(self):
        """Recalculate total exposure and unrealized P&L"""
        total_exposure = 0.0
        total_unrealized = 0.0
        
        for position in self.positions.values():
            # Exposure is maximum potential loss
            back_exposure = position.back_stake
            lay_exposure = position.lay_liability
            total_exposure += max(back_exposure, lay_exposure)
            
            # Unrealized P&L needs current market prices (would be updated separately)
        
        self.exposure = total_exposure
        self.margin_util = self.exposure / self.cash if self.cash > 0 else 0.0
    
    def total_balance(self) -> float:
        """Total account balance including unrealized P&L"""
        return self.cash + self.realized_pnl + self.unrealized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash": self.cash,
            "exposure": self.exposure,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "margin_util": self.margin_util,
            "positions": {k: v.to_dict() for k, v in self.positions.items()}
        }


class Strategy(Protocol):
    """Strategy protocol for SIM strategies"""
    
    @abstractmethod
    def on_tick(self, match_state: 'MatchState', market_state: 'MarketState', 
                account_state: AccountState) -> List[StrategyAction]:
        """
        Called on each market/match state update
        
        Args:
            match_state: Current match state
            market_state: Current market state
            account_state: Current account state
            
        Returns:
            List of strategy actions to execute
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        pass


class EdgeKellyStrategy:
    """
    Edge-based Kelly criterion strategy
    
    Bets when model_prob - implied_prob >= threshold using fractional Kelly sizing
    """
    
    def __init__(self, edge_threshold: float = 0.02, kelly_fraction: float = 0.25,
                 max_stake_pct: float = 0.05, min_odds: float = 1.1, max_odds: float = 10.0):
        self.edge_threshold = edge_threshold
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.order_counter = 0
    
    def on_tick(self, match_state: 'MatchState', market_state: 'MarketState', 
                account_state: AccountState) -> List[StrategyAction]:
        """Generate actions based on edge detection"""
        actions = []
        
        # Only trade on match odds for now
        if not market_state.is_market_open("match_odds"):
            return actions
        
        snapshot = market_state.get_snapshot("match_odds")
        if not snapshot or not snapshot.selections:
            return actions
        
        # Get model probability (mock for now - would integrate with actual models)
        model_prob = self._get_model_probability(match_state)
        if model_prob is None:
            return actions
        
        # Check each selection for edge
        for selection in snapshot.selections:
            back_price = selection.best_back_price()
            if not back_price or back_price < self.min_odds or back_price > self.max_odds:
                continue
            
            implied_prob = 1.0 / back_price
            edge = model_prob - implied_prob
            
            if edge >= self.edge_threshold:
                # Calculate Kelly stake
                kelly_stake = self._calculate_kelly_stake(
                    edge, back_price, account_state.total_balance()
                )
                
                if kelly_stake > 0:
                    self.order_counter += 1
                    action = StrategyAction(
                        ts=datetime.now().isoformat(),
                        type=OrderType.LIMIT,
                        side=OrderSide.BACK,
                        market_id="match_odds",
                        selection_id=selection.selection_id,
                        price=back_price,
                        size=kelly_stake,
                        client_order_id=f"edge_kelly_{self.order_counter}",
                        post_only=True
                    )
                    actions.append(action)
        
        return actions
    
    def _get_model_probability(self, match_state: 'MatchState') -> Optional[float]:
        """Get model probability for current match state (enhanced mock implementation)"""
        if not match_state.current_event:
            return None
        
        # Enhanced probability model that generates more betting opportunities
        if match_state.innings == 1:
            # First innings - create more dynamic probabilities
            over = match_state.over
            wickets = match_state.wickets
            
            # Base probability varies with match situation
            if over < 6:  # Powerplay
                base_prob = 0.52 + (match_state.score - 50) * 0.002
            elif over < 15:  # Middle overs
                base_prob = 0.48 + (match_state.score - 120) * 0.001
            else:  # Death overs
                base_prob = 0.55 + (match_state.score - 180) * 0.0015
            
            # Adjust for wickets
            base_prob -= wickets * 0.03
            
        else:
            # Second innings - more aggressive probability swings
            if match_state.target and match_state.balls_remaining > 0:
                runs_needed = match_state.target - match_state.score
                required_rate = runs_needed * 6.0 / match_state.balls_remaining
                current_rate = match_state.current_rate if hasattr(match_state, 'current_rate') else 6.0
                
                # Enhanced probability model with more volatility
                rate_diff = current_rate - required_rate
                base_prob = 0.5 + rate_diff * 0.04  # Increased sensitivity
                
                # Add pressure factor for close games
                if abs(runs_needed) < 30:
                    pressure_factor = 0.1 * random.choice([-1, 1])  # Random pressure swings
                    base_prob += pressure_factor
                    
            else:
                base_prob = 0.5
        
        # Add more significant randomness to create betting opportunities
        noise = random.gauss(0, 0.08)  # Increased noise for more edge opportunities
        prob = max(0.1, min(0.9, base_prob + noise))
        
        return prob
    
    def _calculate_kelly_stake(self, edge: float, odds: float, bankroll: float) -> float:
        """Calculate Kelly criterion stake size"""
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = win probability, q = 1 - p
        b = odds - 1.0
        p = edge + (1.0 / odds)  # Approximate win probability
        q = 1.0 - p
        
        if b <= 0 or p <= 0:
            return 0.0
        
        kelly_fraction_optimal = (b * p - q) / b
        
        # Apply fractional Kelly and maximum stake constraints
        kelly_stake = kelly_fraction_optimal * self.kelly_fraction * bankroll
        max_stake = self.max_stake_pct * bankroll
        
        return min(kelly_stake, max_stake)
    
    def get_name(self) -> str:
        return "edge_kelly_v3"
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "edge_threshold": self.edge_threshold,
            "kelly_fraction": self.kelly_fraction,
            "max_stake_pct": self.max_stake_pct,
            "min_odds": self.min_odds,
            "max_odds": self.max_odds
        }


class MeanRevertLOBStrategy:
    """
    Mean reversion strategy that fades microstructure dislocations
    """
    
    def __init__(self, revert_threshold: float = 0.02, max_position: float = 1000.0,
                 hold_time_seconds: float = 30.0):
        self.revert_threshold = revert_threshold
        self.max_position = max_position
        self.hold_time_seconds = hold_time_seconds
        self.order_counter = 0
        self.last_mid_price = None
    
    def on_tick(self, match_state: 'MatchState', market_state: 'MarketState', 
                account_state: AccountState) -> List[StrategyAction]:
        """Generate mean reversion actions"""
        actions = []
        
        if not market_state.is_market_open("match_odds"):
            return actions
        
        snapshot = market_state.get_snapshot("match_odds")
        if not snapshot or len(snapshot.selections) < 2:
            return actions
        
        # Look for price dislocations
        for selection in snapshot.selections:
            mid_price = selection.mid_price()
            if mid_price is None:
                continue
            
            # Check for mean reversion opportunity
            if self.last_mid_price is not None:
                price_change = (mid_price - self.last_mid_price) / self.last_mid_price
                
                if abs(price_change) > self.revert_threshold:
                    # Fade the move
                    if price_change > 0:  # Price moved up, lay it
                        lay_price = selection.best_lay_price()
                        if lay_price:
                            self.order_counter += 1
                            action = StrategyAction(
                                ts=datetime.now().isoformat(),
                                type=OrderType.LIMIT,
                                side=OrderSide.LAY,
                                market_id="match_odds",
                                selection_id=selection.selection_id,
                                price=lay_price,
                                size=min(self.max_position, 500.0),
                                client_order_id=f"mean_revert_{self.order_counter}"
                            )
                            actions.append(action)
                    
                    elif price_change < 0:  # Price moved down, back it
                        back_price = selection.best_back_price()
                        if back_price:
                            self.order_counter += 1
                            action = StrategyAction(
                                ts=datetime.now().isoformat(),
                                type=OrderType.LIMIT,
                                side=OrderSide.BACK,
                                market_id="match_odds",
                                selection_id=selection.selection_id,
                                price=back_price,
                                size=min(self.max_position, 500.0),
                                client_order_id=f"mean_revert_{self.order_counter}"
                            )
                            actions.append(action)
            
            self.last_mid_price = mid_price
        
        return actions
    
    def get_name(self) -> str:
        return "mean_revert_lob"
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "revert_threshold": self.revert_threshold,
            "max_position": self.max_position,
            "hold_time_seconds": self.hold_time_seconds
        }


class MomentumFollowStrategy:
    """
    Momentum following strategy that adds on price trends and cricket events
    """
    
    def __init__(self, momentum_threshold: float = 0.01, event_multiplier: float = 1.5,
                 max_stake: float = 1000.0):
        self.momentum_threshold = momentum_threshold
        self.event_multiplier = event_multiplier
        self.max_stake = max_stake
        self.order_counter = 0
        self.price_history = []
    
    def on_tick(self, match_state: 'MatchState', market_state: 'MarketState', 
                account_state: AccountState) -> List[StrategyAction]:
        """Generate momentum following actions"""
        actions = []
        
        if not market_state.is_market_open("match_odds"):
            return actions
        
        # Check for significant cricket events
        event_multiplier = 1.0
        if match_state.current_event:
            if match_state.current_event.runs_batter >= 4:  # Boundary
                event_multiplier = self.event_multiplier
            elif match_state.current_event.wicket.is_wicket:  # Wicket
                event_multiplier = self.event_multiplier
        
        snapshot = market_state.get_snapshot("match_odds")
        if not snapshot:
            return actions
        
        # Track price momentum
        for selection in snapshot.selections:
            mid_price = selection.mid_price()
            if mid_price is None:
                continue
            
            self.price_history.append(mid_price)
            if len(self.price_history) > 10:  # Keep last 10 prices
                self.price_history.pop(0)
            
            if len(self.price_history) >= 3:
                # Calculate momentum
                recent_change = (self.price_history[-1] - self.price_history[-3]) / self.price_history[-3]
                
                if abs(recent_change) > self.momentum_threshold:
                    stake = min(self.max_stake * event_multiplier, account_state.cash * 0.02)
                    
                    if recent_change > 0:  # Upward momentum, back it
                        back_price = selection.best_back_price()
                        if back_price and stake > 0:
                            self.order_counter += 1
                            action = StrategyAction(
                                ts=datetime.now().isoformat(),
                                type=OrderType.LIMIT,
                                side=OrderSide.BACK,
                                market_id="match_odds",
                                selection_id=selection.selection_id,
                                price=back_price,
                                size=stake,
                                client_order_id=f"momentum_{self.order_counter}"
                            )
                            actions.append(action)
        
        return actions
    
    def get_name(self) -> str:
        return "momentum_follow"
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "momentum_threshold": self.momentum_threshold,
            "event_multiplier": self.event_multiplier,
            "max_stake": self.max_stake
        }


# Strategy factory
def create_strategy(name: str, params: Dict[str, Any]) -> Strategy:
    """Create strategy instance from name and parameters"""
    if name == "edge_kelly_v3":
        return EdgeKellyStrategy(**params)
    elif name == "mean_revert_lob":
        return MeanRevertLOBStrategy(**params)
    elif name == "momentum_follow":
        return MomentumFollowStrategy(**params)
    else:
        raise ValueError(f"Unknown strategy: {name}")


# Import types for type hints
try:
    from .state import MatchState, MarketState
except ImportError:
    from state import MatchState, MarketState
