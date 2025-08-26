# Purpose: SIM metrics calculation and KPI reporting
# Author: WicketWise AI, Last Modified: 2024

import math
import statistics
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

try:
    from .config import KPIResults
    from .state import MatchState, MarketState
    from .strategy import StrategyAction, FillEvent, AccountState, OrderSide
except ImportError:
    from config import KPIResults
    from state import MatchState, MarketState
    from strategy import StrategyAction, FillEvent, AccountState, OrderSide


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot"""
    timestamp: str
    balance: float
    pnl: float
    exposure: float
    drawdown: float
    num_positions: int


@dataclass
class TradeRecord:
    """Individual trade record for analysis"""
    entry_time: str
    exit_time: Optional[str]
    market_id: str
    selection_id: str
    side: OrderSide
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float
    duration_seconds: float
    edge: Optional[float]  # Theoretical edge at entry


class KPICalculator:
    """Calculator for key performance indicators"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_return
    
    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(returns)
        negative_returns = [r for r in returns if r < risk_free_rate]
        
        if not negative_returns:
            return float('inf') if mean_return > risk_free_rate else 0.0
        
        downside_deviation = math.sqrt(statistics.mean([(r - risk_free_rate) ** 2 for r in negative_returns]))
        
        if downside_deviation == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / downside_deviation
    
    @staticmethod
    def calculate_max_drawdown(balance_history: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not balance_history:
            return 0.0
        
        peak = balance_history[0]
        max_dd = 0.0
        
        for balance in balance_history:
            if balance > peak:
                peak = balance
            
            drawdown = (peak - balance) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100.0  # Return as percentage
    
    @staticmethod
    def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / (max_drawdown / 100.0)
    
    @staticmethod
    def calculate_hit_rate(trades: List[TradeRecord]) -> float:
        """Calculate hit rate (percentage of winning trades)"""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        return winning_trades / len(trades)
    
    @staticmethod
    def calculate_average_edge(trades: List[TradeRecord]) -> float:
        """Calculate average theoretical edge"""
        edges = [trade.edge for trade in trades if trade.edge is not None]
        return statistics.mean(edges) if edges else 0.0
    
    @staticmethod
    def calculate_slippage_bps(fills: List[Tuple[FillEvent, StrategyAction]]) -> float:
        """Calculate average slippage in basis points"""
        if not fills:
            return 0.0
        
        total_slippage = sum(fill.slippage for fill, _ in fills)
        return total_slippage / len(fills)
    
    @staticmethod
    def calculate_fill_rate(actions: List[StrategyAction], fills: List[FillEvent]) -> float:
        """Calculate fill rate (percentage of orders that got filled)"""
        if not actions:
            return 0.0
        
        filled_orders = set(fill.client_order_id for fill in fills if fill.fill_qty > 0)
        return len(filled_orders) / len(actions)


class SimMetrics:
    """
    Simulation metrics collector and calculator
    
    Tracks performance throughout simulation and calculates final KPIs
    """
    
    def __init__(self):
        # Time series data
        self.balance_history: List[float] = []
        self.pnl_history: List[float] = []
        self.exposure_history: List[float] = []
        self.timestamp_history: List[str] = []
        
        # Trade tracking
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}  # key: market_id:selection_id
        
        # Action and fill tracking
        self.actions: List[StrategyAction] = []
        self.fills: List[Tuple[FillEvent, StrategyAction]] = []
        
        # Performance snapshots
        self.snapshots: List[PerformanceSnapshot] = []
        
        # Phase-specific metrics
        self.phase_metrics: Dict[str, Dict[str, float]] = {
            "powerplay": {"pnl": 0.0, "trades": 0},
            "middle": {"pnl": 0.0, "trades": 0},
            "death": {"pnl": 0.0, "trades": 0}
        }
        
        # Initial values
        self.initial_balance = 0.0
        self.start_time: Optional[datetime] = None
    
    def initialize(self, initial_balance: float):
        """Initialize metrics tracking"""
        self.initial_balance = initial_balance
        self.start_time = datetime.now()
        
        # Record initial state
        self.balance_history.append(initial_balance)
        self.pnl_history.append(0.0)
        self.exposure_history.append(0.0)
        self.timestamp_history.append(self.start_time.isoformat())
    
    def update_tick(self, match_state: MatchState, market_state: MarketState, account_state: AccountState):
        """Update metrics on each simulation tick"""
        
        current_time = datetime.now().isoformat()
        current_balance = account_state.total_balance()
        current_pnl = current_balance - self.initial_balance
        current_exposure = account_state.exposure
        
        # Update time series
        self.balance_history.append(current_balance)
        self.pnl_history.append(current_pnl)
        self.exposure_history.append(current_exposure)
        self.timestamp_history.append(current_time)
        
        # Calculate current drawdown
        peak_balance = max(self.balance_history)
        current_drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0.0
        
        # Create performance snapshot every 100 ticks
        if len(self.balance_history) % 100 == 0:
            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                balance=current_balance,
                pnl=current_pnl,
                exposure=current_exposure,
                drawdown=current_drawdown * 100.0,
                num_positions=len([pos for pos in account_state.positions.values() if pos.net_position() != 0])
            )
            self.snapshots.append(snapshot)
    
    def update_action(self, action: StrategyAction):
        """Record strategy action"""
        self.actions.append(action)
    
    def update_fill(self, fill: FillEvent, action: StrategyAction, account_state: AccountState):
        """Update metrics when fill occurs"""
        
        # Record fill
        self.fills.append((fill, action))
        
        # Update trade tracking
        self._update_trade_tracking(fill, action, account_state)
    
    def _update_trade_tracking(self, fill: FillEvent, action: StrategyAction, account_state: AccountState):
        """Update individual trade tracking"""
        
        position_key = f"{action.market_id}:{action.selection_id}"
        
        if fill.fill_qty <= 0:
            return
        
        # Get current position
        position = account_state.get_position(action.market_id, action.selection_id)
        
        # Check if this opens or closes a position
        if position_key not in self.open_positions:
            # Opening new position
            trade = TradeRecord(
                entry_time=fill.ts,
                exit_time=None,
                market_id=action.market_id,
                selection_id=action.selection_id,
                side=action.side,
                entry_price=fill.avg_price,
                exit_price=None,
                size=fill.fill_qty,
                pnl=0.0,
                duration_seconds=0.0,
                edge=self._estimate_edge(action)  # Would integrate with actual model
            )
            self.open_positions[position_key] = trade
        
        else:
            # Potentially closing or adding to position
            existing_trade = self.open_positions[position_key]
            
            # Check if position is now closed
            if position.net_position() == 0:
                # Position closed
                existing_trade.exit_time = fill.ts
                existing_trade.exit_price = fill.avg_price
                
                # Calculate P&L and duration
                entry_time = datetime.fromisoformat(existing_trade.entry_time.replace('Z', '+00:00'))
                exit_time = datetime.fromisoformat(fill.ts.replace('Z', '+00:00'))
                existing_trade.duration_seconds = (exit_time - entry_time).total_seconds()
                
                # Calculate realized P&L (simplified)
                if existing_trade.side == OrderSide.BACK:
                    existing_trade.pnl = existing_trade.size * (existing_trade.exit_price - existing_trade.entry_price)
                else:
                    existing_trade.pnl = existing_trade.size * (existing_trade.entry_price - existing_trade.exit_price)
                
                # Move to completed trades
                self.trades.append(existing_trade)
                del self.open_positions[position_key]
    
    def _estimate_edge(self, action: StrategyAction) -> Optional[float]:
        """Estimate theoretical edge for trade (simplified)"""
        # This would integrate with actual model predictions
        # For now, return a mock edge based on odds
        if action.price > 1.5:
            return 0.02  # 2% edge for longer odds
        else:
            return 0.01  # 1% edge for shorter odds
    
    def calculate_final_kpis(self, account_state: AccountState, initial_bankroll: float) -> KPIResults:
        """Calculate final KPIs for simulation"""
        
        if not self.balance_history:
            return KPIResults()
        
        # Basic P&L metrics
        final_balance = account_state.total_balance()
        total_pnl = final_balance - initial_bankroll
        realized_pnl = account_state.realized_pnl
        unrealized_pnl = account_state.unrealized_pnl
        
        # Calculate returns for ratio calculations
        returns = []
        for i in range(1, len(self.balance_history)):
            if self.balance_history[i-1] > 0:
                ret = (self.balance_history[i] - self.balance_history[i-1]) / self.balance_history[i-1]
                returns.append(ret)
        
        # Risk-adjusted metrics
        sharpe = KPICalculator.calculate_sharpe_ratio(returns)
        sortino = KPICalculator.calculate_sortino_ratio(returns)
        max_drawdown = KPICalculator.calculate_max_drawdown(self.balance_history)
        calmar = KPICalculator.calculate_calmar_ratio(total_pnl / initial_bankroll * 100, max_drawdown)
        
        # Trading metrics
        hit_rate = KPICalculator.calculate_hit_rate(self.trades)
        avg_edge = KPICalculator.calculate_average_edge(self.trades)
        slippage_bps = KPICalculator.calculate_slippage_bps(self.fills)
        fill_rate = KPICalculator.calculate_fill_rate(self.actions, [fill for fill, _ in self.fills])
        
        # Exposure metrics
        peak_exposure = max(self.exposure_history) if self.exposure_history else 0.0
        exposure_peak_pct = (peak_exposure / initial_bankroll) * 100.0 if initial_bankroll > 0 else 0.0
        
        # Volume metrics
        total_turnover = sum(
            fill.fill_qty * fill.avg_price 
            for fill, action in self.fills 
            if action.side == OrderSide.BACK
        )
        total_turnover += sum(
            fill.fill_qty * (fill.avg_price - 1.0)
            for fill, action in self.fills 
            if action.side == OrderSide.LAY
        )
        
        return KPIResults(
            pnl_total=total_pnl,
            pnl_realized=realized_pnl,
            pnl_unrealized=unrealized_pnl,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_drawdown,
            calmar=calmar,
            hit_rate=hit_rate,
            avg_edge=avg_edge,
            slippage_bps=slippage_bps,
            fill_rate=fill_rate,
            exposure_peak_pct=exposure_peak_pct,
            turnover=total_turnover,
            num_trades=len(self.trades)
        )
    
    def get_phase_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by match phase"""
        return self.phase_metrics.copy()
    
    def get_performance_snapshots(self) -> List[PerformanceSnapshot]:
        """Get performance snapshots for charting"""
        return self.snapshots.copy()
    
    def get_trade_analysis(self) -> Dict[str, Any]:
        """Get detailed trade analysis"""
        if not self.trades:
            return {}
        
        # Winning vs losing trades
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        # Average metrics
        avg_win = statistics.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = statistics.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        avg_duration = statistics.mean([t.duration_seconds for t in self.trades])
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "hit_rate": len(winning_trades) / len(self.trades),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_duration_seconds": avg_duration,
            "largest_win": max([t.pnl for t in self.trades]),
            "largest_loss": min([t.pnl for t in self.trades])
        }
    
    def export_time_series(self) -> Dict[str, List]:
        """Export time series data for analysis"""
        return {
            "timestamps": self.timestamp_history.copy(),
            "balance": self.balance_history.copy(),
            "pnl": self.pnl_history.copy(),
            "exposure": self.exposure_history.copy()
        }
