# Purpose: SIM state management - MatchState, MarketState, Events
# Author: WicketWise AI, Last Modified: 2024

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json


class MarketStatus(Enum):
    """Market status enumeration"""
    OPEN = "open"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class WicketKind(Enum):
    """Types of wickets in cricket"""
    BOWLED = "bowled"
    CAUGHT = "caught"
    LBW = "lbw"
    STUMPED = "stumped"
    RUN_OUT = "run_out"
    HIT_WICKET = "hit_wicket"
    HANDLED_BALL = "handled_ball"
    OBSTRUCTING_FIELD = "obstructing_field"
    TIMED_OUT = "timed_out"


class MatchPhase(Enum):
    """Cricket match phases"""
    POWERPLAY = "powerplay"
    MIDDLE = "middle"
    DEATH = "death"
    FINISHED = "finished"


@dataclass
class WicketInfo:
    """Wicket event information"""
    is_wicket: bool = False
    kind: Optional[WicketKind] = None
    player_out: Optional[str] = None
    fielder: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_wicket": self.is_wicket,
            "kind": self.kind.value if self.kind else None,
            "player_out": self.player_out,
            "fielder": self.fielder
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WicketInfo':
        data = data.copy()
        if data.get("kind"):
            data["kind"] = WicketKind(data["kind"])
        return cls(**data)


@dataclass
class BoundaryInfo:
    """Boundary dimensions for venue"""
    short: float = 65.0
    long: float = 75.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {"short": self.short, "long": self.long}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundaryInfo':
        return cls(**data)


@dataclass
class MatchContext:
    """Match contextual information"""
    venue: str
    boundary: BoundaryInfo = field(default_factory=BoundaryInfo)
    weather: Optional[str] = None
    pitch_type: Optional[str] = None
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue": self.venue,
            "boundary": self.boundary.to_dict(),
            "weather": self.weather,
            "pitch_type": self.pitch_type,
            "toss_winner": self.toss_winner,
            "toss_decision": self.toss_decision
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchContext':
        data = data.copy()
        if "boundary" in data:
            data["boundary"] = BoundaryInfo.from_dict(data["boundary"])
        return cls(**data)


@dataclass
class MatchEvent:
    """Ball-level match event (Cricsheet format)"""
    ts: str
    over: int
    ball: int
    innings: int
    bat_striker: str
    bat_non_striker: str
    bowler: str
    runs_batter: int = 0
    runs_extras: int = 0
    wicket: WicketInfo = field(default_factory=WicketInfo)
    phase: MatchPhase = MatchPhase.MIDDLE
    context: Optional[MatchContext] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "over": self.over,
            "ball": self.ball,
            "innings": self.innings,
            "bat_striker": self.bat_striker,
            "bat_non_striker": self.bat_non_striker,
            "bowler": self.bowler,
            "runs_batter": self.runs_batter,
            "runs_extras": self.runs_extras,
            "wicket": self.wicket.to_dict(),
            "phase": self.phase.value,
            "context": self.context.to_dict() if self.context else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchEvent':
        data = data.copy()
        if "wicket" in data:
            data["wicket"] = WicketInfo.from_dict(data["wicket"])
        if "phase" in data:
            data["phase"] = MatchPhase(data["phase"])
        if "context" in data and data["context"]:
            data["context"] = MatchContext.from_dict(data["context"])
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MatchEvent':
        return cls.from_dict(json.loads(json_str))


@dataclass
class LadderLevel:
    """Single price level in order book ladder"""
    price: float
    size: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"price": self.price, "size": self.size}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LadderLevel':
        return cls(price=data["price"], size=data["size"])


@dataclass
class SelectionBook:
    """Order book for a single selection"""
    selection_id: str
    back: List[LadderLevel] = field(default_factory=list)
    lay: List[LadderLevel] = field(default_factory=list)
    traded_volume: float = 0.0
    last_traded_price: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "back": [[level.price, level.size] for level in self.back],
            "lay": [[level.price, level.size] for level in self.lay],
            "traded_volume": self.traded_volume,
            "last_traded_price": self.last_traded_price
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelectionBook':
        back = [LadderLevel(price=p, size=s) for p, s in data.get("back", [])]
        lay = [LadderLevel(price=p, size=s) for p, s in data.get("lay", [])]
        
        return cls(
            selection_id=data["selection_id"],
            back=back,
            lay=lay,
            traded_volume=data.get("traded_volume", 0.0),
            last_traded_price=data.get("last_traded_price")
        )
    
    def best_back_price(self) -> Optional[float]:
        """Get best available back price"""
        return self.back[0].price if self.back else None
    
    def best_lay_price(self) -> Optional[float]:
        """Get best available lay price"""
        return self.lay[0].price if self.lay else None
    
    def mid_price(self) -> Optional[float]:
        """Calculate mid price between best back and lay"""
        back = self.best_back_price()
        lay = self.best_lay_price()
        if back and lay:
            return (back + lay) / 2.0
        return back or lay
    
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points"""
        back = self.best_back_price()
        lay = self.best_lay_price()
        if back and lay and back > 0:
            return ((lay - back) / back) * 10000
        return None


@dataclass
class MarketSnapshot:
    """Exchange-style market snapshot"""
    ts: str
    market_id: str
    status: MarketStatus = MarketStatus.OPEN
    selections: List[SelectionBook] = field(default_factory=list)
    total_matched: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "market_id": self.market_id,
            "status": self.status.value,
            "selections": [sel.to_dict() for sel in self.selections],
            "total_matched": self.total_matched
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketSnapshot':
        selections = [SelectionBook.from_dict(sel) for sel in data.get("selections", [])]
        
        return cls(
            ts=data["ts"],
            market_id=data["market_id"],
            status=MarketStatus(data.get("status", "open")),
            selections=selections,
            total_matched=data.get("total_matched", 0.0)
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MarketSnapshot':
        return cls.from_dict(json.loads(json_str))
    
    def get_selection(self, selection_id: str) -> Optional[SelectionBook]:
        """Get selection book by ID"""
        for selection in self.selections:
            if selection.selection_id == selection_id:
                return selection
        return None


@dataclass
class MatchState:
    """Current match state for strategy decisions"""
    match_id: str
    current_event: Optional[MatchEvent] = None
    innings: int = 1
    over: int = 0
    ball: int = 0
    score: int = 0
    wickets: int = 0
    target: Optional[int] = None
    balls_remaining: int = 120  # 20 overs default
    required_rate: Optional[float] = None
    current_rate: float = 0.0
    phase: MatchPhase = MatchPhase.POWERPLAY
    batting_team: Optional[str] = None
    bowling_team: Optional[str] = None
    striker: Optional[str] = None
    non_striker: Optional[str] = None
    bowler: Optional[str] = None
    context: Optional[MatchContext] = None
    
    def update_from_event(self, event: MatchEvent):
        """Update match state from new event"""
        self.current_event = event
        self.innings = event.innings
        self.over = event.over
        self.ball = event.ball
        self.striker = event.bat_striker
        self.non_striker = event.bat_non_striker
        self.bowler = event.bowler
        self.phase = event.phase
        self.context = event.context
        
        # Update score
        self.score += event.runs_batter + event.runs_extras
        
        # Update wickets
        if event.wicket.is_wicket:
            self.wickets += 1
        
        # Calculate balls remaining (assuming T20)
        total_balls = 120  # 20 overs
        balls_bowled = (event.over * 6) + event.ball
        self.balls_remaining = max(0, total_balls - balls_bowled)
        
        # Calculate current run rate
        if balls_bowled > 0:
            self.current_rate = (self.score * 6.0) / balls_bowled
        
        # Calculate required rate for chase
        if self.target and self.balls_remaining > 0:
            runs_needed = self.target - self.score
            self.required_rate = (runs_needed * 6.0) / self.balls_remaining
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "current_event": self.current_event.to_dict() if self.current_event else None,
            "innings": self.innings,
            "over": self.over,
            "ball": self.ball,
            "score": self.score,
            "wickets": self.wickets,
            "target": self.target,
            "balls_remaining": self.balls_remaining,
            "required_rate": self.required_rate,
            "current_rate": self.current_rate,
            "phase": self.phase.value,
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "striker": self.striker,
            "non_striker": self.non_striker,
            "bowler": self.bowler,
            "context": self.context.to_dict() if self.context else None
        }


@dataclass
class MarketState:
    """Current market state for strategy decisions"""
    market_snapshots: Dict[str, MarketSnapshot] = field(default_factory=dict)
    last_update: Optional[str] = None
    suspended_markets: List[str] = field(default_factory=list)
    
    def update_snapshot(self, snapshot: MarketSnapshot):
        """Update market snapshot"""
        self.market_snapshots[snapshot.market_id] = snapshot
        self.last_update = snapshot.ts
        
        # Update suspension status
        if snapshot.status == MarketStatus.SUSPENDED:
            if snapshot.market_id not in self.suspended_markets:
                self.suspended_markets.append(snapshot.market_id)
        else:
            if snapshot.market_id in self.suspended_markets:
                self.suspended_markets.remove(snapshot.market_id)
    
    def get_snapshot(self, market_id: str) -> Optional[MarketSnapshot]:
        """Get latest snapshot for market"""
        return self.market_snapshots.get(market_id)
    
    def is_market_open(self, market_id: str) -> bool:
        """Check if market is open for trading"""
        snapshot = self.get_snapshot(market_id)
        return snapshot is not None and snapshot.status == MarketStatus.OPEN
    
    def get_best_prices(self, market_id: str, selection_id: str) -> Tuple[Optional[float], Optional[float]]:
        """Get best back and lay prices for selection"""
        snapshot = self.get_snapshot(market_id)
        if not snapshot:
            return None, None
        
        selection = snapshot.get_selection(selection_id)
        if not selection:
            return None, None
        
        return selection.best_back_price(), selection.best_lay_price()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_snapshots": {k: v.to_dict() for k, v in self.market_snapshots.items()},
            "last_update": self.last_update,
            "suspended_markets": self.suspended_markets
        }
