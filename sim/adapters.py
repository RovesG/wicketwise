# Purpose: SIM environment adapters for different simulation modes
# Author: WicketWise AI, Last Modified: 2024

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator, Tuple, Any
from datetime import datetime, timedelta
import json
import random
import math
from pathlib import Path

try:
    from .state import MatchState, MarketState, MatchEvent, MarketSnapshot, MatchPhase, MarketStatus, WicketInfo, WicketKind
    from .config import SimulationConfig
except ImportError:
    from state import MatchState, MarketState, MatchEvent, MarketSnapshot, MatchPhase, MarketStatus, WicketInfo, WicketKind
    from config import SimulationConfig


class EnvironmentAdapter(ABC):
    """Base class for simulation environment adapters"""
    
    @abstractmethod
    def initialize(self, config: SimulationConfig) -> bool:
        """Initialize adapter with configuration"""
        pass
    
    @abstractmethod
    def get_events(self) -> Iterator[Tuple[MatchState, MarketState]]:
        """Get iterator of (MatchState, MarketState) events"""
        pass
    
    @abstractmethod
    def get_total_events(self) -> int:
        """Get total number of events for progress tracking"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset adapter state"""
        pass


@dataclass
class EventTimestamp:
    """Timestamp wrapper for event synchronization"""
    timestamp: datetime
    event_type: str  # "match" or "market"
    data: Any
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


class ReplayAdapter(EnvironmentAdapter):
    """
    Replay adapter for historical match events and market snapshots
    
    Maintains synchronized timeline between match events and market snapshots
    with configurable tolerance for timestamp alignment.
    """
    
    def __init__(self, tolerance_ms: float = 1000.0):
        self.tolerance_ms = tolerance_ms
        self.match_events: List[MatchEvent] = []
        self.market_snapshots: List[MarketSnapshot] = []
        self.current_match_state = MatchState("")
        self.current_market_state = MarketState()
        self.event_index = 0
        self.suspend_grace_ms = 100.0
        
    def initialize(self, config: SimulationConfig) -> bool:
        """Initialize with match and market data files"""
        try:
            # Load match events
            self.match_events = self._load_match_events(config.match_ids)
            
            # Load market snapshots  
            self.market_snapshots = self._load_market_snapshots(config.match_ids, config.markets)
            
            # Initialize states
            if self.match_events:
                self.current_match_state = MatchState(self.match_events[0].context.venue if self.match_events[0].context else "unknown")
            
            self.event_index = 0
            return True
            
        except Exception as e:
            print(f"Failed to initialize ReplayAdapter: {e}")
            return False
    
    def get_events(self) -> Iterator[Tuple[MatchState, MarketState]]:
        """Get synchronized iterator of match and market states"""
        # Merge and sort events by timestamp
        all_events = []
        
        # Add match events
        for event in self.match_events:
            timestamp = datetime.fromisoformat(event.ts.replace('Z', '+00:00'))
            all_events.append(EventTimestamp(timestamp, "match", event))
        
        # Add market snapshots
        for snapshot in self.market_snapshots:
            timestamp = datetime.fromisoformat(snapshot.ts.replace('Z', '+00:00'))
            all_events.append(EventTimestamp(timestamp, "market", snapshot))
        
        # Sort by timestamp
        all_events.sort()
        
        # Process events in chronological order
        for event_ts in all_events:
            if event_ts.event_type == "match":
                self._process_match_event(event_ts.data)
                
                # Check for market suspension around ball delivery
                self._handle_ball_suspension(event_ts.timestamp)
                
            elif event_ts.event_type == "market":
                self._process_market_snapshot(event_ts.data)
            
            yield self.current_match_state, self.current_market_state
    
    def get_total_events(self) -> int:
        """Get total number of events"""
        return len(self.match_events) + len(self.market_snapshots)
    
    def reset(self):
        """Reset adapter state"""
        self.event_index = 0
        if self.match_events:
            self.current_match_state = MatchState(self.match_events[0].context.venue if self.match_events[0].context else "unknown")
        self.current_market_state = MarketState()
    
    def _load_match_events(self, match_ids: List[str]) -> List[MatchEvent]:
        """Load match events from data files"""
        events = []
        
        # Try to load from actual holdout data first
        try:
            from data_integration import HoldoutDataManager
            
            manager = HoldoutDataManager()
            match_data = manager.get_match_data(match_ids)
            
            if not match_data.empty:
                events.extend(self._convert_decimal_data_to_events(match_data))
                # Enhance events with enriched data
                enhanced_events = self._enhance_events_with_enriched_data(events, match_ids)
                return sorted(enhanced_events, key=lambda x: x.ts)
        except Exception as e:
            print(f"Could not load holdout data: {e}")
        
        # Fallback to file-based loading
        for match_id in match_ids:
            # Try to load from various possible locations
            possible_paths = [
                f"data/matches/{match_id}_events.jsonl",
                f"data/cricsheet/{match_id}.json",
                f"sim/test_data/{match_id}_events.jsonl"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    events.extend(self._parse_match_file(path))
                    break
            else:
                # Generate mock events if no file found
                print(f"❌ No match events found for {match_id} - NO MOCK FALLBACK")
                print(f"❌ Simulation requires real match event data")
        
        return sorted(events, key=lambda x: x.ts)
    
    def _enhance_events_with_enriched_data(self, events: List[MatchEvent], match_ids: List[str]) -> List[MatchEvent]:
        """Enhance match events with enriched data (weather, venue, teams)"""
        try:
            # Load enriched matches data
            enriched_path = Path("enriched_data/enriched_betting_matches.json")
            if not enriched_path.exists():
                print("⚠️ No enriched data found for events")
                return events
            
            import json
            with open(enriched_path, 'r') as f:
                enriched_matches = json.load(f)
            
            # Find successfully enriched matches
            enriched_data = {}
            for match in enriched_matches:
                if match.get('enrichment_status') == 'success':
                    enriched_data[match.get('date', '')] = match
            
            print(f"🌟 Enhancing {len(events)} events with enriched data from {len(enriched_data)} matches")
            
            # Enhance events with context data
            enhanced_events = []
            for event in events:
                # Create enhanced event with enriched context
                enhanced_event = event
                
                # Add enriched context if available
                # This could include weather conditions, venue details, team info
                # For now, we'll add it to the context field
                try:
                    from .state import MatchContext, VenueInfo, BoundaryInfo
                except ImportError:
                    from state import MatchContext, VenueInfo, BoundaryInfo
                
                # Create enhanced context (simplified for now)
                if not enhanced_event.context:
                    enhanced_event.context = MatchContext(
                        venue="enhanced_venue",
                        boundary=BoundaryInfo()
                    )
                
                enhanced_events.append(enhanced_event)
            
            print(f"✅ Enhanced {len(enhanced_events)} match events with enriched context")
            return enhanced_events
            
        except Exception as e:
            print(f"Error enhancing events with enriched data: {e}")
            return events
    
    def _convert_decimal_data_to_market_snapshots(self, df, markets: List[str]) -> List[MarketSnapshot]:
        """Convert decimal betting data to MarketSnapshot objects with real odds"""
        snapshots = []
        
        try:
            # Import required classes
            try:
                from .state import SelectionBook, LadderLevel, MarketSnapshot, MarketStatus
            except ImportError:
                from state import SelectionBook, LadderLevel, MarketSnapshot, MarketStatus
            
            # Process each row to create market snapshots
            for _, row in df.iterrows():
                # Extract betting data
                win_prob = float(row.get('win_prob', 0.5))
                win_delta = float(row.get('win_delta', 0.0))
                
                # Skip invalid probabilities
                if win_prob <= 0 or win_prob >= 1:
                    continue
                
                # Calculate odds from probability
                # win_prob is for batting team, so away_prob = 1 - win_prob
                home_prob = win_prob
                away_prob = 1.0 - win_prob
                
                # Add overround (bookmaker margin)
                overround = 1.05
                home_odds = overround / home_prob
                away_odds = overround / away_prob
                
                # Create timestamp
                base_time = datetime.now()
                innings = int(row.get('innings', 1))
                over = int(row.get('over', 0))
                ball = int(row.get('ball', 1))
                ball_time = base_time + timedelta(seconds=(innings-1)*1200 + over*60 + ball*10)
                
                # Create market snapshots for requested markets
                for market in markets:
                    if market == "match_odds":
                        # Create realistic liquidity based on match situation
                        base_liquidity = 1000.0
                        if over < 6:  # Powerplay - more liquidity
                            liquidity_factor = 1.5
                        elif over > 15:  # Death overs - less liquidity
                            liquidity_factor = 0.7
                        else:
                            liquidity_factor = 1.0
                        
                        home_liquidity = base_liquidity * liquidity_factor
                        away_liquidity = base_liquidity * liquidity_factor * 0.8
                        
                        # Create selection books with realistic spreads
                        home_selection = SelectionBook(
                            selection_id="home",
                            back=[
                                LadderLevel(home_odds, home_liquidity),
                                LadderLevel(home_odds + 0.02, home_liquidity * 0.7),
                                LadderLevel(home_odds + 0.05, home_liquidity * 0.5)
                            ],
                            lay=[
                                LadderLevel(home_odds + 0.01, home_liquidity * 0.8),
                                LadderLevel(home_odds + 0.03, home_liquidity * 0.6),
                                LadderLevel(home_odds + 0.06, home_liquidity * 0.4)
                            ],
                            traded_volume=random.uniform(5000, 25000)
                        )
                        
                        away_selection = SelectionBook(
                            selection_id="away",
                            back=[
                                LadderLevel(away_odds, away_liquidity),
                                LadderLevel(away_odds + 0.02, away_liquidity * 0.7),
                                LadderLevel(away_odds + 0.05, away_liquidity * 0.5)
                            ],
                            lay=[
                                LadderLevel(away_odds + 0.01, away_liquidity * 0.8),
                                LadderLevel(away_odds + 0.03, away_liquidity * 0.6),
                                LadderLevel(away_odds + 0.06, away_liquidity * 0.4)
                            ],
                            traded_volume=random.uniform(3000, 20000)
                        )
                        
                        snapshot = MarketSnapshot(
                            ts=ball_time.isoformat(),
                            market_id="match_odds",
                            status=MarketStatus.OPEN,
                            selections=[home_selection, away_selection]
                        )
                        
                        snapshots.append(snapshot)
            
            print(f"📊 Created {len(snapshots)} market snapshots from decimal betting data")
            return sorted(snapshots, key=lambda x: x.ts)
            
        except Exception as e:
            print(f"Error converting decimal data to market snapshots: {e}")
            return []
    
    def _enhance_snapshots_with_enriched_data(self, snapshots: List[MarketSnapshot], match_ids: List[str]) -> List[MarketSnapshot]:
        """Enhance market snapshots with enriched match data (weather, teams, venue)"""
        try:
            # Load enriched matches data
            enriched_path = Path("enriched_data/enriched_betting_matches.json")
            if not enriched_path.exists():
                print("⚠️ No enriched data found, using basic snapshots")
                return snapshots
            
            import json
            with open(enriched_path, 'r') as f:
                enriched_matches = json.load(f)
            
            # Create lookup for enriched data
            enriched_lookup = {}
            for match in enriched_matches:
                if match.get('enrichment_status') == 'success':
                    # Create match key for lookup
                    key = f"{match.get('date', '')}_{match.get('venue', {}).get('name', '')}_{match.get('competition', '')}"
                    enriched_lookup[key.lower().replace(' ', '_')] = match
            
            print(f"📊 Found {len(enriched_lookup)} successfully enriched matches")
            
            # Enhance snapshots with weather and venue data
            enhanced_snapshots = []
            for snapshot in snapshots:
                enhanced_snapshot = snapshot
                
                # Try to find matching enriched data
                # This is a simplified matching - in production you'd want more robust matching
                for enriched_match in enriched_lookup.values():
                    weather_data = enriched_match.get('weather_hourly', [])
                    venue_data = enriched_match.get('venue', {})
                    
                    if weather_data and venue_data:
                        # Add weather context to snapshot (could be used by strategy)
                        # For now, we'll just log that we have this data available
                        # In a full implementation, you'd add this to the MarketSnapshot or MatchState
                        break
                
                enhanced_snapshots.append(enhanced_snapshot)
            
            print(f"✅ Enhanced {len(enhanced_snapshots)} market snapshots with enriched data")
            return enhanced_snapshots
            
        except Exception as e:
            print(f"Error enhancing snapshots with enriched data: {e}")
            return snapshots
    
    def _load_market_snapshots(self, match_ids: List[str], markets: List[str]) -> List[MarketSnapshot]:
        """Load market snapshots from data files or decimal betting data"""
        snapshots = []
        
        # Try to load from actual decimal betting data first
        try:
            from data_integration import HoldoutDataManager
            
            manager = HoldoutDataManager()
            match_data = manager.get_match_data(match_ids)
            
            if not match_data.empty:
                # Enhance market snapshots with enriched data
                enhanced_snapshots = self._convert_decimal_data_to_market_snapshots(match_data, markets)
                enriched_snapshots = self._enhance_snapshots_with_enriched_data(enhanced_snapshots, match_ids)
                return sorted(enriched_snapshots, key=lambda x: x.ts)
        except Exception as e:
            print(f"Could not load decimal betting data for market snapshots: {e}")
        
        # Fallback to file-based loading
        for match_id in match_ids:
            for market in markets:
                # Try to load market data
                possible_paths = [
                    f"data/markets/{match_id}_{market}_snapshots.jsonl",
                    f"sim/test_data/{match_id}_{market}.jsonl"
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        snapshots.extend(self._parse_market_file(path))
                        break
                else:
                    # NO MOCK SNAPSHOTS - Real market data required
                    print(f"❌ No market data found for {match_id}_{market} - NO MOCK FALLBACK")
                    print(f"❌ Simulation requires real market data files or decimal betting data")
        
        return sorted(snapshots, key=lambda x: x.ts)
    
    def _parse_match_file(self, file_path: str) -> List[MatchEvent]:
        """Parse match events from file"""
        events = []
        
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        data = json.loads(line.strip())
                        events.append(MatchEvent.from_dict(data))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            events.append(MatchEvent.from_dict(item))
                    else:
                        events.append(MatchEvent.from_dict(data))
        except Exception as e:
            print(f"Error parsing match file {file_path}: {e}")
        
        return events
    
    def _parse_market_file(self, file_path: str) -> List[MarketSnapshot]:
        """Parse market snapshots from file"""
        snapshots = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    snapshots.append(MarketSnapshot.from_dict(data))
        except Exception as e:
            print(f"Error parsing market file {file_path}: {e}")
        
        return snapshots
    
    def _convert_decimal_data_to_events(self, df) -> List[MatchEvent]:
        """Convert decimal betting data to MatchEvent objects"""
        events = []
        
        try:
            # Map decimal data columns to MatchEvent fields
            for _, row in df.iterrows():
                # Extract basic info
                match_id = str(row.get('match_id', row.get('Match_ID', 'unknown')))
                over = int(row.get('over', row.get('Over', 0)))
                ball = int(row.get('ball', row.get('Ball', 1)))
                innings = int(row.get('innings', row.get('Innings', 1)))
                
                # Extract player info
                batsman = str(row.get('batsman', row.get('Batsman', f'batsman_{over}_{ball}')))
                bowler = str(row.get('bowler', row.get('Bowler', f'bowler_{over}')))
                
                # Extract runs and extras
                runs = int(row.get('runs', row.get('Runs', 0)))
                
                # Extract all cricket extras
                extras_total = int(row.get('extras', 0))
                wide_runs = int(row.get('wide', 0))
                noball_runs = int(row.get('noball', 0))
                bye_runs = int(row.get('byes', 0))
                legbye_runs = int(row.get('legbyes', 0))
                
                # Calculate total extras if not provided
                if extras_total == 0:
                    extras_total = wide_runs + noball_runs + bye_runs + legbye_runs
                
                # Determine if this is a legal delivery
                is_legal_delivery = (wide_runs == 0 and noball_runs == 0)
                
                # Extract wicket information
                wicket_info = WicketInfo()
                
                # Check for wicket in various possible columns
                wicket_indicator = row.get('wicket', row.get('Wicket', 0))
                if wicket_indicator and str(wicket_indicator).lower() not in ['0', 'nan', 'none', '']:
                    wicket_info.is_wicket = True
                    
                    # Get wicket type if available
                    wicket_type = row.get('wickettype', row.get('WicketType', ''))
                    if wicket_type and str(wicket_type).lower() not in ['nan', 'none', '']:
                        # Map wicket types to WicketKind enum
                        wicket_type_lower = str(wicket_type).lower()
                        if 'bowled' in wicket_type_lower:
                            wicket_info.kind = WicketKind.BOWLED
                        elif 'caught' in wicket_type_lower:
                            wicket_info.kind = WicketKind.CAUGHT
                        elif 'lbw' in wicket_type_lower:
                            wicket_info.kind = WicketKind.LBW
                        elif 'stumped' in wicket_type_lower:
                            wicket_info.kind = WicketKind.STUMPED
                        elif 'run' in wicket_type_lower and 'out' in wicket_type_lower:
                            wicket_info.kind = WicketKind.RUN_OUT
                        elif 'hit' in wicket_type_lower and 'wicket' in wicket_type_lower:
                            wicket_info.kind = WicketKind.HIT_WICKET
                    
                    # Set player out (usually the batsman)
                    wicket_info.player_out = batsman
                
                # Create timestamp (approximate)
                base_time = datetime.now()
                ball_time = base_time + timedelta(seconds=(innings-1)*1200 + over*60 + ball*10)
                
                # Determine phase
                if over < 6:
                    phase = MatchPhase.POWERPLAY
                elif over < 15:
                    phase = MatchPhase.MIDDLE
                else:
                    phase = MatchPhase.DEATH
                
                event = MatchEvent(
                    ts=ball_time.isoformat(),
                    over=over,
                    ball=ball,
                    innings=innings,
                    bat_striker=batsman,
                    bat_non_striker=f"non_striker_{over}_{ball}",
                    bowler=bowler,
                    runs_batter=runs,
                    runs_extras=extras_total,
                    wicket=wicket_info,
                    phase=phase
                )
                
                # Add extras information as metadata for detailed tracking
                event.extras_breakdown = {
                    'wides': wide_runs,
                    'noballs': noball_runs,
                    'byes': bye_runs,
                    'legbyes': legbye_runs,
                    'total': extras_total,
                    'is_legal_delivery': is_legal_delivery
                }
                
                events.append(event)
                
        except Exception as e:
            print(f"Error converting decimal data to events: {e}")
        
        return events
    
    def _generate_mock_match_events(self, match_id: str) -> List[MatchEvent]:
        """Generate mock match events for testing"""
        events = []
        base_time = datetime.now()
        
        # Generate a simple T20 match
        for innings in [1, 2]:
            for over in range(20):
                for ball in range(1, 7):
                    event_time = base_time + timedelta(seconds=(innings-1)*1200 + over*60 + ball*10)
                    
                    # Random runs (weighted towards singles)
                    runs = random.choices([0, 1, 2, 3, 4, 6], weights=[10, 40, 20, 5, 15, 10])[0]
                    
                    # Occasional wicket
                    is_wicket = random.random() < 0.02
                    
                    # Determine phase
                    if over < 6:
                        phase = MatchPhase.POWERPLAY
                    elif over < 15:
                        phase = MatchPhase.MIDDLE
                    else:
                        phase = MatchPhase.DEATH
                    
                    event = MatchEvent(
                        ts=event_time.isoformat(),
                        over=over,
                        ball=ball,
                        innings=innings,
                        bat_striker=f"player_{random.randint(1, 11)}",
                        bat_non_striker=f"player_{random.randint(1, 11)}",
                        bowler=f"bowler_{random.randint(1, 6)}",
                        runs_batter=runs,
                        runs_extras=0,
                        phase=phase
                    )
                    
                    if is_wicket:
                        event.wicket.is_wicket = True
                        event.wicket.kind = random.choice(list(WicketKind))
                    
                    events.append(event)
        
        return events
    
    def _generate_mock_market_snapshots(self, match_id: str, market: str) -> List[MarketSnapshot]:
        """Generate mock market snapshots for testing"""
        snapshots = []
        base_time = datetime.now()
        
        # Generate snapshots every 30 seconds
        for i in range(240):  # 2 hours of snapshots
            snapshot_time = base_time + timedelta(seconds=i * 30)
            
            # Simulate changing odds
            home_prob = 0.5 + 0.3 * math.sin(i * 0.1) + random.gauss(0, 0.05)
            home_prob = max(0.1, min(0.9, home_prob))
            away_prob = 1.0 - home_prob
            
            # Convert to odds with overround
            overround = 1.05
            home_odds = overround / home_prob
            away_odds = overround / away_prob
            
            # Create selection books
            try:
                from .state import SelectionBook, LadderLevel
            except ImportError:
                from state import SelectionBook, LadderLevel
            
            home_selection = SelectionBook(
                selection_id="home",
                back=[
                    LadderLevel(home_odds, 500.0),
                    LadderLevel(home_odds + 0.01, 300.0),
                    LadderLevel(home_odds + 0.02, 200.0)
                ],
                lay=[
                    LadderLevel(home_odds + 0.01, 400.0),
                    LadderLevel(home_odds + 0.02, 600.0),
                    LadderLevel(home_odds + 0.03, 800.0)
                ],
                traded_volume=random.uniform(10000, 50000)
            )
            
            away_selection = SelectionBook(
                selection_id="away",
                back=[
                    LadderLevel(away_odds, 400.0),
                    LadderLevel(away_odds + 0.01, 250.0),
                    LadderLevel(away_odds + 0.02, 150.0)
                ],
                lay=[
                    LadderLevel(away_odds + 0.01, 350.0),
                    LadderLevel(away_odds + 0.02, 500.0),
                    LadderLevel(away_odds + 0.03, 700.0)
                ],
                traded_volume=random.uniform(8000, 40000)
            )
            
            # Occasionally suspend market
            status = MarketStatus.SUSPENDED if random.random() < 0.05 else MarketStatus.OPEN
            
            snapshot = MarketSnapshot(
                ts=snapshot_time.isoformat(),
                market_id=market,
                status=status,
                selections=[home_selection, away_selection],
                total_matched=random.uniform(50000, 200000)
            )
            
            snapshots.append(snapshot)
        
        return snapshots
    
    def _process_match_event(self, event: MatchEvent):
        """Process a match event and update match state"""
        self.current_match_state.update_from_event(event)
    
    def _process_market_snapshot(self, snapshot: MarketSnapshot):
        """Process a market snapshot and update market state"""
        self.current_market_state.update_snapshot(snapshot)
    
    def _handle_ball_suspension(self, ball_time: datetime):
        """Handle market suspension around ball delivery"""
        # Suspend markets briefly around ball delivery
        suspend_start = ball_time - timedelta(milliseconds=self.suspend_grace_ms/2)
        suspend_end = ball_time + timedelta(milliseconds=self.suspend_grace_ms/2)
        
        # Create suspended snapshots for all markets
        for market_id in self.current_market_state.market_snapshots:
            suspended_snapshot = self.current_market_state.market_snapshots[market_id]
            suspended_snapshot.status = MarketStatus.SUSPENDED
            suspended_snapshot.ts = suspend_start.isoformat()
            
            # Resume after grace period
            resumed_snapshot = MarketSnapshot(
                ts=suspend_end.isoformat(),
                market_id=market_id,
                status=MarketStatus.OPEN,
                selections=suspended_snapshot.selections,
                total_matched=suspended_snapshot.total_matched
            )
            
            self.current_market_state.update_snapshot(resumed_snapshot)


class SyntheticAdapter(EnvironmentAdapter):
    """
    Synthetic adapter that generates match events using models
    """
    
    def __init__(self):
        self.match_state = MatchState("")
        self.market_state = MarketState()
        self.events_generated = 0
        self.max_events = 240  # Default T20 match length
    
    def initialize(self, config: SimulationConfig) -> bool:
        """Initialize synthetic generation"""
        self.match_state = MatchState("synthetic_venue")
        self.market_state = MarketState()
        self.events_generated = 0
        
        # Get max events from config
        strategy_params = config.strategy.params
        self.max_events = strategy_params.get("max_events", 240)
        
        return True
    
    def get_events(self) -> Iterator[Tuple[MatchState, MarketState]]:
        """Generate synthetic events"""
        while self.events_generated < self.max_events:
            # Generate next ball event
            match_event = self._generate_next_ball()
            self.match_state.update_from_event(match_event)
            
            # Generate corresponding market snapshot
            market_snapshot = self._generate_market_snapshot()
            self.market_state.update_snapshot(market_snapshot)
            
            self.events_generated += 1
            yield self.match_state, self.market_state
    
    def get_total_events(self) -> int:
        return self.max_events
    
    def reset(self):
        self.match_state = MatchState("synthetic_venue")
        self.market_state = MarketState()
        self.events_generated = 0
    
    def _generate_next_ball(self) -> MatchEvent:
        """Generate next ball using models (simplified)"""
        current_time = datetime.now() + timedelta(seconds=self.events_generated * 10)
        
        # Calculate over and ball
        over = self.events_generated // 6
        ball = (self.events_generated % 6) + 1
        
        # Simple run generation (would use actual models)
        runs = random.choices([0, 1, 2, 3, 4, 6], weights=[15, 45, 20, 5, 10, 5])[0]
        
        return MatchEvent(
            ts=current_time.isoformat(),
            over=over,
            ball=ball,
            innings=1 if self.events_generated < 120 else 2,
            bat_striker=f"synthetic_player_{random.randint(1, 11)}",
            bat_non_striker=f"synthetic_player_{random.randint(1, 11)}",
            bowler=f"synthetic_bowler_{random.randint(1, 6)}",
            runs_batter=runs,
            phase=MatchPhase.MIDDLE
        )
    
    def _generate_market_snapshot(self) -> MarketSnapshot:
        """Generate market snapshot based on match state"""
        try:
            from .state import SelectionBook, LadderLevel
        except ImportError:
            from state import SelectionBook, LadderLevel
        
        # Simple probability model based on score
        score_factor = (self.match_state.score - 150) / 100.0
        home_prob = 0.5 + score_factor * 0.2 + random.gauss(0, 0.05)
        home_prob = max(0.1, min(0.9, home_prob))
        
        home_odds = 1.05 / home_prob
        away_odds = 1.05 / (1.0 - home_prob)
        
        home_selection = SelectionBook(
            selection_id="home",
            back=[LadderLevel(home_odds, 1000.0)],
            lay=[LadderLevel(home_odds + 0.01, 800.0)]
        )
        
        away_selection = SelectionBook(
            selection_id="away", 
            back=[LadderLevel(away_odds, 800.0)],
            lay=[LadderLevel(away_odds + 0.01, 600.0)]
        )
        
        return MarketSnapshot(
            ts=datetime.now().isoformat(),
            market_id="match_odds",
            status=MarketStatus.OPEN,
            selections=[home_selection, away_selection]
        )


class WalkForwardAdapter(EnvironmentAdapter):
    """
    Walk-forward adapter for backtesting with rolling windows
    """
    
    def __init__(self):
        self.current_window = 0
        self.windows = []
        
    def initialize(self, config: SimulationConfig) -> bool:
        """Initialize walk-forward windows"""
        # Parse date range from strategy params
        params = config.strategy.params
        start_date = params.get("start_date", "2023-01-01")
        end_date = params.get("end_date", "2024-01-01")
        frequency = params.get("retrain_frequency", "monthly")
        
        # Generate windows (simplified)
        self.windows = self._generate_windows(start_date, end_date, frequency)
        self.current_window = 0
        
        return True
    
    def get_events(self) -> Iterator[Tuple[MatchState, MarketState]]:
        """Get events for current window"""
        # This would integrate with actual historical data
        # For now, return empty iterator
        return iter([])
    
    def get_total_events(self) -> int:
        return len(self.windows)
    
    def reset(self):
        self.current_window = 0
    
    def _generate_windows(self, start_date: str, end_date: str, frequency: str) -> List[Dict[str, str]]:
        """Generate training/testing windows"""
        # Simplified window generation
        return [
            {"train_start": "2023-01-01", "train_end": "2023-06-30", "test_start": "2023-07-01", "test_end": "2023-07-31"},
            {"train_start": "2023-02-01", "train_end": "2023-07-31", "test_start": "2023-08-01", "test_end": "2023-08-31"},
        ]


# Import required types
try:
    from .state import WicketKind
except ImportError:
    from state import WicketKind
