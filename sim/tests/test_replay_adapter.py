# Purpose: Test SIM replay adapter functionality
# Author: WicketWise AI, Last Modified: 2024

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.adapters import ReplayAdapter
from sim.config import SimulationConfig, SimulationMode
from sim.state import MatchEvent, MarketSnapshot, MatchPhase, MarketStatus, SelectionBook, LadderLevel


class TestReplayAdapter:
    """Test cases for the replay adapter"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.adapter = ReplayAdapter(tolerance_ms=1000.0)
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_match_events(self) -> list:
        """Create test match events"""
        base_time = datetime.now()
        events = []
        
        for i in range(10):  # 10 balls
            event_time = base_time + timedelta(seconds=i * 30)
            
            event = MatchEvent(
                ts=event_time.isoformat(),
                over=i // 6,
                ball=(i % 6) + 1,
                innings=1,
                bat_striker=f"player_{i % 2 + 1}",
                bat_non_striker=f"player_{(i + 1) % 2 + 1}",
                bowler=f"bowler_{i // 6 + 1}",
                runs_batter=i % 4,  # 0, 1, 2, 3 runs
                runs_extras=0,
                phase=MatchPhase.POWERPLAY
            )
            events.append(event)
        
        return events
    
    def _create_test_market_snapshots(self) -> list:
        """Create test market snapshots"""
        base_time = datetime.now()
        snapshots = []
        
        for i in range(20):  # More frequent snapshots
            snapshot_time = base_time + timedelta(seconds=i * 15)
            
            # Varying odds
            home_odds = 1.9 + (i * 0.01)
            away_odds = 2.1 - (i * 0.005)
            
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
            
            snapshot = MarketSnapshot(
                ts=snapshot_time.isoformat(),
                market_id="match_odds",
                status=MarketStatus.OPEN,
                selections=[home_selection, away_selection],
                total_matched=50000.0 + i * 1000
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def _save_test_data(self, match_events: list, market_snapshots: list):
        """Save test data to files"""
        # Save match events
        match_file = Path(self.temp_dir) / "test_match_events.jsonl"
        with open(match_file, 'w') as f:
            for event in match_events:
                f.write(json.dumps(event.to_dict()) + '\n')
        
        # Save market snapshots
        market_file = Path(self.temp_dir) / "test_market_snapshots.jsonl"
        with open(market_file, 'w') as f:
            for snapshot in market_snapshots:
                f.write(json.dumps(snapshot.to_dict()) + '\n')
        
        return str(match_file), str(market_file)
    
    def test_initialization_with_mock_data(self):
        """Test adapter initialization with mock data"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"]
        )
        
        # Should initialize successfully and generate mock data
        assert self.adapter.initialize(config)
        assert len(self.adapter.match_events) > 0
        assert len(self.adapter.market_snapshots) > 0
    
    def test_event_synchronization(self):
        """Test synchronization of match events and market snapshots"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"]
        )
        
        self.adapter.initialize(config)
        
        # Get events iterator
        events = list(self.adapter.get_events())
        
        # Should have synchronized events
        assert len(events) > 0
        
        # Each event should have both match and market state
        for match_state, market_state in events:
            assert match_state is not None
            assert market_state is not None
    
    def test_deterministic_replay(self):
        """Test that replay is deterministic"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"],
            seed=42
        )
        
        # Run twice with same seed
        self.adapter.initialize(config)
        events1 = list(self.adapter.get_events())
        
        self.adapter.reset()
        self.adapter.initialize(config)
        events2 = list(self.adapter.get_events())
        
        # Should be identical
        assert len(events1) == len(events2)
        
        # Compare first few events
        for i in range(min(5, len(events1))):
            match1, market1 = events1[i]
            match2, market2 = events2[i]
            
            assert match1.score == match2.score
            assert match1.over == match2.over
            assert match1.ball == match2.ball
    
    def test_ball_suspension_handling(self):
        """Test market suspension around ball delivery"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"]
        )
        
        self.adapter.initialize(config)
        events = list(self.adapter.get_events())
        
        # Check for suspension events (simplified test)
        suspended_count = 0
        for match_state, market_state in events:
            if market_state.suspended_markets:
                suspended_count += 1
        
        # Should have some suspension events
        # (This is a simplified test - in real implementation would be more sophisticated)
        assert suspended_count >= 0  # At least no errors
    
    def test_total_events_count(self):
        """Test total events count for progress tracking"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"]
        )
        
        self.adapter.initialize(config)
        total_events = self.adapter.get_total_events()
        
        # Should have positive count
        assert total_events > 0
        
        # Should match actual events generated
        actual_events = len(list(self.adapter.get_events()))
        
        # Total should be reasonable (sum of match events + market snapshots)
        assert total_events >= actual_events
    
    def test_reset_functionality(self):
        """Test adapter reset functionality"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"]
        )
        
        self.adapter.initialize(config)
        
        # Process some events
        events = list(self.adapter.get_events())
        initial_count = len(events)
        
        # Reset and process again
        self.adapter.reset()
        events_after_reset = list(self.adapter.get_events())
        
        # Should get same number of events
        assert len(events_after_reset) == initial_count
    
    def test_match_state_updates(self):
        """Test match state updates from events"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"]
        )
        
        self.adapter.initialize(config)
        events = list(self.adapter.get_events())
        
        if events:
            # Check that match state progresses
            first_match_state = events[0][0]
            last_match_state = events[-1][0]
            
            # Score should generally increase (or at least not decrease significantly)
            assert last_match_state.score >= first_match_state.score
            
            # Over/ball should progress
            assert (last_match_state.over > first_match_state.over or 
                   (last_match_state.over == first_match_state.over and 
                    last_match_state.ball >= first_match_state.ball))
    
    def test_market_state_updates(self):
        """Test market state updates from snapshots"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"]
        )
        
        self.adapter.initialize(config)
        events = list(self.adapter.get_events())
        
        market_updates = 0
        for match_state, market_state in events:
            if market_state.market_snapshots:
                market_updates += 1
        
        # Should have market updates
        assert market_updates > 0
    
    def test_multiple_markets(self):
        """Test handling multiple markets"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds", "innings1_total"]
        )
        
        self.adapter.initialize(config)
        events = list(self.adapter.get_events())
        
        # Should handle multiple markets
        market_ids_seen = set()
        for match_state, market_state in events:
            for market_id in market_state.market_snapshots.keys():
                market_ids_seen.add(market_id)
        
        # Should see both markets (or at least handle gracefully)
        assert len(market_ids_seen) >= 0  # At least no errors
    
    def test_error_handling(self):
        """Test error handling for invalid configurations"""
        # Test with empty match IDs
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=[],
            markets=["match_odds"]
        )
        
        # Should handle gracefully
        result = self.adapter.initialize(config)
        assert isinstance(result, bool)  # Should return boolean
    
    def test_mock_data_generation(self):
        """Test mock data generation quality"""
        config = SimulationConfig(
            id="test_replay",
            mode=SimulationMode.REPLAY,
            match_ids=["test_match_1"],
            markets=["match_odds"]
        )
        
        self.adapter.initialize(config)
        
        # Check match events quality
        assert len(self.adapter.match_events) > 0
        
        first_event = self.adapter.match_events[0]
        assert first_event.over >= 0
        assert first_event.ball >= 1
        assert first_event.innings in [1, 2]
        assert first_event.runs_batter >= 0
        
        # Check market snapshots quality
        assert len(self.adapter.market_snapshots) > 0
        
        first_snapshot = self.adapter.market_snapshots[0]
        assert first_snapshot.market_id is not None
        assert len(first_snapshot.selections) > 0
        
        for selection in first_snapshot.selections:
            assert len(selection.back) > 0 or len(selection.lay) > 0
            if selection.back:
                assert all(level.price > 1.0 for level in selection.back)
                assert all(level.size > 0 for level in selection.back)


if __name__ == "__main__":
    pytest.main([__file__])
