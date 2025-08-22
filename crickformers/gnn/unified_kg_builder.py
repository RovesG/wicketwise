# Purpose: Unified Cricket Knowledge Graph Builder with Ball-by-Ball Granularity
# Author: WicketWise Team, Last Modified: 2025-08-17

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import logging
from collections import defaultdict
import pickle
import json
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PlayerProfile:
    """Unified player profile with all cricket roles"""
    name: str
    player_id: str
    primary_role: str  # batsman, bowler, all-rounder, wicket-keeper
    hand: str  # left, right
    bowling_style: Optional[str] = None  # pace, spin, off-spin, leg-spin, etc.
    
    # Aggregate Career Stats
    batting_stats: Dict[str, float] = None
    bowling_stats: Dict[str, float] = None
    fielding_stats: Dict[str, float] = None
    
    # Situational Analysis
    vs_pace: Dict[str, float] = None
    vs_spin: Dict[str, float] = None
    in_powerplay: Dict[str, float] = None
    in_death_overs: Dict[str, float] = None
    by_venue: Dict[str, Dict[str, float]] = None
    
    # Metadata
    matches_played: int = 0
    career_span: Tuple[str, str] = None
    teams: List[str] = None
    
    def __post_init__(self):
        if self.batting_stats is None:
            self.batting_stats = {}
        if self.bowling_stats is None:
            self.bowling_stats = {}
        if self.fielding_stats is None:
            self.fielding_stats = {}
        if self.vs_pace is None:
            self.vs_pace = {}
        if self.vs_spin is None:
            self.vs_spin = {}
        if self.in_powerplay is None:
            self.in_powerplay = {}
        if self.in_death_overs is None:
            self.in_death_overs = {}
        if self.by_venue is None:
            self.by_venue = {}
        if self.teams is None:
            self.teams = []


@dataclass
class BallEvent:
    """Individual ball event with full context"""
    ball_id: str
    match_id: str
    innings: int
    over: float
    ball_in_over: int
    
    # Players
    batter: str
    bowler: str
    non_striker: Optional[str] = None
    
    # Outcome
    runs_scored: int = 0
    extras: int = 0
    is_wicket: bool = False
    wicket_type: Optional[str] = None
    
    # Context
    team_score: int = 0
    team_wickets: int = 0
    run_rate: float = 0.0
    required_rate: float = 0.0
    powerplay: bool = False
    phase: str = "middle_overs"  # powerplay, middle_overs, death_overs
    
    # Player attributes
    batter_hand: str = "right"
    bowler_type: str = "pace"
    
    # Match context
    venue: str = ""
    competition: str = ""
    date: Optional[str] = None
    
    # Advanced context
    field_positions: Optional[Dict] = None
    ball_tracking: Optional[Dict] = None


class UnifiedKGBuilder:
    """
    Builds a unified cricket knowledge graph from ball-by-ball data
    
    Features:
    - Unified player nodes with batting, bowling, fielding stats
    - Ball-by-ball event preservation
    - Situational analysis (vs spinners, death overs, venues)
    - Advanced relationship modeling
    - Efficient querying capabilities
    """
    
    def __init__(self, data_dir: str, enriched_data_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.enriched_data_path = Path(enriched_data_path) if enriched_data_path else None
        self.graph = nx.Graph()
        self.players = {}  # player_name -> PlayerProfile
        self.balls = []    # List of BallEvent objects
        self.matches = {}  # match_id -> match_info
        self.venues = {}   # venue_name -> venue_info
        self.enrichments = {}  # match_key -> enriched_data
        
        # Statistics tracking
        self.stats = {
            'total_players': 0,
            'total_balls': 0,
            'total_matches': 0,
            'total_venues': 0,
            'enriched_matches': 0,
            'processing_time': 0
        }
    
    def build_from_available_data(
        self,
        data_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, str, int, Dict[str, Any]], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> nx.Graph:
        """
        Build unified knowledge graph from available cricket data
        
        Args:
            data_path: Optional specific path to data file (auto-detects if None)
            
        Returns:
            NetworkX graph with unified structure
        """
        logger.info("🏗️ Building Unified Cricket Knowledge Graph")
        start_time = datetime.now()
        
        # Step 0: Load enrichment data if available
        logger.info("🌟 Loading enrichment data...")
        self.enrichments = self._load_enrichment_data()
        
        # Step 1: Find and load available cricket data
        logger.info("📊 Auto-detecting available cricket data...")
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")
        balls_df = self._load_available_data(data_path)
        if progress_callback is not None:
            try:
                progress_callback(
                    "load_data",
                    f"Loaded {len(balls_df):,} ball records",
                    20,
                    {"balls": int(len(balls_df))}
                )
            except Exception:
                pass
        
        # Step 2: Extract player profiles
        logger.info("👥 Building player profiles...")
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")
        self._build_player_profiles(balls_df, progress_callback=progress_callback, should_cancel=should_cancel)
        
        # Step 3: Process ball events
        logger.info("⚾ Processing ball events...")
        # Provide a progress-aware processing of ball events
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")
        self._process_ball_events(balls_df, progress_callback=progress_callback, should_cancel=should_cancel)
        if progress_callback is not None:
            try:
                progress_callback(
                    "ball_events",
                    f"Processed {self.stats['total_balls']:,} ball events",
                    75,
                    {"balls_processed": int(self.stats['total_balls'])}
                )
            except Exception:
                pass
        
        # Step 4: Build graph structure
        logger.info("🕸️ Building graph relationships...")
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")
        self._build_graph_structure(balls_df)
        if progress_callback is not None:
            try:
                progress_callback(
                    "relationships",
                    f"Added {self.graph.number_of_edges():,} relationships",
                    85,
                    {"edges": int(self.graph.number_of_edges())}
                )
            except Exception:
                pass
        
        # Step 5: Calculate advanced statistics
        logger.info("📈 Computing situational statistics...")
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")
        self._compute_situational_stats(balls_df)
        if progress_callback is not None:
            try:
                progress_callback(
                    "situational_stats",
                    "Computed situational statistics",
                    92,
                    {}
                )
            except Exception:
                pass
        
        # Step 6: Add graph metadata
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")
        self._add_graph_metadata()
        if progress_callback is not None:
            try:
                progress_callback(
                    "metadata",
                    "Added graph metadata",
                    95,
                    {
                        "players": int(self.stats['total_players']),
                        "balls": int(self.stats['total_balls']),
                        "venues": int(self.stats['total_venues'])
                    }
                )
            except Exception:
                pass
        
        # Final statistics
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Knowledge Graph Built Successfully!")
        logger.info(f"   📊 {self.stats['total_players']:,} players")
        logger.info(f"   ⚾ {self.stats['total_balls']:,} balls")
        logger.info(f"   🏟️ {self.stats['total_venues']:,} venues")
        logger.info(f"   🕒 {self.stats['processing_time']:.1f}s processing time")
        
        return self.graph
    
    def _load_enrichment_data(self) -> Dict[str, Any]:
        """Load enriched match data if available"""
        if not self.enriched_data_path or not self.enriched_data_path.exists():
            logger.info("📊 No enriched data found, building KG without weather/squad enhancements")
            return {}
        
        logger.info(f"📊 Loading enriched data from {self.enriched_data_path}")
        try:
            with open(self.enriched_data_path, 'r') as f:
                enriched_data = json.load(f)
            
            # Convert to match_key -> enrichment mapping
            enrichments = {}
            for match in enriched_data:
                # Create match key from enriched data
                date = match.get('date', '')
                teams = [team['name'] for team in match.get('teams', [])]
                venue = match.get('venue', {}).get('name', '')
                
                if len(teams) >= 2:
                    match_key = self._create_match_key(date, teams[0], teams[1], venue)
                    enrichments[match_key] = match
            
            logger.info(f"✅ Loaded {len(enrichments)} enriched matches")
            return enrichments
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load enriched data: {e}")
            return {}
    
    def _create_match_key(self, date: str, team1: str, team2: str, venue: str) -> str:
        """Create standardized match key for enrichment matching"""
        # Normalize team names for consistent matching
        team1_norm = team1.strip().lower()
        team2_norm = team2.strip().lower()
        venue_norm = venue.strip().lower()
        
        # Sort teams alphabetically for consistency
        teams_sorted = sorted([team1_norm, team2_norm])
        
        return f"{date}_{teams_sorted[0]}_{teams_sorted[1]}_{venue_norm}"
    
    def _find_venue_enrichment(self, venue_name: str) -> Optional[Dict[str, Any]]:
        """Find enrichment data for a venue using fuzzy matching"""
        if not self.enrichments:
            return None
        
        venue_norm = venue_name.strip().lower()
        
        # Look for exact matches first
        for enrichment in self.enrichments.values():
            enriched_venue = enrichment.get('venue', {})
            if enriched_venue.get('name', '').strip().lower() == venue_norm:
                return enriched_venue
        
        # Fuzzy matching for venue names
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0.6  # Minimum similarity threshold
        
        for enrichment in self.enrichments.values():
            enriched_venue = enrichment.get('venue', {})
            enriched_name = enriched_venue.get('name', '').strip().lower()
            
            if enriched_name:
                similarity = SequenceMatcher(None, venue_norm, enriched_name).ratio()
                if similarity > best_score:
                    best_score = similarity
                    best_match = enriched_venue
        
        return best_match
    
    def _find_match_enrichment(self, date: str, teams: List[str], venue: str) -> Optional[Dict[str, Any]]:
        """Find enrichment data for a specific match"""
        if not self.enrichments or len(teams) < 2:
            return None
        
        match_key = self._create_match_key(date, teams[0], teams[1], venue)
        return self.enrichments.get(match_key)
    
    def _load_available_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load and clean available cricket data"""
        try:
            # Auto-detect available data files
            possible_paths = [
                data_path,
                "artifacts/kg_background/events/events.parquet",
                "artifacts/train_exports/t20_from_json/t20_events.parquet",
                "nvplay_data_v3.csv",
                "decimal_data_v3.csv"
            ]
            
            df = None
            data_source = None
            
            for path in possible_paths:
                if path is None:
                    continue
                    
                file_path = Path(path)
                if file_path.exists():
                    try:
                        if path.endswith('.parquet'):
                            df = pd.read_parquet(path)
                            data_source = f"Parquet: {path}"
                        elif path.endswith('.csv'):
                            df = pd.read_csv(path)
                            data_source = f"CSV: {path}"
                        
                        if df is not None and len(df) > 0:
                            logger.info(f"   📈 Loaded {len(df):,} records from {data_source}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")
                        continue
            
            if df is None:
                raise FileNotFoundError("No suitable cricket data files found. Please ensure data is available.")
            
            # Standardize column names (different data sources may have different schemas)
            df = self._standardize_columns(df)
            
            # Basic data cleaning
            required_cols = ['batter_name', 'bowler_name']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            df = df.dropna(subset=required_cols)
            
            # Ensure numeric columns
            if 'over' in df.columns:
                df['over'] = pd.to_numeric(df['over'], errors='coerce').fillna(1.0)
            else:
                df['over'] = 1.0  # Default over
                
            if 'runs_scored' in df.columns:
                df['runs_scored'] = pd.to_numeric(df['runs_scored'], errors='coerce').fillna(0).astype(int)
            else:
                df['runs_scored'] = 0  # Default runs
                
            if 'extras' in df.columns:
                df['extras'] = pd.to_numeric(df['extras'], errors='coerce').fillna(0).astype(int)
            else:
                df['extras'] = 0  # Default extras
            
            # Derive additional fields
            if 'dismissal_kind' in df.columns:
                df['is_wicket'] = (df['dismissal_kind'] != 'None') & df['dismissal_kind'].notna()
                df['wicket_type'] = df['dismissal_kind']  # For compatibility
            else:
                df['is_wicket'] = df.get('wicket_type', pd.Series()).notna()
            
            df['phase'] = df['over'].apply(self._determine_phase)
            df['powerplay'] = df['over'] <= 6.0
            
            # Add missing fields with defaults
            if 'batter_hand' not in df.columns:
                df['batter_hand'] = 'right'
            if 'bowler_type' not in df.columns:
                df['bowler_type'] = 'pace'
            if 'venue' not in df.columns:
                df['venue'] = 'Unknown Venue'
            if 'match_id' not in df.columns:
                df['match_id'] = range(1, len(df) + 1)
            if 'team_batting' not in df.columns:
                df['team_batting'] = 'Unknown Team'
            if 'team_bowling' not in df.columns:
                df['team_bowling'] = 'Unknown Team'
            
            self.stats['total_balls'] = len(df)
            logger.info(f"   ✅ Processed {len(df):,} ball records successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load cricket data: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources"""
        # Common column mappings for cricket data
        column_mappings = {
            # Batter variations
            'batsman': 'batter_name',
            'batsman_name': 'batter_name',
            'striker': 'batter_name',
            'striker_name': 'batter_name',
            # batter_name already exists - no mapping needed
            
            # Bowler variations  
            'bowler': 'bowler_name',
            # bowler_name already exists - no mapping needed
            
            # Runs variations
            'runs': 'runs_scored',
            'runs_off_bat': 'runs_scored',
            'batter_runs': 'runs_scored',
            'runs_batter': 'runs_scored',  # Cricsheet format
            'runs_total': 'total_runs',
            
            # Extras variations
            'extra_runs': 'extras',
            'runs_extras': 'extras',  # Cricsheet format
            'wides': 'extras',
            'byes': 'extras',
            
            # Over variations
            'over_number': 'over',  # Cricsheet format
            'over_id': 'over',
            
            # Venue variations (venue already exists)
            'ground': 'venue',
            'stadium': 'venue',
            'venue_name': 'venue',
            
            # Match variations
            'match': 'match_id',
            'game_id': 'match_id',
            'match_number': 'match_id',
            'source_match_id': 'match_id',  # Cricsheet format
            
            # Team variations
            'batting_team': 'team_batting',  # We have team_batting, not batting_team
            'bowling_team': 'team_bowling',  # We have team_bowling, not bowling_team
            
            # Dismissal variations
            'wicket_type': 'dismissal_kind',  # Cricsheet format
            'dismissal_type': 'dismissal_kind',
            'out_type': 'dismissal_kind'
        }
        
        # Apply mappings
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        return df
    
    def _determine_phase(self, over: float) -> str:
        """Determine match phase from over number"""
        if over <= 6.0:
            return "powerplay"
        elif over <= 15.0:
            return "middle_overs"
        else:
            return "death_overs"
    
    def _build_player_profiles(self, df: pd.DataFrame, progress_callback=None, should_cancel=None):
        """Build comprehensive player profiles from ball data using optimized vectorized operations"""
        logger.info("   👥 Processing player profiles...")
        
        # Vectorized approach for large datasets
        logger.info("   📊 Computing player statistics...")
        
        # Get player counts using value_counts (much faster than manual filtering)
        batter_counts = df['batter_name'].value_counts()
        bowler_counts = df['bowler_name'].value_counts()
        
        # Get all unique players
        all_players = set(batter_counts.index) | set(bowler_counts.index)
        total_players = len(all_players)
        logger.info(f"   🔍 Found {total_players:,} unique players to process")
        
        # Progress tracking
        processed = 0
        batch_size = max(1, total_players // 20)  # Process in 5% increments
        
        # Get player attributes efficiently
        batter_attrs = df.groupby('batter_name').agg({
            'batter_hand': 'first',
            'team_batting': lambda x: list(x.unique()),
            'match_id': 'nunique'
        }).to_dict('index')
        
        bowler_attrs = df.groupby('bowler_name').agg({
            'bowler_type': 'first', 
            'team_bowling': lambda x: list(x.unique()),
            'match_id': 'nunique'
        }).to_dict('index')
        
        # Combined match counts per player
        all_matches = df.groupby(['match_id']).agg({
            'batter_name': lambda x: list(x.unique()),
            'bowler_name': lambda x: list(x.unique())
        })
        
        for player_name in all_players:
            if should_cancel and should_cancel():
                logger.info("   🛑 Player profile building cancelled")
                return
                
            # Get counts efficiently
            batting_balls = batter_counts.get(player_name, 0)
            bowling_balls = bowler_counts.get(player_name, 0)
            
            # Determine primary role
            if batting_balls > bowling_balls * 3:
                primary_role = "batsman"
            elif bowling_balls > batting_balls * 3:
                primary_role = "bowler"
            else:
                primary_role = "all-rounder"
            
            # Get attributes efficiently
            hand = "right"  # Default
            bowling_style = None
            teams = set()
            matches_played = 0
            
            if player_name in batter_attrs:
                hand = batter_attrs[player_name].get('batter_hand', 'right')
                teams.update(batter_attrs[player_name].get('team_batting', []))
                matches_played = max(matches_played, batter_attrs[player_name].get('match_id', 0))
            
            if player_name in bowler_attrs:
                bowling_style = bowler_attrs[player_name].get('bowler_type')
                teams.update(bowler_attrs[player_name].get('team_bowling', []))
                matches_played = max(matches_played, bowler_attrs[player_name].get('match_id', 0))
            
            # Create player profile
            profile = PlayerProfile(
                name=player_name,
                player_id=player_name.lower().replace(' ', '_'),
                primary_role=primary_role,
                hand=hand,
                bowling_style=bowling_style,
                matches_played=matches_played,
                teams=list(teams)
            )
            
            self.players[player_name] = profile
            processed += 1
            
            # Progress updates
            if processed % batch_size == 0 or processed == total_players:
                progress_pct = 25 + (processed / total_players) * 10  # 25-35% of total progress
                if progress_callback:
                    try:
                        progress_callback(
                            "player_profiles",
                            f"Processed {processed:,}/{total_players:,} player profiles",
                            int(progress_pct),
                            {"players_processed": processed, "total_players": total_players}
                        )
                    except Exception:
                        pass
                logger.info(f"   ⚡ Processed {processed:,}/{total_players:,} players ({processed/total_players*100:.1f}%)")
        
        self.stats['total_players'] = len(self.players)
        logger.info(f"   ✅ Created {len(self.players):,} player profiles")
    
    def _process_ball_events(self, df: pd.DataFrame, progress_callback: Optional[Callable[[str, str, int, Dict[str, Any]], None]] = None, should_cancel: Optional[Callable[[], bool]] = None):
        """Process individual ball events"""
        logger.info("   ⚾ Creating ball event objects...")
        total_rows = len(df)
        # Compute an update interval to avoid spamming: ~50 updates across the dataset
        update_every = max(100000, total_rows // 50) if total_rows > 0 else 100000
        last_percent = 35

        for idx, row in df.iterrows():
            if should_cancel and should_cancel():
                raise RuntimeError("Canceled")
            innings_val = int(row['innings']) if 'innings' in row and not pd.isna(row['innings']) else (
                int(row['innings_index']) + 1 if 'innings_index' in row and not pd.isna(row['innings_index']) else 1
            )

            ball_in_over_val = (
                int(row['delivery_index']) + 1 if 'delivery_index' in row and not pd.isna(row['delivery_index']) else
                int(row['ball_in_over']) if 'ball_in_over' in row and not pd.isna(row['ball_in_over']) else 1
            )

            over_val = float(row['over']) if 'over' in row and not pd.isna(row['over']) else (
                float(row['over_number']) if 'over_number' in row and not pd.isna(row['over_number']) else 0.0
            )

            ball_event = BallEvent(
                ball_id=f"{row['match_id']}_ball_{idx}",
                match_id=str(row['match_id']),
                innings=innings_val,
                over=over_val,
                ball_in_over=ball_in_over_val,
                
                batter=row['batter_name'],
                bowler=row['bowler_name'],
                
                runs_scored=int(row['runs_scored']),
                extras=int(row['extras']),
                is_wicket=bool(row['is_wicket']),
                wicket_type=row.get('wicket_type'),
                
                team_score=int(row.get('team_score', 0)),
                team_wickets=int(row.get('team_wickets', 0)),
                run_rate=float(row.get('run_rate', 0)),
                required_rate=float(row.get('req_run_rate', 0)),
                powerplay=row['powerplay'],
                phase=row['phase'],
                
                batter_hand=row.get('batter_hand', 'right'),
                bowler_type=row.get('bowler_type', 'pace'),
                
                venue=row.get('venue', ''),
                competition=row.get('competition_name', ''),
                date=row.get('date')
            )
            
            self.balls.append(ball_event)

            # Periodic progress update during heavy loop
            if progress_callback is not None and ((idx + 1) % update_every == 0 or (idx + 1) == total_rows):
                try:
                    # Map progress within [35, 75]
                    frac = (idx + 1) / max(1, total_rows)
                    percent = 35 + int(40 * frac)
                    if percent > last_percent:
                        last_percent = percent
                    progress_callback(
                        "ball_events_progress",
                        f"Processed {idx + 1:,}/{total_rows:,} balls",
                        last_percent,
                        {"balls_processed": int(idx + 1), "balls_total": int(total_rows)}
                    )
                except Exception:
                    pass
        
        logger.info(f"   ✅ Processed {len(self.balls):,} ball events")
    
    def _build_graph_structure(self, df: pd.DataFrame = None):
        """Build the actual graph structure with nodes and relationships using optimized DataFrame operations"""
        logger.info("   🕸️ Adding nodes and relationships...")
        
        # Add player nodes
        logger.info("   👥 Adding player nodes...")
        for player_name, profile in self.players.items():
            self.graph.add_node(
                player_name,
                type='player',
                primary_role=profile.primary_role,
                hand=profile.hand,
                bowling_style=profile.bowling_style,
                matches_played=profile.matches_played,
                teams=profile.teams
            )
        
        # Add venue nodes using efficient DataFrame operations
        logger.info("   🏟️ Adding venue nodes...")
        if df is not None and 'venue' in df.columns:
            # Use DataFrame for efficient venue processing
            venue_stats = df.groupby('venue').agg({
                'match_id': 'nunique',
                'total_runs': ['count', 'mean']  # count = balls_played, mean = avg_score
            }).round(2)
            
            venue_stats.columns = ['matches_played', 'balls_played', 'avg_score']
            
            for venue in venue_stats.index:
                if pd.notna(venue) and venue:  # Skip null/empty venues
                    stats = venue_stats.loc[venue]
                    
                    # Base venue attributes
                    venue_attrs = {
                        'type': 'venue',
                        'matches_played': int(stats['matches_played']),
                        'balls_played': int(stats['balls_played']),
                        'avg_score': float(stats['avg_score']) if pd.notna(stats['avg_score']) else 0.0
                    }
                    
                    # Add enrichment data if available
                    venue_enrichment = self._find_venue_enrichment(venue)
                    if venue_enrichment:
                        venue_attrs.update({
                            'latitude': venue_enrichment.get('latitude', 0.0),
                            'longitude': venue_enrichment.get('longitude', 0.0),
                            'timezone': venue_enrichment.get('timezone', ''),
                            'city': venue_enrichment.get('city', ''),
                            'country': venue_enrichment.get('country', ''),
                            'has_enrichment': True
                        })
                        self.stats['enriched_matches'] += 1
                    else:
                        venue_attrs['has_enrichment'] = False
                    
                    self.graph.add_node(venue, **venue_attrs)
                    
                    # Store in venues dict
                    self.venues[venue] = {
                        'matches': int(stats['matches_played']),
                        'balls': int(stats['balls_played']),
                        'avg_score': float(stats['avg_score']) if pd.notna(stats['avg_score']) else 0.0,
                        'has_enrichment': venue_enrichment is not None
                    }
            
            self.stats['total_venues'] = len(venue_stats)
            logger.info(f"   ✅ Added {len(venue_stats)} venue nodes")
        else:
            logger.warning("   ⚠️ No venue data available or DataFrame not provided")
            self.stats['total_venues'] = 0
        
        # Add match nodes using efficient DataFrame operations
        logger.info("   🏏 Adding match nodes...")
        if df is not None:
            match_stats = df.groupby('match_id').agg({
                'venue': 'first',
                'competition': 'first',
                'match_id': 'count'  # count = total_balls
            }).rename(columns={'match_id': 'total_balls'})
            
            for match_id in match_stats.index:
                if pd.notna(match_id):
                    stats = match_stats.loc[match_id]
                    self.graph.add_node(
                        f"match_{match_id}",
                        type='match',
                        match_id=match_id,
                        total_balls=int(stats['total_balls']),
                        venue=stats['venue'] if pd.notna(stats['venue']) else '',
                        competition=stats['competition'] if pd.notna(stats['competition']) else ''
                    )
            
            self.stats['total_matches'] = len(match_stats)
            logger.info(f"   ✅ Added {len(match_stats)} match nodes")
        else:
            logger.warning("   ⚠️ No match data available or DataFrame not provided")
            self.stats['total_matches'] = 0
        
        # Add relationships using optimized approach
        self._add_relationships(df)
    
    def _add_relationships(self, df: pd.DataFrame = None):
        """Add edges between nodes using optimized DataFrame operations"""
        logger.info("   🔗 Adding relationships...")
        
        if df is None:
            logger.warning("   ⚠️ No DataFrame provided for relationship building")
            return
        
        # Player-Venue relationships using efficient aggregation
        logger.info("   🏟️ Creating player-venue relationships...")
        
        # Batter-Venue relationships
        if 'venue' in df.columns and 'batter_name' in df.columns:
            # Build aggregation dict based on available columns
            agg_dict = {'batter_name': 'count'}  # balls faced
            
            # Check for runs scored columns (different datasets may have different names)
            runs_col = None
            for col_name in ['runs_scored', 'runs', 'runs_off_bat', 'batter_runs', 'total_runs']:
                if col_name in df.columns:
                    runs_col = col_name
                    agg_dict[col_name] = 'sum'
                    break
            
            batter_venue_stats = df.groupby(['batter_name', 'venue']).agg(agg_dict).rename(columns={'batter_name': 'balls'})
            
            for (batter, venue), stats in batter_venue_stats.iterrows():
                if batter in self.graph and venue in self.graph:
                    edge_attrs = {
                        'relationship': 'played_at',
                        'balls': int(stats['balls'])
                    }
                    
                    # Add runs if available
                    if runs_col and runs_col in stats.index:
                        edge_attrs['runs'] = int(stats[runs_col]) if pd.notna(stats[runs_col]) else 0
                    else:
                        edge_attrs['runs'] = 0
                    
                    self.graph.add_edge(batter, venue, **edge_attrs)
        
        # Bowler-Venue relationships
        if 'venue' in df.columns and 'bowler_name' in df.columns:
            # Build aggregation dict based on available columns
            agg_dict = {'bowler_name': 'count'}  # balls bowled
            
            # Check for runs conceded columns (different datasets may have different names)
            runs_col = None
            for col_name in ['runs_conceded', 'bowler_runs', 'runs_off_bowler', 'runs_total', 'total_runs']:
                if col_name in df.columns:
                    runs_col = col_name
                    agg_dict[col_name] = 'sum'
                    break
            
            # Check for wicket columns
            wicket_col = None
            for col_name in ['is_wicket', 'wicket', 'dismissal', 'wicket_taken']:
                if col_name in df.columns:
                    wicket_col = col_name
                    agg_dict[col_name] = 'sum'
                    break
            
            bowler_venue_stats = df.groupby(['bowler_name', 'venue']).agg(agg_dict).rename(columns={'bowler_name': 'balls'})
            
            for (bowler, venue), stats in bowler_venue_stats.iterrows():
                if bowler in self.graph and venue in self.graph:
                    edge_attrs = {
                        'relationship': 'bowled_at',
                        'balls': int(stats['balls'])
                    }
                    
                    # Add runs if available
                    if runs_col and runs_col in stats.index:
                        edge_attrs['runs'] = int(stats[runs_col]) if pd.notna(stats[runs_col]) else 0
                    else:
                        edge_attrs['runs'] = 0
                    
                    # Add wickets if available
                    if wicket_col and wicket_col in stats.index:
                        edge_attrs['wickets'] = int(stats[wicket_col]) if pd.notna(stats[wicket_col]) else 0
                    else:
                        edge_attrs['wickets'] = 0
                    
                    self.graph.add_edge(bowler, venue, **edge_attrs)
        
        logger.info(f"   ✅ Added {self.graph.number_of_edges():,} relationships")
    
    def _compute_situational_stats(self, df: pd.DataFrame = None):
        """Compute comprehensive situational statistics for each player using optimized DataFrame operations"""
        logger.info("   📈 Computing advanced situational statistics...")
        
        if df is None:
            logger.warning("   ⚠️ No DataFrame provided for situational statistics - skipping advanced stats")
            return
        
        logger.info("   🎯 Computing comprehensive batting analysis (powerplay, pace vs spin, pressure situations)...")
        
        # Compute comprehensive batting statistics using efficient aggregation
        if 'batter_name' in df.columns:
            # Find boundary columns flexibly
            four_col = None
            for col_name in ['is_four', 'four', 'fours', 'boundary_four', 'is_boundary_four']:
                if col_name in df.columns:
                    four_col = col_name
                    break
            
            six_col = None
            for col_name in ['is_six', 'six', 'sixes', 'boundary_six', 'is_boundary_six']:
                if col_name in df.columns:
                    six_col = col_name
                    break
            
            # Build aggregation dictionary dynamically
            agg_dict = {}
            if 'runs_scored' in df.columns:
                agg_dict['runs_scored'] = ['sum', 'count', 'mean']
            else:
                agg_dict['batter_name'] = 'count'  # fallback to count balls
            
            if four_col:
                agg_dict[four_col] = 'sum'
            if six_col:
                agg_dict[six_col] = 'sum'
            
            # Basic batting stats
            batting_stats = df.groupby('batter_name').agg(agg_dict).round(2)
            
            # Update player profiles with basic batting stats
            for player_name in batting_stats.index:
                if player_name in self.players:
                    try:
                        stats = batting_stats.loc[player_name]
                        if 'runs_scored' in df.columns:
                            self.players[player_name].batting_stats = {
                                'runs': int(stats[('runs_scored', 'sum')]) if pd.notna(stats[('runs_scored', 'sum')]) else 0,
                                'balls': int(stats[('runs_scored', 'count')]) if pd.notna(stats[('runs_scored', 'count')]) else 0,
                                'average': float(stats[('runs_scored', 'mean')]) if pd.notna(stats[('runs_scored', 'mean')]) else 0.0,
                                'fours': int(stats[(four_col, 'sum')]) if four_col and pd.notna(stats[(four_col, 'sum')]) else 0,
                                'sixes': int(stats[(six_col, 'sum')]) if six_col and pd.notna(stats[(six_col, 'sum')]) else 0
                            }
                        else:
                            # Fallback when runs_scored not available
                            self.players[player_name].batting_stats = {
                                'runs': 0,
                                'balls': int(stats[('batter_name', 'count')]) if pd.notna(stats[('batter_name', 'count')]) else 0,
                                'average': 0.0,
                                'fours': int(stats[(four_col, 'sum')]) if four_col and pd.notna(stats[(four_col, 'sum')]) else 0,
                                'sixes': int(stats[(six_col, 'sum')]) if six_col and pd.notna(stats[(six_col, 'sum')]) else 0
                            }
                    except Exception as e:
                        # Skip problematic stats
                        pass
            
            # === POWERPLAY vs MIDDLE vs DEATH OVERS ANALYSIS ===
            logger.info("   ⚡ Computing powerplay vs middle vs death overs performance...")
            
            # Identify phase columns
            phase_col = None
            for col_name in ['phase', 'match_phase', 'over_phase', 'powerplay_phase']:
                if col_name in df.columns:
                    phase_col = col_name
                    break
            
            over_col = None
            for col_name in ['over', 'over_number', 'current_over']:
                if col_name in df.columns:
                    over_col = col_name
                    break
            
            # Create phase categories if not present
            if phase_col is None and over_col is not None:
                df = df.copy()
                df['computed_phase'] = df[over_col].apply(
                    lambda x: 'powerplay' if x <= 6 else ('death_overs' if x >= 16 else 'middle_overs')
                )
                phase_col = 'computed_phase'
            
            if phase_col and 'runs_scored' in df.columns:
                # Build phase aggregation dict
                phase_agg_dict = {
                    'runs_scored': ['sum', 'count', 'mean']
                }
                if four_col:
                    phase_agg_dict[four_col] = 'sum'
                if six_col:
                    phase_agg_dict[six_col] = 'sum'
                
                phase_stats = df.groupby(['batter_name', phase_col]).agg(phase_agg_dict).round(2)
                
                for player_name in phase_stats.index.get_level_values(0).unique():
                    if player_name in self.players:
                        player_phases = phase_stats.loc[player_name] if player_name in phase_stats.index else None
                        if player_phases is not None:
                            try:
                                # Powerplay stats
                                if 'powerplay' in player_phases.index:
                                    pp_stats = player_phases.loc['powerplay']
                                    self.players[player_name].in_powerplay = {
                                        'runs': int(pp_stats[('runs_scored', 'sum')]) if pd.notna(pp_stats[('runs_scored', 'sum')]) else 0,
                                        'balls': int(pp_stats[('runs_scored', 'count')]) if pd.notna(pp_stats[('runs_scored', 'count')]) else 0,
                                        'strike_rate': float(pp_stats[('runs_scored', 'sum')] / pp_stats[('runs_scored', 'count')] * 100) if pp_stats[('runs_scored', 'count')] > 0 else 0.0,
                                        'fours': int(pp_stats[(four_col, 'sum')]) if four_col and pd.notna(pp_stats[(four_col, 'sum')]) else 0,
                                        'sixes': int(pp_stats[(six_col, 'sum')]) if six_col and pd.notna(pp_stats[(six_col, 'sum')]) else 0
                                    }
                                
                                # Death overs stats
                                if 'death_overs' in player_phases.index:
                                    death_stats = player_phases.loc['death_overs']
                                    self.players[player_name].in_death_overs = {
                                        'runs': int(death_stats[('runs_scored', 'sum')]) if pd.notna(death_stats[('runs_scored', 'sum')]) else 0,
                                        'balls': int(death_stats[('runs_scored', 'count')]) if pd.notna(death_stats[('runs_scored', 'count')]) else 0,
                                        'strike_rate': float(death_stats[('runs_scored', 'sum')] / death_stats[('runs_scored', 'count')] * 100) if death_stats[('runs_scored', 'count')] > 0 else 0.0,
                                        'fours': int(death_stats[(four_col, 'sum')]) if four_col and pd.notna(death_stats[(four_col, 'sum')]) else 0,
                                        'sixes': int(death_stats[(six_col, 'sum')]) if six_col and pd.notna(death_stats[(six_col, 'sum')]) else 0
                                    }
                            except Exception:
                                # Skip problematic phase stats
                                pass
            
            # === PACE vs SPIN BOWLING ANALYSIS ===
            logger.info("   🎳 Computing pace vs spin bowling performance...")
            
            bowler_type_col = None
            for col_name in ['bowler_type', 'bowling_style', 'bowler_style', 'bowling_type']:
                if col_name in df.columns:
                    bowler_type_col = col_name
                    break
            
            if bowler_type_col and 'runs_scored' in df.columns:
                # Create pace/spin categories
                df_copy = df.copy()
                df_copy['bowling_category'] = df_copy[bowler_type_col].apply(
                    lambda x: 'pace' if any(word in str(x).lower() for word in ['pace', 'fast', 'medium', 'quick']) 
                    else 'spin' if any(word in str(x).lower() for word in ['spin', 'leg', 'off', 'left-arm orthodox', 'googly'])
                    else 'unknown'
                )
                
                # Filter out unknown categories for cleaner analysis
                pace_spin_df = df_copy[df_copy['bowling_category'].isin(['pace', 'spin'])]
                
                if not pace_spin_df.empty:
                    # Build pace/spin aggregation dict
                    pace_spin_agg_dict = {
                        'runs_scored': ['sum', 'count', 'mean']
                    }
                    if four_col:
                        pace_spin_agg_dict[four_col] = 'sum'
                    if six_col:
                        pace_spin_agg_dict[six_col] = 'sum'
                    
                    pace_spin_stats = pace_spin_df.groupby(['batter_name', 'bowling_category']).agg(pace_spin_agg_dict).round(2)
                    
                    for player_name in pace_spin_stats.index.get_level_values(0).unique():
                        if player_name in self.players:
                            player_bowling_types = pace_spin_stats.loc[player_name] if player_name in pace_spin_stats.index else None
                            if player_bowling_types is not None:
                                try:
                                    # Vs Pace stats
                                    if 'pace' in player_bowling_types.index:
                                        pace_stats = player_bowling_types.loc['pace']
                                        self.players[player_name].vs_pace = {
                                            'runs': int(pace_stats[('runs_scored', 'sum')]) if pd.notna(pace_stats[('runs_scored', 'sum')]) else 0,
                                            'balls': int(pace_stats[('runs_scored', 'count')]) if pd.notna(pace_stats[('runs_scored', 'count')]) else 0,
                                            'average': float(pace_stats[('runs_scored', 'sum')] / pace_stats[('runs_scored', 'count')]) if pace_stats[('runs_scored', 'count')] > 0 else 0.0,
                                            'strike_rate': float(pace_stats[('runs_scored', 'sum')] / pace_stats[('runs_scored', 'count')] * 100) if pace_stats[('runs_scored', 'count')] > 0 else 0.0,
                                            'fours': int(pace_stats[(four_col, 'sum')]) if four_col and pd.notna(pace_stats[(four_col, 'sum')]) else 0,
                                            'sixes': int(pace_stats[(six_col, 'sum')]) if six_col and pd.notna(pace_stats[(six_col, 'sum')]) else 0
                                        }
                                    
                                    # Vs Spin stats
                                    if 'spin' in player_bowling_types.index:
                                        spin_stats = player_bowling_types.loc['spin']
                                        self.players[player_name].vs_spin = {
                                            'runs': int(spin_stats[('runs_scored', 'sum')]) if pd.notna(spin_stats[('runs_scored', 'sum')]) else 0,
                                            'balls': int(spin_stats[('runs_scored', 'count')]) if pd.notna(spin_stats[('runs_scored', 'count')]) else 0,
                                            'average': float(spin_stats[('runs_scored', 'sum')] / spin_stats[('runs_scored', 'count')]) if spin_stats[('runs_scored', 'count')] > 0 else 0.0,
                                            'strike_rate': float(spin_stats[('runs_scored', 'sum')] / spin_stats[('runs_scored', 'count')] * 100) if spin_stats[('runs_scored', 'count')] > 0 else 0.0,
                                            'fours': int(spin_stats[(four_col, 'sum')]) if four_col and pd.notna(spin_stats[(four_col, 'sum')]) else 0,
                                            'sixes': int(spin_stats[(six_col, 'sum')]) if six_col and pd.notna(spin_stats[(six_col, 'sum')]) else 0
                                        }
                                except Exception:
                                    # Skip problematic pace/spin stats
                                    pass
            
            # === PRESSURE SITUATION ANALYSIS ===
            logger.info("   🔥 Computing pressure situation performance...")
            
            # Identify pressure situations based on available columns
            pressure_indicators = []
            
            # Check for match situation columns
            if 'required_run_rate' in df.columns:
                pressure_indicators.append('required_run_rate')
            if 'current_run_rate' in df.columns:
                pressure_indicators.append('current_run_rate')
            if 'wickets_lost' in df.columns:
                pressure_indicators.append('wickets_lost')
            if 'balls_remaining' in df.columns:
                pressure_indicators.append('balls_remaining')
            
            # Create pressure situation categories
            if pressure_indicators and 'runs_scored' in df.columns:
                df_pressure = df.copy()
                
                # Define pressure based on available indicators
                pressure_conditions = []
                if 'required_run_rate' in df.columns and 'current_run_rate' in df.columns:
                    pressure_conditions.append(df_pressure['required_run_rate'] > df_pressure['current_run_rate'] + 2)
                if 'wickets_lost' in df.columns:
                    pressure_conditions.append(df_pressure['wickets_lost'] >= 6)  # 6+ wickets down
                if 'balls_remaining' in df.columns:
                    pressure_conditions.append(df_pressure['balls_remaining'] <= 30)  # Last 5 overs
                if over_col:
                    pressure_conditions.append(df_pressure[over_col] >= 16)  # Death overs
                
                if pressure_conditions:
                    # Combine pressure conditions (any one indicates pressure)
                    df_pressure['is_pressure'] = False
                    for condition in pressure_conditions:
                        df_pressure['is_pressure'] = df_pressure['is_pressure'] | condition
                    
                    # Build pressure aggregation dict
                    pressure_agg_dict = {
                        'runs_scored': ['sum', 'count', 'mean']
                    }
                    if four_col:
                        pressure_agg_dict[four_col] = 'sum'
                    if six_col:
                        pressure_agg_dict[six_col] = 'sum'
                    
                    pressure_stats = df_pressure.groupby(['batter_name', 'is_pressure']).agg(pressure_agg_dict).round(2)
                    
                    for player_name in pressure_stats.index.get_level_values(0).unique():
                        if player_name in self.players:
                            player_pressure = pressure_stats.loc[player_name] if player_name in pressure_stats.index else None
                            if player_pressure is not None:
                                try:
                                    # Under pressure stats
                                    if True in player_pressure.index:
                                        pressure_stats_data = player_pressure.loc[True]
                                        self.players[player_name].under_pressure = {
                                            'runs': int(pressure_stats_data[('runs_scored', 'sum')]) if pd.notna(pressure_stats_data[('runs_scored', 'sum')]) else 0,
                                            'balls': int(pressure_stats_data[('runs_scored', 'count')]) if pd.notna(pressure_stats_data[('runs_scored', 'count')]) else 0,
                                            'average': float(pressure_stats_data[('runs_scored', 'sum')] / pressure_stats_data[('runs_scored', 'count')]) if pressure_stats_data[('runs_scored', 'count')] > 0 else 0.0,
                                            'strike_rate': float(pressure_stats_data[('runs_scored', 'sum')] / pressure_stats_data[('runs_scored', 'count')] * 100) if pressure_stats_data[('runs_scored', 'count')] > 0 else 0.0,
                                            'fours': int(pressure_stats_data[(four_col, 'sum')]) if four_col and pd.notna(pressure_stats_data[(four_col, 'sum')]) else 0,
                                            'sixes': int(pressure_stats_data[(six_col, 'sum')]) if six_col and pd.notna(pressure_stats_data[(six_col, 'sum')]) else 0
                                        }
                                except Exception:
                                    # Skip problematic pressure stats
                                    pass
        
        # Compute basic bowling statistics using efficient aggregation  
        if 'bowler_name' in df.columns:
            # Find available runs columns for bowling
            runs_col = None
            for col_name in ['runs_conceded', 'bowler_runs', 'runs_off_bowler', 'runs_total', 'total_runs']:
                if col_name in df.columns:
                    runs_col = col_name
                    break
                    
            wicket_col = None
            for col_name in ['is_wicket', 'wicket', 'dismissal', 'wicket_taken']:
                if col_name in df.columns:
                    wicket_col = col_name
                    break
            
            # Build aggregation dict
            agg_dict = {'bowler_name': 'count'}  # balls bowled
            if runs_col:
                agg_dict[runs_col] = 'sum'
            if wicket_col:
                agg_dict[wicket_col] = 'sum'
                
            bowling_stats = df.groupby('bowler_name').agg(agg_dict).round(2)
            bowling_stats = bowling_stats.rename(columns={'bowler_name': 'balls'})
            
            # Update player profiles with bowling stats
            for player_name in bowling_stats.index:
                if player_name in self.players:
                    try:
                        stats = bowling_stats.loc[player_name]
                        self.players[player_name].bowling_stats = {
                            'balls': int(stats['balls']) if pd.notna(stats['balls']) else 0,
                            'runs': int(stats[runs_col]) if runs_col and pd.notna(stats[runs_col]) else 0,
                            'wickets': int(stats[wicket_col]) if wicket_col and pd.notna(stats[wicket_col]) else 0
                        }
                    except Exception:
                        # Skip problematic stats
                        pass
        
            # === VENUE-SPECIFIC PERFORMANCE ===
            logger.info("   🏟️ Computing venue-specific performance...")
            
            venue_col = None
            for col_name in ['venue', 'ground', 'stadium', 'location']:
                if col_name in df.columns:
                    venue_col = col_name
                    break
            
            if venue_col and 'runs_scored' in df.columns:
                # Build venue aggregation dict
                venue_agg_dict = {
                    'runs_scored': ['sum', 'count', 'mean']
                }
                if four_col:
                    venue_agg_dict[four_col] = 'sum'
                if six_col:
                    venue_agg_dict[six_col] = 'sum'
                
                venue_stats = df.groupby(['batter_name', venue_col]).agg(venue_agg_dict).round(2)
                
                for player_name in venue_stats.index.get_level_values(0).unique():
                    if player_name in self.players:
                        player_venues = venue_stats.loc[player_name] if player_name in venue_stats.index else None
                        if player_venues is not None:
                            try:
                                # Store venue-specific stats
                                venue_performance = {}
                                for venue in player_venues.index:
                                    venue_data = player_venues.loc[venue]
                                    venue_performance[venue] = {
                                        'runs': int(venue_data[('runs_scored', 'sum')]) if pd.notna(venue_data[('runs_scored', 'sum')]) else 0,
                                        'balls': int(venue_data[('runs_scored', 'count')]) if pd.notna(venue_data[('runs_scored', 'count')]) else 0,
                                        'average': float(venue_data[('runs_scored', 'sum')] / venue_data[('runs_scored', 'count')]) if venue_data[('runs_scored', 'count')] > 0 else 0.0,
                                        'strike_rate': float(venue_data[('runs_scored', 'sum')] / venue_data[('runs_scored', 'count')] * 100) if venue_data[('runs_scored', 'count')] > 0 else 0.0,
                                        'fours': int(venue_data[(four_col, 'sum')]) if four_col and pd.notna(venue_data[(four_col, 'sum')]) else 0,
                                        'sixes': int(venue_data[(six_col, 'sum')]) if six_col and pd.notna(venue_data[(six_col, 'sum')]) else 0
                                    }
                                self.players[player_name].by_venue = venue_performance
                            except Exception:
                                # Skip problematic venue stats
                                pass
            
            # === UPDATE GRAPH NODES WITH COMPUTED STATS ===
            logger.info("   📊 Updating graph nodes with comprehensive statistics...")
            
            for player_name, profile in self.players.items():
                if player_name in self.graph:
                    node_attrs = {
                        'batting_stats': getattr(profile, 'batting_stats', {}),
                        'bowling_stats': getattr(profile, 'bowling_stats', {}),
                        'vs_pace': getattr(profile, 'vs_pace', {}),
                        'vs_spin': getattr(profile, 'vs_spin', {}),
                        'in_powerplay': getattr(profile, 'in_powerplay', {}),
                        'in_death_overs': getattr(profile, 'in_death_overs', {}),
                        'under_pressure': getattr(profile, 'under_pressure', {}),
                        'by_venue': getattr(profile, 'by_venue', {})
                    }
                    self.graph.nodes[player_name].update(node_attrs)
        
        logger.info("   ✅ Computed comprehensive situational statistics for all players")
        logger.info("   🎯 Advanced analytics ready: powerplay/death overs, pace vs spin, pressure situations, venue performance")
    
    def _compute_batting_stats(self, balls: List[BallEvent]) -> Dict[str, float]:
        """Compute batting statistics from ball events"""
        if not balls:
            return {}
        
        total_runs = sum(b.runs_scored for b in balls)
        total_balls = len(balls)
        boundaries = sum(1 for b in balls if b.runs_scored == 4)
        sixes = sum(1 for b in balls if b.runs_scored == 6)
        wickets = sum(1 for b in balls if b.is_wicket)
        
        return {
            'runs': total_runs,
            'balls': total_balls,
            'average': total_runs / max(wickets, 1),
            'strike_rate': (total_runs / total_balls * 100) if total_balls > 0 else 0,
            'boundaries': boundaries,
            'sixes': sixes,
            'dismissals': wickets
        }
    
    def _compute_bowling_stats(self, balls: List[BallEvent]) -> Dict[str, float]:
        """Compute bowling statistics from ball events"""
        if not balls:
            return {}
        
        runs_conceded = sum(b.runs_scored + b.extras for b in balls)
        total_balls = len(balls)
        wickets = sum(1 for b in balls if b.is_wicket)
        
        return {
            'balls_bowled': total_balls,
            'runs_conceded': runs_conceded,
            'wickets': wickets,
            'average': runs_conceded / max(wickets, 1),
            'economy': (runs_conceded / total_balls * 6) if total_balls > 0 else 0,
            'strike_rate': total_balls / max(wickets, 1)
        }
    
    def _add_graph_metadata(self):
        """Add metadata to the graph"""
        self.graph.graph.update({
            'type': 'unified_cricket_kg',
            'version': '2.0',
            'created': datetime.now().isoformat(),
            'statistics': self.stats,
            'node_types': {
                'player': sum(1 for n, d in self.graph.nodes(data=True) if d.get('type') == 'player'),
                'venue': sum(1 for n, d in self.graph.nodes(data=True) if d.get('type') == 'venue'),
                'match': sum(1 for n, d in self.graph.nodes(data=True) if d.get('type') == 'match')
            }
        })
    
    def save_graph(self, output_path: str):
        """Save the knowledge graph to disk"""
        logger.info(f"💾 Saving knowledge graph to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info("✅ Knowledge graph saved successfully")
    
    def save_player_profiles(self, output_path: str):
        """Save player profiles as JSON for analysis"""
        logger.info(f"💾 Saving player profiles to {output_path}")
        
        profiles_data = {}
        for name, profile in self.players.items():
            profiles_data[name] = asdict(profile)
        
        with open(output_path, 'w') as f:
            json.dump(profiles_data, f, indent=2, default=str)
        
        logger.info("✅ Player profiles saved successfully")


def main():
    """Example usage of the Unified KG Builder"""
    # This would be called from the admin interface
    builder = UnifiedKGBuilder("/path/to/data")
    
    # Build from available data (auto-detects)
    graph = builder.build_from_available_data()
    
    # Save outputs
    builder.save_graph("models/unified_cricket_kg.pkl")
    builder.save_player_profiles("reports/player_profiles.json")
    
    return graph


if __name__ == "__main__":
    main()
