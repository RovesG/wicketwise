# Purpose: Context node system for enhanced Knowledge Graph
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Context node system that adds rich contextual information to the cricket Knowledge Graph:
- Tournament stage nodes (group, knockout, final)
- Pitch type nodes (batting-friendly, bowling-friendly, balanced)
- Match importance nodes (regular, playoff, final)
- Weather condition nodes (clear, overcast, rain-affected)
- Time-of-day nodes (day, day-night, night)
- Season nodes (early, mid, late)

These context nodes enable more nuanced analysis of player and team performance
under different conditions and circumstances.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
import json

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class ContextNodeType(Enum):
    """Types of context nodes"""
    TOURNAMENT_STAGE = "tournament_stage"
    PITCH_TYPE = "pitch_type"
    MATCH_IMPORTANCE = "match_importance"
    WEATHER_CONDITION = "weather_condition"
    TIME_OF_DAY = "time_of_day"
    SEASON_PHASE = "season_phase"
    VENUE_CHARACTERISTICS = "venue_characteristics"
    TEAM_COMPOSITION = "team_composition"


class TournamentStage(Enum):
    """Tournament stage classifications"""
    GROUP_STAGE = "group_stage"
    LEAGUE_STAGE = "league_stage"
    QUALIFIER = "qualifier"
    ELIMINATOR = "eliminator"
    SEMI_FINAL = "semi_final"
    FINAL = "final"
    SUPER_OVER = "super_over"


class PitchType(Enum):
    """Pitch type classifications"""
    BATTING_PARADISE = "batting_paradise"      # Very high scoring
    BATTING_FRIENDLY = "batting_friendly"      # High scoring
    BALANCED = "balanced"                      # Moderate scoring
    BOWLING_FRIENDLY = "bowling_friendly"      # Low scoring
    BOWLER_PARADISE = "bowler_paradise"        # Very low scoring
    TURNING_TRACK = "turning_track"            # Spin-friendly
    GREEN_TOP = "green_top"                    # Pace-friendly
    SLOW_LOW = "slow_low"                      # Slow and low bounce


class MatchImportance(Enum):
    """Match importance levels"""
    PRACTICE = "practice"                      # Practice matches
    REGULAR = "regular"                        # Regular season
    CRUCIAL = "crucial"                        # Important for qualification
    PLAYOFF = "playoff"                        # Playoff matches
    FINAL = "final"                           # Championship final
    WORLD_CUP = "world_cup"                   # World Cup matches
    RIVALRY = "rivalry"                       # High-profile rivalries


class WeatherCondition(Enum):
    """Weather condition classifications"""
    CLEAR = "clear"                           # Clear skies
    PARTLY_CLOUDY = "partly_cloudy"           # Some clouds
    OVERCAST = "overcast"                     # Heavily clouded
    DRIZZLE = "drizzle"                       # Light rain
    RAIN_AFFECTED = "rain_affected"           # Rain interruptions
    WINDY = "windy"                          # High wind conditions
    HOT = "hot"                              # Extremely hot
    HUMID = "humid"                          # High humidity
    DEW_FACTOR = "dew_factor"                # Evening dew


class TimeOfDay(Enum):
    """Time of day classifications"""
    MORNING = "morning"                       # 09:00 - 12:00
    AFTERNOON = "afternoon"                   # 12:00 - 16:00
    EVENING = "evening"                       # 16:00 - 19:00
    NIGHT = "night"                          # 19:00 - 23:00
    DAY_NIGHT = "day_night"                  # Matches spanning day-night


@dataclass
class ContextNodeData:
    """Data structure for context nodes"""
    node_id: str
    node_type: ContextNodeType
    properties: Dict[str, Union[str, float, int, bool]]
    relationships: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class ContextNodeExtractor(ABC):
    """Abstract base class for context node extractors"""
    
    @abstractmethod
    def extract_context_nodes(self, match_data: Dict[str, any]) -> List[ContextNodeData]:
        """Extract context nodes from match data"""
        pass
    
    @abstractmethod
    def get_node_relationships(self, node: ContextNodeData, 
                              other_nodes: List[ContextNodeData]) -> List[Tuple[str, str, str]]:
        """Get relationships between nodes"""
        pass


class TournamentStageExtractor(ContextNodeExtractor):
    """Extract tournament stage context nodes"""
    
    def __init__(self):
        self.stage_keywords = {
            TournamentStage.SEMI_FINAL: ["semi final", "semifinal", "semi-final"],
            TournamentStage.FINAL: [" final", "championship final", "grand final"],
            TournamentStage.QUALIFIER: ["qualifier", "q1", "q2"],
            TournamentStage.ELIMINATOR: ["eliminator", "elimination"],
            TournamentStage.SUPER_OVER: ["super over", "tie-breaker"],
            TournamentStage.GROUP_STAGE: ["group", "round robin"],
            TournamentStage.LEAGUE_STAGE: ["league match", "regular season"]
        }
    
    def extract_context_nodes(self, match_data: Dict[str, any]) -> List[ContextNodeData]:
        """Extract tournament stage nodes"""
        nodes = []
        
        # Determine tournament stage
        stage = self._classify_tournament_stage(match_data)
        
        if stage:
            node = ContextNodeData(
                node_id=f"tournament_stage_{stage.value}",
                node_type=ContextNodeType.TOURNAMENT_STAGE,
                properties={
                    "stage": stage.value,
                    "stage_importance": self._get_stage_importance(stage),
                    "elimination_match": stage in [TournamentStage.QUALIFIER,
                                                  TournamentStage.ELIMINATOR, 
                                                  TournamentStage.SEMI_FINAL, 
                                                  TournamentStage.FINAL],
                    "pressure_level": self._get_pressure_level(stage)
                },
                metadata={
                    "tournament": match_data.get("tournament", ""),
                    "match_type": match_data.get("match_type", "")
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _classify_tournament_stage(self, match_data: Dict[str, any]) -> Optional[TournamentStage]:
        """Classify tournament stage based on match data"""
        match_description = " ".join([
            str(match_data.get("tournament", "")),
            str(match_data.get("match_type", "")),
            str(match_data.get("stage", "")),
            str(match_data.get("description", ""))
        ]).lower()
        
        for stage, keywords in self.stage_keywords.items():
            if any(keyword in match_description for keyword in keywords):
                return stage
        
        # Default classification based on tournament structure
        if " final" in match_description or match_description.endswith("final"):
            return TournamentStage.FINAL
        elif any(term in match_description for term in ["playoff", "knockout"]):
            return TournamentStage.QUALIFIER
        else:
            return TournamentStage.LEAGUE_STAGE
    
    def _get_stage_importance(self, stage: TournamentStage) -> float:
        """Get numerical importance score for stage"""
        importance_scores = {
            TournamentStage.GROUP_STAGE: 1.0,
            TournamentStage.LEAGUE_STAGE: 1.0,
            TournamentStage.QUALIFIER: 2.0,
            TournamentStage.ELIMINATOR: 2.5,
            TournamentStage.SEMI_FINAL: 3.0,
            TournamentStage.FINAL: 4.0,
            TournamentStage.SUPER_OVER: 4.5
        }
        return importance_scores.get(stage, 1.0)
    
    def _get_pressure_level(self, stage: TournamentStage) -> float:
        """Get pressure level for stage (0-1 scale)"""
        pressure_levels = {
            TournamentStage.GROUP_STAGE: 0.3,
            TournamentStage.LEAGUE_STAGE: 0.4,
            TournamentStage.QUALIFIER: 0.7,
            TournamentStage.ELIMINATOR: 0.8,
            TournamentStage.SEMI_FINAL: 0.9,
            TournamentStage.FINAL: 1.0,
            TournamentStage.SUPER_OVER: 1.0
        }
        return pressure_levels.get(stage, 0.5)
    
    def get_node_relationships(self, node: ContextNodeData, 
                              other_nodes: List[ContextNodeData]) -> List[Tuple[str, str, str]]:
        """Get tournament stage relationships"""
        relationships = []
        
        # Tournament stages can be related to match importance
        for other_node in other_nodes:
            if other_node.node_type == ContextNodeType.MATCH_IMPORTANCE:
                relationships.append((
                    node.node_id, 
                    "influences_importance", 
                    other_node.node_id
                ))
        
        return relationships


class PitchTypeExtractor(ContextNodeExtractor):
    """Extract pitch type context nodes"""
    
    def extract_context_nodes(self, match_data: Dict[str, any]) -> List[ContextNodeData]:
        """Extract pitch type nodes"""
        nodes = []
        
        # Classify pitch type based on match statistics
        pitch_type = self._classify_pitch_type(match_data)
        
        if pitch_type:
            node = ContextNodeData(
                node_id=f"pitch_type_{pitch_type.value}",
                node_type=ContextNodeType.PITCH_TYPE,
                properties={
                    "pitch_type": pitch_type.value,
                    "batting_difficulty": self._get_batting_difficulty(pitch_type),
                    "bowling_advantage": self._get_bowling_advantage(pitch_type),
                    "spin_factor": self._get_spin_factor(pitch_type),
                    "pace_factor": self._get_pace_factor(pitch_type),
                    "expected_score": self._get_expected_score(pitch_type)
                },
                metadata={
                    "venue": match_data.get("venue", ""),
                    "total_runs": match_data.get("total_runs", 0),
                    "total_wickets": match_data.get("total_wickets", 0)
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _classify_pitch_type(self, match_data: Dict[str, any]) -> Optional[PitchType]:
        """Classify pitch type based on match statistics"""
        total_runs = match_data.get("total_runs", 0)
        total_overs = match_data.get("total_overs", 20)
        total_wickets = match_data.get("total_wickets", 0)
        
        if total_overs == 0:
            return None
        
        run_rate = total_runs / total_overs
        wicket_rate = total_wickets / total_overs
        
        # Classification logic
        if run_rate > 12 and wicket_rate < 0.4:
            return PitchType.BATTING_PARADISE
        elif run_rate > 9 and wicket_rate < 0.6:
            return PitchType.BATTING_FRIENDLY
        elif run_rate < 6 or wicket_rate > 1.2:
            return PitchType.BOWLING_FRIENDLY
        elif run_rate < 4:
            return PitchType.BOWLER_PARADISE
        else:
            return PitchType.BALANCED
    
    def _get_batting_difficulty(self, pitch_type: PitchType) -> float:
        """Get batting difficulty score (0-1, higher = more difficult)"""
        difficulty_scores = {
            PitchType.BATTING_PARADISE: 0.1,
            PitchType.BATTING_FRIENDLY: 0.3,
            PitchType.BALANCED: 0.5,
            PitchType.BOWLING_FRIENDLY: 0.7,
            PitchType.BOWLER_PARADISE: 0.9,
            PitchType.TURNING_TRACK: 0.6,
            PitchType.GREEN_TOP: 0.8,
            PitchType.SLOW_LOW: 0.7
        }
        return difficulty_scores.get(pitch_type, 0.5)
    
    def _get_bowling_advantage(self, pitch_type: PitchType) -> float:
        """Get bowling advantage score (0-1, higher = more advantage)"""
        advantage_scores = {
            PitchType.BATTING_PARADISE: 0.1,
            PitchType.BATTING_FRIENDLY: 0.2,
            PitchType.BALANCED: 0.5,
            PitchType.BOWLING_FRIENDLY: 0.8,
            PitchType.BOWLER_PARADISE: 1.0,
            PitchType.TURNING_TRACK: 0.7,
            PitchType.GREEN_TOP: 0.8,
            PitchType.SLOW_LOW: 0.6
        }
        return advantage_scores.get(pitch_type, 0.5)
    
    def _get_spin_factor(self, pitch_type: PitchType) -> float:
        """Get spin bowling factor (0-1)"""
        spin_factors = {
            PitchType.BATTING_PARADISE: 0.2,
            PitchType.BATTING_FRIENDLY: 0.3,
            PitchType.BALANCED: 0.5,
            PitchType.BOWLING_FRIENDLY: 0.6,
            PitchType.BOWLER_PARADISE: 0.7,
            PitchType.TURNING_TRACK: 1.0,
            PitchType.GREEN_TOP: 0.2,
            PitchType.SLOW_LOW: 0.8
        }
        return spin_factors.get(pitch_type, 0.5)
    
    def _get_pace_factor(self, pitch_type: PitchType) -> float:
        """Get pace bowling factor (0-1)"""
        pace_factors = {
            PitchType.BATTING_PARADISE: 0.3,
            PitchType.BATTING_FRIENDLY: 0.4,
            PitchType.BALANCED: 0.5,
            PitchType.BOWLING_FRIENDLY: 0.7,
            PitchType.BOWLER_PARADISE: 0.8,
            PitchType.TURNING_TRACK: 0.3,
            PitchType.GREEN_TOP: 1.0,
            PitchType.SLOW_LOW: 0.4
        }
        return pace_factors.get(pitch_type, 0.5)
    
    def _get_expected_score(self, pitch_type: PitchType) -> int:
        """Get expected T20 score for pitch type"""
        expected_scores = {
            PitchType.BATTING_PARADISE: 220,
            PitchType.BATTING_FRIENDLY: 180,
            PitchType.BALANCED: 160,
            PitchType.BOWLING_FRIENDLY: 140,
            PitchType.BOWLER_PARADISE: 120,
            PitchType.TURNING_TRACK: 150,
            PitchType.GREEN_TOP: 145,
            PitchType.SLOW_LOW: 135
        }
        return expected_scores.get(pitch_type, 160)
    
    def get_node_relationships(self, node: ContextNodeData, 
                              other_nodes: List[ContextNodeData]) -> List[Tuple[str, str, str]]:
        """Get pitch type relationships"""
        relationships = []
        
        # Pitch type affects weather conditions
        for other_node in other_nodes:
            if other_node.node_type == ContextNodeType.WEATHER_CONDITION:
                relationships.append((
                    node.node_id, 
                    "interacts_with", 
                    other_node.node_id
                ))
        
        return relationships


class WeatherConditionExtractor(ContextNodeExtractor):
    """Extract weather condition context nodes"""
    
    def extract_context_nodes(self, match_data: Dict[str, any]) -> List[ContextNodeData]:
        """Extract weather condition nodes"""
        nodes = []
        
        weather_conditions = self._extract_weather_conditions(match_data)
        
        for condition in weather_conditions:
            node = ContextNodeData(
                node_id=f"weather_{condition.value}",
                node_type=ContextNodeType.WEATHER_CONDITION,
                properties={
                    "condition": condition.value,
                    "batting_impact": self._get_batting_impact(condition),
                    "bowling_impact": self._get_bowling_impact(condition),
                    "fielding_impact": self._get_fielding_impact(condition),
                    "visibility_factor": self._get_visibility_factor(condition),
                    "swing_factor": self._get_swing_factor(condition)
                },
                metadata={
                    "temperature": match_data.get("temperature"),
                    "humidity": match_data.get("humidity"),
                    "wind_speed": match_data.get("wind_speed")
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _extract_weather_conditions(self, match_data: Dict[str, any]) -> List[WeatherCondition]:
        """Extract weather conditions from match data"""
        conditions = []
        
        # Check for specific weather indicators
        weather_desc = str(match_data.get("weather_description", "")).lower()
        temperature = match_data.get("temperature", 25)
        humidity = match_data.get("humidity", 50)
        wind_speed = match_data.get("wind_speed", 0)
        
        # Primary condition
        if "rain" in weather_desc or match_data.get("rain_affected", False):
            conditions.append(WeatherCondition.RAIN_AFFECTED)
        elif "overcast" in weather_desc or "cloudy" in weather_desc:
            conditions.append(WeatherCondition.OVERCAST)
        elif "clear" in weather_desc or "sunny" in weather_desc:
            conditions.append(WeatherCondition.CLEAR)
        else:
            conditions.append(WeatherCondition.PARTLY_CLOUDY)
        
        # Additional conditions
        if temperature > 35:
            conditions.append(WeatherCondition.HOT)
        if humidity > 70:
            conditions.append(WeatherCondition.HUMID)
        if wind_speed > 20:
            conditions.append(WeatherCondition.WINDY)
        if match_data.get("dew_factor", False):
            conditions.append(WeatherCondition.DEW_FACTOR)
        
        return conditions
    
    def _get_batting_impact(self, condition: WeatherCondition) -> float:
        """Get batting impact score (-1 to 1, negative = harder)"""
        impact_scores = {
            WeatherCondition.CLEAR: 0.2,
            WeatherCondition.PARTLY_CLOUDY: 0.0,
            WeatherCondition.OVERCAST: -0.3,
            WeatherCondition.DRIZZLE: -0.5,
            WeatherCondition.RAIN_AFFECTED: -0.7,
            WeatherCondition.WINDY: -0.2,
            WeatherCondition.HOT: -0.1,
            WeatherCondition.HUMID: -0.2,
            WeatherCondition.DEW_FACTOR: -0.4
        }
        return impact_scores.get(condition, 0.0)
    
    def _get_bowling_impact(self, condition: WeatherCondition) -> float:
        """Get bowling impact score (-1 to 1, positive = advantage)"""
        impact_scores = {
            WeatherCondition.CLEAR: -0.1,
            WeatherCondition.PARTLY_CLOUDY: 0.0,
            WeatherCondition.OVERCAST: 0.4,
            WeatherCondition.DRIZZLE: 0.5,
            WeatherCondition.RAIN_AFFECTED: 0.3,
            WeatherCondition.WINDY: 0.2,
            WeatherCondition.HOT: 0.0,
            WeatherCondition.HUMID: 0.1,
            WeatherCondition.DEW_FACTOR: 0.6
        }
        return impact_scores.get(condition, 0.0)
    
    def _get_fielding_impact(self, condition: WeatherCondition) -> float:
        """Get fielding impact score (-1 to 1, negative = harder)"""
        impact_scores = {
            WeatherCondition.CLEAR: 0.2,
            WeatherCondition.PARTLY_CLOUDY: 0.0,
            WeatherCondition.OVERCAST: -0.1,
            WeatherCondition.DRIZZLE: -0.6,
            WeatherCondition.RAIN_AFFECTED: -0.8,
            WeatherCondition.WINDY: -0.3,
            WeatherCondition.HOT: -0.2,
            WeatherCondition.HUMID: -0.1,
            WeatherCondition.DEW_FACTOR: -0.5
        }
        return impact_scores.get(condition, 0.0)
    
    def _get_visibility_factor(self, condition: WeatherCondition) -> float:
        """Get visibility factor (0-1, higher = better visibility)"""
        visibility_scores = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.PARTLY_CLOUDY: 0.9,
            WeatherCondition.OVERCAST: 0.7,
            WeatherCondition.DRIZZLE: 0.5,
            WeatherCondition.RAIN_AFFECTED: 0.3,
            WeatherCondition.WINDY: 0.8,
            WeatherCondition.HOT: 0.9,
            WeatherCondition.HUMID: 0.8,
            WeatherCondition.DEW_FACTOR: 0.7
        }
        return visibility_scores.get(condition, 0.8)
    
    def _get_swing_factor(self, condition: WeatherCondition) -> float:
        """Get ball swing factor (0-1, higher = more swing)"""
        swing_scores = {
            WeatherCondition.CLEAR: 0.2,
            WeatherCondition.PARTLY_CLOUDY: 0.3,
            WeatherCondition.OVERCAST: 0.8,
            WeatherCondition.DRIZZLE: 0.9,
            WeatherCondition.RAIN_AFFECTED: 0.7,
            WeatherCondition.WINDY: 0.4,
            WeatherCondition.HOT: 0.1,
            WeatherCondition.HUMID: 0.6,
            WeatherCondition.DEW_FACTOR: 0.9
        }
        return swing_scores.get(condition, 0.4)
    
    def get_node_relationships(self, node: ContextNodeData, 
                              other_nodes: List[ContextNodeData]) -> List[Tuple[str, str, str]]:
        """Get weather condition relationships"""
        relationships = []
        
        # Weather affects time of day matches differently
        for other_node in other_nodes:
            if other_node.node_type == ContextNodeType.TIME_OF_DAY:
                relationships.append((
                    node.node_id, 
                    "affects_timing", 
                    other_node.node_id
                ))
        
        return relationships


class ContextNodeManager:
    """Manager for all context node operations"""
    
    def __init__(self):
        self.extractors = {
            ContextNodeType.TOURNAMENT_STAGE: TournamentStageExtractor(),
            ContextNodeType.PITCH_TYPE: PitchTypeExtractor(),
            ContextNodeType.WEATHER_CONDITION: WeatherConditionExtractor(),
        }
        
        self.node_cache: Dict[str, ContextNodeData] = {}
        self.relationship_cache: List[Tuple[str, str, str]] = []
        
    def extract_all_context_nodes(self, match_data: Dict[str, any]) -> List[ContextNodeData]:
        """Extract all context nodes from match data"""
        all_nodes = []
        
        for extractor_type, extractor in self.extractors.items():
            try:
                nodes = extractor.extract_context_nodes(match_data)
                all_nodes.extend(nodes)
                logger.debug(f"Extracted {len(nodes)} {extractor_type.value} nodes")
            except Exception as e:
                logger.error(f"Failed to extract {extractor_type.value} nodes: {e}")
        
        # Cache nodes
        for node in all_nodes:
            self.node_cache[node.node_id] = node
        
        return all_nodes
    
    def build_context_relationships(self, nodes: List[ContextNodeData]) -> List[Tuple[str, str, str]]:
        """Build relationships between context nodes"""
        all_relationships = []
        
        for node in nodes:
            node_type = node.node_type
            if node_type in self.extractors:
                extractor = self.extractors[node_type]
                relationships = extractor.get_node_relationships(node, nodes)
                all_relationships.extend(relationships)
        
        # Cache relationships
        self.relationship_cache.extend(all_relationships)
        
        return all_relationships
    
    def add_context_nodes_to_graph(self, graph: nx.DiGraph, 
                                  context_nodes: List[ContextNodeData]) -> nx.DiGraph:
        """Add context nodes to existing knowledge graph"""
        
        # Add nodes
        for node in context_nodes:
            graph.add_node(
                node.node_id,
                node_type=node.node_type.value,
                **node.properties,
                metadata=node.metadata,
                created_at=node.created_at.isoformat()
            )
        
        # Add relationships
        relationships = self.build_context_relationships(context_nodes)
        for source, relation, target in relationships:
            if source in graph and target in graph:
                graph.add_edge(source, target, edge_type=relation)
        
        logger.info(f"Added {len(context_nodes)} context nodes and {len(relationships)} relationships to graph")
        
        return graph
    
    def get_context_summary(self) -> Dict[str, any]:
        """Get summary of context nodes and relationships"""
        node_counts = {}
        for node in self.node_cache.values():
            node_type = node.node_type.value
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        relationship_counts = {}
        for _, relation, _ in self.relationship_cache:
            relationship_counts[relation] = relationship_counts.get(relation, 0) + 1
        
        return {
            "total_context_nodes": len(self.node_cache),
            "node_types": node_counts,
            "total_relationships": len(self.relationship_cache),
            "relationship_types": relationship_counts,
            "extractors_available": [t.value for t in self.extractors.keys()]
        }
    
    def clear_cache(self):
        """Clear node and relationship caches"""
        self.node_cache.clear()
        self.relationship_cache.clear()
        logger.info("Context node caches cleared")