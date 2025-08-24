# Purpose: Enhanced Knowledge Graph API with natural language query interface
# Author: WicketWise Team, Last Modified: 2025-08-23

"""
Enhanced Knowledge Graph API that provides:
- Natural language query processing
- Cypher query generation and optimization
- Temporal decay integration
- Context-aware responses
- Caching layer for performance
- Confidence scoring for results

This API enables cricket analysts to query the knowledge graph using natural language
and get sophisticated, context-aware responses with temporal weighting.
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json

import networkx as nx
import numpy as np

from .temporal_decay import TemporalDecayEngine, DecayType, PerformanceEvent
from .context_nodes import ContextNodeManager, ContextNodeData

logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context for knowledge graph queries"""
    user_intent: str
    entity_types: List[str]
    time_range: Optional[Tuple[datetime, datetime]] = None
    context_filters: Optional[Dict[str, str]] = None
    confidence_threshold: float = 0.7
    max_results: int = 100
    include_temporal_decay: bool = True


@dataclass
class QueryResult:
    """Result from knowledge graph query"""
    query_id: str
    results: List[Dict[str, Any]]
    confidence_score: float
    execution_time_ms: float
    cypher_query: str
    temporal_weights_applied: bool
    context_nodes_used: List[str]
    metadata: Dict[str, Any]


class NaturalLanguageProcessor:
    """Process natural language queries into structured queries"""
    
    def __init__(self):
        self.intent_patterns = {
            "player_performance": [
                r"how (?:well )?did (.+) perform",
                r"(.+)'s performance",
                r"stats for (.+)",
                r"(.+) batting average",
                r"(.+) bowling figures"
            ],
            "head_to_head": [
                r"(.+) vs (.+)",
                r"(.+) against (.+)",
                r"head to head (.+) (.+)",
                r"matchup between (.+) and (.+)"
            ],
            "team_analysis": [
                r"how did (.+) team perform",
                r"(.+) team stats",
                r"(.+) team performance in (.+)"
            ],
            "venue_analysis": [
                r"(.+) at (.+)",
                r"performance at (.+)",
                r"(.+) venue stats",
                r"how do teams perform at (.+)"
            ],
            "condition_analysis": [
                r"performance in (.+) conditions",
                r"(.+) in (.+) weather",
                r"(.+) on (.+) pitches",
                r"(.+) during (.+) matches"
            ],
            "temporal_analysis": [
                r"(.+) recent form",
                r"(.+) last (\d+) matches",
                r"(.+) this season",
                r"(.+) trend over time"
            ]
        }
        
        self.entity_extractors = {
            "player": self._extract_player_names,
            "team": self._extract_team_names,
            "venue": self._extract_venue_names,
            "condition": self._extract_conditions
        }
    
    def process_query(self, natural_query: str) -> QueryContext:
        """Process natural language query into structured query context"""
        query_lower = natural_query.lower().strip()
        
        # Determine intent
        intent = self._classify_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query_lower, intent)
        
        # Extract time context
        time_range = self._extract_time_range(query_lower)
        
        # Extract context filters
        context_filters = self._extract_context_filters(query_lower)
        
        return QueryContext(
            user_intent=intent,
            entity_types=list(entities.keys()),
            time_range=time_range,
            context_filters=context_filters,
            confidence_threshold=0.7,
            max_results=100,
            include_temporal_decay=True
        )
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the query"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        
        return "general_query"
    
    def _extract_entities(self, query: str, intent: str) -> Dict[str, List[str]]:
        """Extract entities from query based on intent"""
        entities = {}
        
        for entity_type, extractor in self.entity_extractors.items():
            extracted = extractor(query, intent)
            if extracted:
                entities[entity_type] = extracted
        
        return entities
    
    def _extract_player_names(self, query: str, intent: str) -> List[str]:
        """Extract player names from query"""
        # Common player name patterns
        player_patterns = [
            r"\b(kohli|virat)\b",
            r"\b(dhoni|ms)\b", 
            r"\b(rohit|sharma)\b",
            r"\b(bumrah|jasprit)\b",
            r"\b(hardik|pandya)\b"
        ]
        
        players = []
        for pattern in player_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            players.extend(matches)
        
        return list(set(players))
    
    def _extract_team_names(self, query: str, intent: str) -> List[str]:
        """Extract team names from query"""
        team_patterns = [
            r"\b(rcb|royal challengers)\b",
            r"\b(csk|chennai super kings)\b",
            r"\b(mi|mumbai indians)\b",
            r"\b(kkr|kolkata knight riders)\b",
            r"\b(india|indian team)\b",
            r"\b(australia|aussies)\b"
        ]
        
        teams = []
        for pattern in team_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            teams.extend(matches)
        
        return list(set(teams))
    
    def _extract_venue_names(self, query: str, intent: str) -> List[str]:
        """Extract venue names from query"""
        venue_patterns = [
            r"\b(wankhede|wankhede stadium)\b",
            r"\b(eden gardens)\b",
            r"\b(mcg|melbourne cricket ground)\b",
            r"\b(lord's|lords)\b",
            r"\b(chinnaswamy|m chinnaswamy)\b"
        ]
        
        venues = []
        for pattern in venue_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            venues.extend(matches)
        
        return list(set(venues))
    
    def _extract_conditions(self, query: str, intent: str) -> List[str]:
        """Extract condition keywords from query"""
        condition_patterns = [
            r"\b(overcast|cloudy)\b",
            r"\b(rain|wet)\b",
            r"\b(clear|sunny)\b",
            r"\b(day|night)\b",
            r"\b(final|playoff|qualifier)\b",
            r"\b(batting friendly|bowling friendly|balanced)\b"
        ]
        
        conditions = []
        for pattern in condition_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            conditions.extend(matches)
        
        return list(set(conditions))
    
    def _extract_time_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract time range from query"""
        now = datetime.now()
        
        # Recent patterns
        if re.search(r"\blast (\d+) (days?|weeks?|months?)\b", query):
            match = re.search(r"\blast (\d+) (days?|weeks?|months?)\b", query)
            if match:
                number = int(match.group(1))
                unit = match.group(2)
                
                if "day" in unit:
                    start_date = now - timedelta(days=number)
                elif "week" in unit:
                    start_date = now - timedelta(weeks=number)
                elif "month" in unit:
                    start_date = now - timedelta(days=number * 30)
                else:
                    start_date = now - timedelta(days=30)
                
                return (start_date, now)
        
        # Season patterns
        if "this season" in query or "current season" in query:
            # Assume season started 6 months ago
            return (now - timedelta(days=180), now)
        
        if "last season" in query:
            # Previous season (6-12 months ago)
            return (now - timedelta(days=365), now - timedelta(days=180))
        
        return None
    
    def _extract_context_filters(self, query: str) -> Dict[str, str]:
        """Extract context filters from query"""
        filters = {}
        
        # Format filters
        if re.search(r"\bt20\b", query, re.IGNORECASE):
            filters["format"] = "t20"
        elif re.search(r"\bodi\b", query, re.IGNORECASE):
            filters["format"] = "odi"
        elif re.search(r"\btest\b", query, re.IGNORECASE):
            filters["format"] = "test"
        
        # Venue type filters
        if re.search(r"\bhome\b", query, re.IGNORECASE):
            filters["venue_type"] = "home"
        elif re.search(r"\baway\b", query, re.IGNORECASE):
            filters["venue_type"] = "away"
        
        # Tournament filters
        if re.search(r"\bipl\b", query, re.IGNORECASE):
            filters["tournament"] = "ipl"
        elif re.search(r"\bworld cup\b", query, re.IGNORECASE):
            filters["tournament"] = "world_cup"
        
        return filters


class CypherQueryGenerator:
    """Generate and optimize Cypher queries for knowledge graph"""
    
    def __init__(self):
        self.query_templates = {
            "player_performance": """
                MATCH (p:Player {name: $player_name})-[r:PERFORMED_IN]->(m:Match)
                WHERE m.date >= $start_date AND m.date <= $end_date
                OPTIONAL MATCH (m)-[:HAS_CONTEXT]->(c:Context)
                RETURN p, r, m, c
                ORDER BY m.date DESC
                LIMIT $max_results
            """,
            "head_to_head": """
                MATCH (p1:Player {name: $player1})-[r1:PERFORMED_IN]->(m:Match)<-[r2:PERFORMED_IN]-(p2:Player {name: $player2})
                WHERE m.date >= $start_date AND m.date <= $end_date
                OPTIONAL MATCH (m)-[:HAS_CONTEXT]->(c:Context)
                RETURN p1, r1, p2, r2, m, c
                ORDER BY m.date DESC
                LIMIT $max_results
            """,
            "venue_analysis": """
                MATCH (v:Venue {name: $venue_name})<-[:PLAYED_AT]-(m:Match)
                WHERE m.date >= $start_date AND m.date <= $end_date
                OPTIONAL MATCH (m)-[:HAS_CONTEXT]->(c:Context)
                OPTIONAL MATCH (m)<-[:PERFORMED_IN]-(p:Player)
                RETURN v, m, c, p
                ORDER BY m.date DESC
                LIMIT $max_results
            """,
            "condition_analysis": """
                MATCH (c:Context {type: $condition_type})<-[:HAS_CONTEXT]-(m:Match)
                WHERE m.date >= $start_date AND m.date <= $end_date
                OPTIONAL MATCH (m)<-[:PERFORMED_IN]-(p:Player)
                RETURN c, m, p
                ORDER BY m.date DESC
                LIMIT $max_results
            """
        }
    
    def generate_cypher_query(self, context: QueryContext, entities: Dict[str, List[str]]) -> str:
        """Generate Cypher query based on context and entities"""
        intent = context.user_intent
        
        if intent in self.query_templates:
            template = self.query_templates[intent]
            
            # Customize template based on entities and filters
            query = self._customize_template(template, context, entities)
            
            return query
        else:
            # Generate generic query
            return self._generate_generic_query(context, entities)
    
    def _customize_template(self, template: str, context: QueryContext, entities: Dict[str, List[str]]) -> str:
        """Customize query template with specific parameters"""
        # Add context filters to WHERE clause
        if context.context_filters:
            additional_filters = []
            for key, value in context.context_filters.items():
                additional_filters.append(f"m.{key} = '{value}'")
            
            if additional_filters:
                filter_clause = " AND " + " AND ".join(additional_filters)
                template = template.replace("LIMIT $max_results", filter_clause + "\nLIMIT $max_results")
        
        return template
    
    def _generate_generic_query(self, context: QueryContext, entities: Dict[str, List[str]]) -> str:
        """Generate generic query for unrecognized intents"""
        base_query = """
            MATCH (n)
            WHERE n.date >= $start_date AND n.date <= $end_date
        """
        
        # Add entity filters
        if entities:
            entity_filters = []
            for entity_type, entity_list in entities.items():
                if entity_list:
                    entity_filter = f"n.{entity_type} IN {entity_list}"
                    entity_filters.append(entity_filter)
            
            if entity_filters:
                base_query += " AND " + " AND ".join(entity_filters)
        
        base_query += """
            RETURN n
            ORDER BY n.date DESC
            LIMIT $max_results
        """
        
        return base_query


class EnhancedKGQueryEngine:
    """Enhanced Knowledge Graph query engine with temporal decay and context awareness"""
    
    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph = graph or nx.DiGraph()
        self.nlp_processor = NaturalLanguageProcessor()
        self.cypher_generator = CypherQueryGenerator()
        self.temporal_engine = TemporalDecayEngine()
        self.context_manager = ContextNodeManager()
        
        # Query cache for performance
        self.query_cache: Dict[str, QueryResult] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Performance metrics
        self.query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_execution_time_ms": 0,
            "intent_distribution": {},
            "entity_usage": {}
        }
    
    async def query(self, natural_query: str, **kwargs) -> QueryResult:
        """
        Execute natural language query against knowledge graph
        
        Args:
            natural_query: Natural language query string
            **kwargs: Additional query parameters
            
        Returns:
            QueryResult with results and metadata
        """
        start_time = time.time()
        query_id = f"query_{int(time.time() * 1000)}"
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(natural_query, kwargs)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.query_stats["cache_hits"] += 1
                logger.info(f"🎯 Cache hit for query: {natural_query[:50]}...")
                return cached_result
            
            # Process natural language query
            query_context = self.nlp_processor.process_query(natural_query)
            
            # Override with kwargs if provided
            if "confidence_threshold" in kwargs:
                query_context.confidence_threshold = kwargs["confidence_threshold"]
            if "max_results" in kwargs:
                query_context.max_results = kwargs["max_results"]
            
            # Extract entities for Cypher generation
            entities = self._extract_entities_from_graph(query_context)
            
            # Generate Cypher query
            cypher_query = self.cypher_generator.generate_cypher_query(query_context, entities)
            
            # Execute query against graph
            raw_results = self._execute_graph_query(query_context, entities)
            
            # Apply temporal decay if requested
            temporal_weights_applied = False
            if query_context.include_temporal_decay and raw_results:
                raw_results = self._apply_temporal_decay(raw_results, query_context)
                temporal_weights_applied = True
            
            # Apply context filtering
            context_nodes_used = []
            if query_context.context_filters:
                raw_results, context_nodes_used = self._apply_context_filtering(raw_results, query_context)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(raw_results, query_context)
            
            # Filter by confidence threshold
            filtered_results = [
                result for result in raw_results 
                if result.get("confidence", 1.0) >= query_context.confidence_threshold
            ]
            
            # Limit results
            final_results = filtered_results[:query_context.max_results]
            
            # Create result object
            execution_time_ms = (time.time() - start_time) * 1000
            
            result = QueryResult(
                query_id=query_id,
                results=final_results,
                confidence_score=confidence_score,
                execution_time_ms=execution_time_ms,
                cypher_query=cypher_query,
                temporal_weights_applied=temporal_weights_applied,
                context_nodes_used=context_nodes_used,
                metadata={
                    "intent": query_context.user_intent,
                    "entity_types": query_context.entity_types,
                    "total_raw_results": len(raw_results),
                    "total_filtered_results": len(filtered_results),
                    "cache_key": cache_key
                }
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update statistics
            self._update_query_stats(query_context, execution_time_ms)
            
            logger.info(
                f"✅ Query executed: {natural_query[:50]}... "
                f"({len(final_results)} results, {execution_time_ms:.1f}ms, confidence={confidence_score:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Query failed: {natural_query[:50]}... Error: {e}")
            
            execution_time_ms = (time.time() - start_time) * 1000
            return QueryResult(
                query_id=query_id,
                results=[],
                confidence_score=0.0,
                execution_time_ms=execution_time_ms,
                cypher_query="",
                temporal_weights_applied=False,
                context_nodes_used=[],
                metadata={"error": str(e), "query": natural_query}
            )
    
    def _execute_graph_query(self, context: QueryContext, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Execute query against NetworkX graph"""
        results = []
        
        # For now, implement basic graph traversal
        # In production, this would use a proper graph database
        
        if context.user_intent == "player_performance" and "player" in entities:
            for player_name in entities["player"]:
                player_nodes = [
                    (node_id, data) for node_id, data in self.graph.nodes(data=True)
                    if data.get("type") == "player" and player_name.lower() in data.get("name", "").lower()
                ]
                
                for node_id, node_data in player_nodes:
                    # Get connected performance data
                    neighbors = list(self.graph.neighbors(node_id))
                    performance_data = {
                        "player_id": node_id,
                        "player_name": node_data.get("name"),
                        "connected_matches": len(neighbors),
                        "node_data": node_data,
                        "confidence": 0.9
                    }
                    results.append(performance_data)
        
        elif context.user_intent == "venue_analysis" and "venue" in entities:
            for venue_name in entities["venue"]:
                venue_nodes = [
                    (node_id, data) for node_id, data in self.graph.nodes(data=True)
                    if data.get("type") == "venue" and venue_name.lower() in data.get("name", "").lower()
                ]
                
                for node_id, node_data in venue_nodes:
                    venue_data = {
                        "venue_id": node_id,
                        "venue_name": node_data.get("name"),
                        "node_data": node_data,
                        "confidence": 0.8
                    }
                    results.append(venue_data)
        
        else:
            # Generic node search
            for node_id, node_data in self.graph.nodes(data=True):
                # Apply basic filtering
                include_node = True
                
                if context.context_filters:
                    for filter_key, filter_value in context.context_filters.items():
                        if node_data.get(filter_key) != filter_value:
                            include_node = False
                            break
                
                if include_node:
                    result_data = {
                        "node_id": node_id,
                        "node_data": node_data,
                        "confidence": 0.7
                    }
                    results.append(result_data)
        
        return results
    
    def _apply_temporal_decay(self, results: List[Dict[str, Any]], context: QueryContext) -> List[Dict[str, Any]]:
        """Apply temporal decay weighting to results"""
        if not context.time_range:
            return results
        
        weighted_results = []
        reference_date = context.time_range[1] if context.time_range else datetime.now()
        
        for result in results:
            # Extract date from result
            result_date = self._extract_date_from_result(result)
            if result_date:
                days_ago = (reference_date - result_date).total_seconds() / (24 * 3600)
                
                # Calculate temporal weight using exponential decay
                decay_config = self.temporal_engine.default_config
                decay_function = self.temporal_engine.decay_functions[DecayType.EXPONENTIAL]
                temporal_weight = decay_function.calculate_weight(days_ago)
                
                # Apply temporal weight to confidence
                original_confidence = result.get("confidence", 1.0)
                weighted_confidence = original_confidence * temporal_weight
                
                result["confidence"] = weighted_confidence
                result["temporal_weight"] = temporal_weight
                result["days_ago"] = days_ago
            
            weighted_results.append(result)
        
        # Sort by weighted confidence
        weighted_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return weighted_results
    
    def _apply_context_filtering(self, results: List[Dict[str, Any]], 
                               context: QueryContext) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Apply context-based filtering using context nodes"""
        if not context.context_filters:
            return results, []
        
        filtered_results = []
        context_nodes_used = []
        
        for result in results:
            include_result = True
            
            # Check context filters
            for filter_key, filter_value in context.context_filters.items():
                node_data = result.get("node_data", {})
                if node_data.get(filter_key) != filter_value:
                    include_result = False
                    break
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results, context_nodes_used
    
    def _calculate_confidence_score(self, results: List[Dict[str, Any]], context: QueryContext) -> float:
        """Calculate overall confidence score for query results"""
        if not results:
            return 0.0
        
        # Base confidence on individual result confidences
        individual_confidences = [result.get("confidence", 0.5) for result in results]
        avg_confidence = np.mean(individual_confidences)
        
        # Adjust based on result count
        result_count_factor = min(1.0, len(results) / 10.0)  # Plateau at 10 results
        
        # Adjust based on entity match quality
        entity_match_factor = 1.0  # Placeholder for entity matching quality
        
        overall_confidence = avg_confidence * result_count_factor * entity_match_factor
        
        return min(1.0, overall_confidence)
    
    def _extract_date_from_result(self, result: Dict[str, Any]) -> Optional[datetime]:
        """Extract date from result for temporal decay"""
        node_data = result.get("node_data", {})
        
        # Try different date field names
        date_fields = ["date", "match_date", "created_at", "timestamp"]
        
        for field in date_fields:
            if field in node_data:
                date_value = node_data[field]
                if isinstance(date_value, datetime):
                    return date_value
                elif isinstance(date_value, str):
                    try:
                        return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    except:
                        continue
        
        return None
    
    def _extract_entities_from_graph(self, context: QueryContext) -> Dict[str, List[str]]:
        """Extract available entities from graph for query"""
        entities = {}
        
        # Extract players
        players = [
            data.get("name", node_id) for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "player"
        ]
        if players:
            entities["player"] = players[:20]  # Limit for performance
        
        # Extract teams
        teams = [
            data.get("name", node_id) for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "team"
        ]
        if teams:
            entities["team"] = teams[:20]
        
        # Extract venues
        venues = [
            data.get("name", node_id) for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "venue"
        ]
        if venues:
            entities["venue"] = venues[:20]
        
        return entities
    
    def _generate_cache_key(self, query: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        key_components = [query.lower().strip()]
        
        # Add sorted kwargs to ensure consistent cache keys
        for key, value in sorted(kwargs.items()):
            key_components.append(f"{key}={value}")
        
        return "|".join(key_components)
    
    def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached result if still valid"""
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            
            # Check if cache is still valid
            cache_age = time.time() - (cached_result.metadata.get("cached_at", 0))
            if cache_age < self.cache_ttl_seconds:
                return cached_result
            else:
                # Remove expired cache entry
                del self.query_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: QueryResult) -> None:
        """Cache query result"""
        result.metadata["cached_at"] = time.time()
        self.query_cache[cache_key] = result
        
        # Limit cache size
        if len(self.query_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.query_cache.keys(),
                key=lambda k: self.query_cache[k].metadata.get("cached_at", 0)
            )[:100]
            
            for key in oldest_keys:
                del self.query_cache[key]
    
    def _update_query_stats(self, context: QueryContext, execution_time_ms: float) -> None:
        """Update query statistics"""
        self.query_stats["total_queries"] += 1
        
        # Update average execution time
        total_time = self.query_stats["avg_execution_time_ms"] * (self.query_stats["total_queries"] - 1)
        total_time += execution_time_ms
        self.query_stats["avg_execution_time_ms"] = total_time / self.query_stats["total_queries"]
        
        # Update intent distribution
        intent = context.user_intent
        self.query_stats["intent_distribution"][intent] = self.query_stats["intent_distribution"].get(intent, 0) + 1
        
        # Update entity usage
        for entity_type in context.entity_types:
            self.query_stats["entity_usage"][entity_type] = self.query_stats["entity_usage"].get(entity_type, 0) + 1
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query engine performance statistics"""
        cache_hit_rate = 0.0
        if self.query_stats["total_queries"] > 0:
            cache_hit_rate = self.query_stats["cache_hits"] / self.query_stats["total_queries"]
        
        return {
            **self.query_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.query_cache),
            "graph_stats": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges()
            }
        }
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def add_context_nodes_to_graph(self, match_data: Dict[str, Any]) -> None:
        """Add context nodes to the knowledge graph"""
        context_nodes = self.context_manager.extract_all_context_nodes(match_data)
        self.graph = self.context_manager.add_context_nodes_to_graph(self.graph, context_nodes)
        
        logger.info(f"Added {len(context_nodes)} context nodes to knowledge graph")
