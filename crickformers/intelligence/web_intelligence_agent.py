# Purpose: Web Intelligence Agent for Real-time Cricket Information
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Web Intelligence Agent

A reusable agent that gathers real-time cricket information from web sources:
- Player injuries and fitness updates
- Recent form and performance trends  
- Team news and lineup changes
- Weather conditions and pitch reports
- Expert analysis and betting insights

Designed to be used by multiple betting agents and analysis systems.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class IntelligenceType(Enum):
    """Types of intelligence the agent can gather"""
    INJURY_STATUS = "injury_status"
    RECENT_FORM = "recent_form"
    TEAM_NEWS = "team_news"
    WEATHER_CONDITIONS = "weather_conditions"
    PITCH_REPORT = "pitch_report"
    EXPERT_ANALYSIS = "expert_analysis"
    BETTING_INSIGHTS = "betting_insights"

@dataclass
class IntelligenceResult:
    """Structured result from web intelligence gathering"""
    intelligence_type: IntelligenceType
    player_name: Optional[str] = None
    team_name: Optional[str] = None
    venue: Optional[str] = None
    content: str = ""
    confidence: float = 0.0
    source: str = ""
    timestamp: datetime = None
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class WebIntelligenceAgent:
    """
    Reusable Web Intelligence Agent for Cricket Information
    
    This agent can be used by:
    - Player Insight Agent
    - Betting Intelligence System
    - Match Analysis Tools
    - Strategy Recommendation Engines
    """
    
    def __init__(self, web_search_tool=None):
        """
        Initialize Web Intelligence Agent
        
        Args:
            web_search_tool: Web search function (injected dependency)
        """
        self.web_search = web_search_tool
        self.cache = {}  # Simple cache for recent searches
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes
        
        # Cricket-specific search patterns
        self.search_patterns = {
            IntelligenceType.INJURY_STATUS: [
                "{player} injury update",
                "{player} fitness status",
                "{player} ruled out injured",
                "{player} medical report"
            ],
            IntelligenceType.RECENT_FORM: [
                "{player} recent scores",
                "{player} last 5 matches",
                "{player} current form",
                "{player} performance trend"
            ],
            IntelligenceType.TEAM_NEWS: [
                "{team} playing XI",
                "{team} team news",
                "{team} lineup changes",
                "{team} squad update"
            ],
            IntelligenceType.WEATHER_CONDITIONS: [
                "{venue} weather forecast",
                "{venue} match day weather",
                "{venue} rain forecast",
                "{venue} conditions"
            ],
            IntelligenceType.PITCH_REPORT: [
                "{venue} pitch report",
                "{venue} surface conditions",
                "{venue} batting bowling conditions",
                "{venue} curator report"
            ],
            IntelligenceType.EXPERT_ANALYSIS: [
                "{player} expert analysis",
                "{player} cricket analysis",
                "{player} performance review",
                "{player} tactical analysis"
            ],
            IntelligenceType.BETTING_INSIGHTS: [
                "{player} betting odds",
                "{player} bookmaker analysis",
                "{player} betting tips",
                "{player} market insights"
            ]
        }
        
        # Relevance keywords for filtering
        self.relevance_keywords = {
            IntelligenceType.INJURY_STATUS: [
                "injury", "injured", "fitness", "medical", "ruled out", 
                "doubtful", "strain", "niggle", "recovery", "rehabilitation"
            ],
            IntelligenceType.RECENT_FORM: [
                "scored", "runs", "wickets", "average", "strike rate",
                "performance", "form", "consistency", "recent", "last"
            ],
            IntelligenceType.TEAM_NEWS: [
                "playing XI", "lineup", "squad", "team", "selection",
                "changes", "dropped", "included", "captain", "vice-captain"
            ]
        }
    
    async def gather_player_intelligence(
        self, 
        player_name: str, 
        intelligence_types: List[IntelligenceType] = None,
        match_context: Dict[str, Any] = None
    ) -> List[IntelligenceResult]:
        """
        Gather comprehensive intelligence about a player
        
        Args:
            player_name: Name of the player
            intelligence_types: Types of intelligence to gather (default: all)
            match_context: Optional match context (venue, opponent, etc.)
            
        Returns:
            List of intelligence results
        """
        if intelligence_types is None:
            intelligence_types = [
                IntelligenceType.INJURY_STATUS,
                IntelligenceType.RECENT_FORM,
                IntelligenceType.EXPERT_ANALYSIS
            ]
        
        logger.info(f"🕵️ Gathering intelligence for {player_name}: {[t.value for t in intelligence_types]}")
        
        results = []
        
        for intel_type in intelligence_types:
            try:
                result = await self._gather_specific_intelligence(
                    intel_type, 
                    player_name=player_name,
                    match_context=match_context
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error gathering {intel_type.value} for {player_name}: {e}")
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"✅ Gathered {len(results)} intelligence items for {player_name}")
        return results
    
    async def gather_match_intelligence(
        self,
        venue: str,
        team1: str,
        team2: str,
        intelligence_types: List[IntelligenceType] = None
    ) -> List[IntelligenceResult]:
        """
        Gather match-specific intelligence
        
        Args:
            venue: Match venue
            team1: First team name
            team2: Second team name  
            intelligence_types: Types of intelligence to gather
            
        Returns:
            List of intelligence results
        """
        if intelligence_types is None:
            intelligence_types = [
                IntelligenceType.WEATHER_CONDITIONS,
                IntelligenceType.PITCH_REPORT,
                IntelligenceType.TEAM_NEWS
            ]
        
        logger.info(f"🏟️ Gathering match intelligence for {team1} vs {team2} at {venue}")
        
        results = []
        
        for intel_type in intelligence_types:
            try:
                if intel_type == IntelligenceType.TEAM_NEWS:
                    # Gather for both teams
                    for team in [team1, team2]:
                        result = await self._gather_specific_intelligence(
                            intel_type,
                            team_name=team,
                            match_context={"venue": venue, "opponent": team2 if team == team1 else team1}
                        )
                        if result:
                            results.append(result)
                else:
                    result = await self._gather_specific_intelligence(
                        intel_type,
                        venue=venue,
                        match_context={"team1": team1, "team2": team2}
                    )
                    if result:
                        results.append(result)
            except Exception as e:
                logger.error(f"Error gathering {intel_type.value} for match: {e}")
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"✅ Gathered {len(results)} match intelligence items")
        return results
    
    async def _gather_specific_intelligence(
        self,
        intel_type: IntelligenceType,
        player_name: str = None,
        team_name: str = None,
        venue: str = None,
        match_context: Dict[str, Any] = None
    ) -> Optional[IntelligenceResult]:
        """Gather specific type of intelligence"""
        
        # Check cache first
        cache_key = f"{intel_type.value}_{player_name}_{team_name}_{venue}"
        if cache_key in self.cache:
            cached_result, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logger.debug(f"📋 Using cached result for {cache_key}")
                return cached_result
        
        # Generate search queries
        search_queries = self._generate_search_queries(
            intel_type, player_name, team_name, venue, match_context
        )
        
        if not search_queries:
            return None
        
        # Perform web search
        search_results = []
        for query in search_queries[:2]:  # Limit to 2 queries per intelligence type
            try:
                if self.web_search:
                    result = await self._perform_web_search(query)
                    if result:
                        search_results.extend(result)
            except Exception as e:
                logger.warning(f"Web search failed for '{query}': {e}")
        
        if not search_results:
            return None
        
        # Process and filter results
        processed_result = self._process_search_results(
            intel_type, search_results, player_name, team_name, venue
        )
        
        # Cache the result
        if processed_result:
            self.cache[cache_key] = (processed_result, datetime.now())
        
        return processed_result
    
    def _generate_search_queries(
        self,
        intel_type: IntelligenceType,
        player_name: str = None,
        team_name: str = None,
        venue: str = None,
        match_context: Dict[str, Any] = None
    ) -> List[str]:
        """Generate search queries for specific intelligence type"""
        
        patterns = self.search_patterns.get(intel_type, [])
        queries = []
        
        for pattern in patterns:
            query = pattern
            
            if player_name and "{player}" in pattern:
                query = pattern.format(player=player_name)
            elif team_name and "{team}" in pattern:
                query = pattern.format(team=team_name)
            elif venue and "{venue}" in pattern:
                query = pattern.format(venue=venue)
            else:
                continue  # Skip if no substitution possible
            
            # Add recent time qualifier for time-sensitive searches
            if intel_type in [IntelligenceType.INJURY_STATUS, IntelligenceType.RECENT_FORM, IntelligenceType.TEAM_NEWS]:
                query += " latest news"
            
            queries.append(query)
        
        return queries
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using injected search tool"""
        if not self.web_search:
            logger.warning("No web search tool available")
            return []
        
        try:
            logger.debug(f"🔍 Searching: {query}")
            
            # This would call the injected web search function
            # For now, we'll simulate the structure
            if hasattr(self.web_search, '__call__'):
                result = await self.web_search(query)
                return result if isinstance(result, list) else [result]
            else:
                # Fallback for non-async web search
                result = self.web_search(query)
                return result if isinstance(result, list) else [result]
                
        except Exception as e:
            logger.error(f"Web search error for '{query}': {e}")
            return []
    
    def _process_search_results(
        self,
        intel_type: IntelligenceType,
        search_results: List[Dict[str, Any]],
        player_name: str = None,
        team_name: str = None,
        venue: str = None
    ) -> Optional[IntelligenceResult]:
        """Process and filter search results into structured intelligence"""
        
        if not search_results:
            return None
        
        # Combine and filter relevant content
        relevant_content = []
        total_relevance = 0.0
        sources = []
        
        for result in search_results:
            content = result.get('content', '') or result.get('snippet', '') or result.get('text', '')
            source = result.get('url', '') or result.get('source', 'web')
            
            if content:
                relevance = self._calculate_relevance(intel_type, content, player_name, team_name, venue)
                if relevance > 0.3:  # Minimum relevance threshold
                    relevant_content.append(content)
                    total_relevance += relevance
                    sources.append(source)
        
        if not relevant_content:
            return None
        
        # Create structured result
        combined_content = self._combine_content(relevant_content, intel_type)
        avg_relevance = total_relevance / len(relevant_content)
        
        return IntelligenceResult(
            intelligence_type=intel_type,
            player_name=player_name,
            team_name=team_name,
            venue=venue,
            content=combined_content,
            confidence=min(avg_relevance, 0.95),  # Cap at 95%
            source=sources[0] if sources else "web",
            relevance_score=avg_relevance
        )
    
    def _calculate_relevance(
        self,
        intel_type: IntelligenceType,
        content: str,
        player_name: str = None,
        team_name: str = None,
        venue: str = None
    ) -> float:
        """Calculate relevance score for content"""
        
        content_lower = content.lower()
        relevance_score = 0.0
        
        # Check for entity mentions
        if player_name and player_name.lower() in content_lower:
            relevance_score += 0.4
        if team_name and team_name.lower() in content_lower:
            relevance_score += 0.3
        if venue and venue.lower() in content_lower:
            relevance_score += 0.2
        
        # Check for relevant keywords
        keywords = self.relevance_keywords.get(intel_type, [])
        keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
        keyword_score = min(keyword_matches * 0.1, 0.4)
        relevance_score += keyword_score
        
        # Recency bonus (if content mentions recent dates)
        recent_patterns = [
            r'today', r'yesterday', r'this week', r'latest', r'recent',
            r'2025', r'august', r'september'  # Current timeframe
        ]
        recency_matches = sum(1 for pattern in recent_patterns if re.search(pattern, content_lower))
        recency_score = min(recency_matches * 0.05, 0.2)
        relevance_score += recency_score
        
        return min(relevance_score, 1.0)
    
    def _combine_content(self, content_list: List[str], intel_type: IntelligenceType) -> str:
        """Combine multiple content pieces into coherent intelligence"""
        
        if len(content_list) == 1:
            return content_list[0]
        
        # For multiple pieces, create a summary
        combined = f"Multiple sources report on {intel_type.value}:\n\n"
        
        for i, content in enumerate(content_list[:3], 1):  # Limit to top 3
            # Truncate very long content
            truncated = content[:200] + "..." if len(content) > 200 else content
            combined += f"{i}. {truncated}\n\n"
        
        return combined.strip()
    
    def get_intelligence_summary(self, results: List[IntelligenceResult]) -> Dict[str, Any]:
        """Generate a summary of gathered intelligence"""
        
        summary = {
            "total_items": len(results),
            "intelligence_types": list(set(r.intelligence_type.value for r in results)),
            "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0.0,
            "high_confidence_items": len([r for r in results if r.confidence > 0.7]),
            "recent_items": len([r for r in results if (datetime.now() - r.timestamp).hours < 24]),
            "key_insights": []
        }
        
        # Extract key insights
        for result in results[:3]:  # Top 3 by relevance
            insight = {
                "type": result.intelligence_type.value,
                "content": result.content[:100] + "..." if len(result.content) > 100 else result.content,
                "confidence": result.confidence
            }
            summary["key_insights"].append(insight)
        
        return summary

# Factory function for easy integration
def create_web_intelligence_agent(web_search_tool=None) -> WebIntelligenceAgent:
    """
    Factory function to create Web Intelligence Agent
    
    Args:
        web_search_tool: Web search function to inject
        
    Returns:
        Configured WebIntelligenceAgent instance
    """
    return WebIntelligenceAgent(web_search_tool=web_search_tool)
