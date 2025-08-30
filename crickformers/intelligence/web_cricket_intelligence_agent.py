# Purpose: Web Intelligence Agent (Cricket) for WicketWise
# Author: WicketWise Team, Last Modified: 2025-08-30

"""
Web Intelligence Agent (Cricket)

A specialized agent that gathers pre-match, in-match, and post-match cricket intelligence
from reputable sources with strict domain restrictions and proper citations.

Features:
- Domain-restricted search (ESPNcricinfo, ICC, BBC Sport, Cricbuzz, The Cricketer)
- ESPNcricinfo player headshot extraction with proper licensing
- Structured JSON responses with citations
- Robust error handling with exponential backoff
- De-duplication and content normalization
"""

import logging
import asyncio
import aiohttp
import json
import re
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse
import time
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class WebIntelIntent(Enum):
    """Types of intelligence requests"""
    PRE_MATCH = "pre_match"
    IN_PLAY = "in_play"
    POST_MATCH = "post_match"
    PLAYER_PROFILE = "player_profile"
    NEWS_WATCH = "news_watch"

class WebIntelStatus(Enum):
    """Response status types"""
    OK = "ok"
    PARTIAL = "partial"
    ERROR = "error"
    NEEDS_HUMAN_REVIEW = "needs_human_review"

class CredibilityLevel(Enum):
    """Content credibility levels"""
    HIGH = "high"
    MEDIUM = "medium"

@dataclass
class WebIntelRequest:
    """Input request structure"""
    intent: WebIntelIntent
    query: Optional[str] = None
    teams: Optional[Dict[str, str]] = None  # {"home": "India", "away": "Pakistan"}
    players: Optional[List[str]] = None
    match: Optional[Dict[str, str]] = None  # {"venue": "Colombo", "date": "2023-09-02"}
    need_player_photos: bool = False
    max_items: int = 12
    language: str = "en"

@dataclass
class WebIntelFact:
    """Fact item structure"""
    type: str = "fact"
    title: str = ""
    excerpt: str = ""
    entities: List[str] = field(default_factory=list)
    published_at: Optional[str] = None
    source_id: str = ""
    credibility: CredibilityLevel = CredibilityLevel.HIGH

@dataclass
class WebIntelPhoto:
    """Photo item structure"""
    type: str = "photo"
    player: str = ""
    photo_url: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    source_id: str = ""
    license_note: str = ""
    referer: str = ""

@dataclass
class WebIntelSource:
    """Source citation structure"""
    id: str
    url: str
    publisher: str
    retrieved_at: str
    notes: Optional[str] = None

@dataclass
class WebIntelError:
    """Error information structure"""
    where: str
    message: str

@dataclass
class WebIntelResponse:
    """Complete response structure"""
    status: WebIntelStatus
    items: List[Union[WebIntelFact, WebIntelPhoto]] = field(default_factory=list)
    sources: List[WebIntelSource] = field(default_factory=list)
    errors: List[WebIntelError] = field(default_factory=list)

class WebCricketIntelligenceAgent:
    """
    Web Intelligence Agent (Cricket) for WicketWise
    
    Gathers cricket intelligence from reputable sources with strict domain restrictions,
    proper citations, and ESPNcricinfo headshot extraction capabilities.
    
    Compatible with existing WicketWise agent architecture.
    """
    
    # Allowed domains (non-negotiable)
    ALLOWED_DOMAINS = {
        "espncricinfo.com": "ESPNcricinfo",
        "icc-cricket.com": "ICC", 
        "bbc.com": "BBC Sport",
        "cricbuzz.com": "Cricbuzz",
        "thecricketer.com": "The Cricketer"
    }
    
    # Rate limiting and retry configuration
    RETRY_DELAYS = [0.25, 0.5, 1.0]  # Exponential backoff
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 10
    MAX_CONCURRENT = 3
    
    # Cache configuration
    CACHE_DURATION = timedelta(hours=12)
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        Initialize Web Cricket Intelligence Agent
        
        Args:
            session: Optional aiohttp session for connection pooling
        """
        self.session = session
        self._own_session = session is None
        self.cache = {}  # Simple in-memory cache
        self.source_counter = 0
        
        # Player name normalization map
        self.player_aliases = {
            "MS Dhoni": ["Mahendra Singh Dhoni", "M.S. Dhoni", "Dhoni"],
            "Virat Kohli": ["V Kohli", "V. Kohli", "Kohli"],
            "Rohit Sharma": ["R Sharma", "R. Sharma", "Rohit"],
            # Add more as needed
        }
        
        logger.info("🌐 Web Cricket Intelligence Agent initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self._own_session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                headers={
                    'User-Agent': 'WicketWise-Intelligence/1.0 (Cricket Analysis Bot)'
                }
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._own_session and self.session:
            await self.session.close()
    
    def _generate_source_id(self) -> str:
        """Generate unique source ID"""
        self.source_counter += 1
        return f"s{self.source_counter}"
    
    def _normalize_player_name(self, name: str) -> str:
        """Normalize player name using alias map"""
        for canonical, aliases in self.player_aliases.items():
            if name in aliases or name == canonical:
                return canonical
        return name
    
    def _extract_entities(self, text: str, teams: Optional[Dict[str, str]] = None, 
                         players: Optional[List[str]] = None) -> List[str]:
        """Extract cricket entities from text"""
        entities = []
        text_lower = text.lower()
        
        # Extract team names
        if teams:
            for team in teams.values():
                if team and team.lower() in text_lower:
                    entities.append(team)
        
        # Extract player names
        if players:
            for player in players:
                if player and player.lower() in text_lower:
                    entities.append(self._normalize_player_name(player))
        
        # Extract common cricket terms
        cricket_terms = [
            "pitch", "wicket", "batting", "bowling", "spin", "pace", 
            "powerplay", "death overs", "toss", "drs", "review"
        ]
        
        for term in cricket_terms:
            if term in text_lower:
                entities.append(term)
        
        return list(set(entities))  # Remove duplicates
    
    def _build_search_query(self, request: WebIntelRequest, domain: str) -> str:
        """Build domain-specific search query"""
        query_parts = [f"site:{domain}"]
        
        # Add intent-specific keywords
        intent_keywords = {
            WebIntelIntent.PRE_MATCH: ["preview", "prediction", "team news", "pitch report"],
            WebIntelIntent.IN_PLAY: ["live", "scorecard", "commentary", "updates"],
            WebIntelIntent.POST_MATCH: ["report", "highlights", "analysis", "result"],
            WebIntelIntent.PLAYER_PROFILE: ["profile", "stats", "career", "biography"],
            WebIntelIntent.NEWS_WATCH: ["news", "latest", "update", "breaking"]
        }
        
        if request.intent in intent_keywords:
            query_parts.extend(intent_keywords[request.intent][:2])  # Limit keywords
        
        # Add specific query
        if request.query:
            query_parts.append(request.query)
        
        # Add team names
        if request.teams:
            for team in request.teams.values():
                if team:
                    query_parts.append(team)
        
        # Add player names
        if request.players:
            query_parts.extend(request.players[:2])  # Limit to 2 players
        
        # Add venue and date
        if request.match:
            if request.match.get("venue"):
                query_parts.append(request.match["venue"])
            if request.match.get("date"):
                query_parts.append(request.match["date"])
        
        return " ".join(query_parts)
    
    def _get_cache_key(self, domain: str, query: str) -> str:
        """Generate cache key for query"""
        key_data = f"{domain}:{query}:{datetime.now().date()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if "timestamp" not in cache_entry:
            return False
        
        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        return datetime.now() - cached_time < self.CACHE_DURATION
    
    async def gather_intelligence(self, request: WebIntelRequest) -> WebIntelResponse:
        """
        Main entry point for gathering cricket intelligence
        
        Args:
            request: WebIntelRequest with search parameters
            
        Returns:
            WebIntelResponse with facts, photos, and citations
        """
        logger.info(f"🔍 Gathering cricket intelligence: {request.intent.value}")
        
        response = WebIntelResponse(status=WebIntelStatus.OK)
        
        try:
            # Validate request
            if not self._validate_request(request, response):
                return response
            
            # Search across allowed domains
            search_tasks = []
            for domain in self.ALLOWED_DOMAINS.keys():
                task = self._search_domain(domain, request, response)
                search_tasks.append(task)
            
            # Execute searches with concurrency limit
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
            
            async def limited_search(task):
                async with semaphore:
                    return await task
            
            await asyncio.gather(*[limited_search(task) for task in search_tasks])
            
            # Get player photos if requested
            if request.need_player_photos and request.players:
                await self._get_player_photos(request.players, response)
            
            # Post-process results
            self._deduplicate_items(response)
            self._limit_results(response, request.max_items)
            self._determine_final_status(response)
            
            logger.info(f"✅ Intelligence gathered: {len(response.items)} items, {len(response.sources)} sources")
            
        except Exception as e:
            logger.error(f"❌ Intelligence gathering failed: {e}")
            response.status = WebIntelStatus.ERROR
            response.errors.append(WebIntelError(
                where="gather_intelligence",
                message=f"Unexpected error: {str(e)}"
            ))
        
        return response
    
    def _validate_request(self, request: WebIntelRequest, response: WebIntelResponse) -> bool:
        """Validate request parameters"""
        if request.max_items > 25:
            response.errors.append(WebIntelError(
                where="validation",
                message="max_items cannot exceed 25"
            ))
            request.max_items = 25
        
        return True
    
    async def _search_domain(self, domain: str, request: WebIntelRequest, 
                           response: WebIntelResponse) -> None:
        """Search a specific domain for intelligence"""
        try:
            query = self._build_search_query(request, domain)
            cache_key = self._get_cache_key(domain, query)
            
            # Check cache first
            if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                logger.debug(f"📋 Using cached results for {domain}")
                cached_data = self.cache[cache_key]
                response.items.extend(cached_data.get("items", []))
                response.sources.extend(cached_data.get("sources", []))
                return
            
            # Perform search with retries
            search_results = await self._perform_search_with_retry(domain, query)
            
            if not search_results:
                response.errors.append(WebIntelError(
                    where=domain,
                    message="No search results found"
                ))
                return
            
            # Parse results
            domain_items = []
            domain_sources = []
            
            for result_url in search_results[:5]:  # Limit to top 5 results per domain
                try:
                    parsed_content = await self._fetch_and_parse(result_url, domain)
                    if parsed_content:
                        fact, source = self._create_fact_and_source(
                            parsed_content, result_url, domain, request
                        )
                        if fact and source:
                            domain_items.append(fact)
                            domain_sources.append(source)
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse {result_url}: {e}")
                    response.errors.append(WebIntelError(
                        where=result_url,
                        message=f"Parse error: {str(e)}"
                    ))
            
            # Cache results
            self.cache[cache_key] = {
                "items": domain_items,
                "sources": domain_sources,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to response
            response.items.extend(domain_items)
            response.sources.extend(domain_sources)
            
        except Exception as e:
            logger.error(f"❌ Domain search failed for {domain}: {e}")
            response.errors.append(WebIntelError(
                where=domain,
                message=f"Search failed: {str(e)}"
            ))
    
    async def _perform_search_with_retry(self, domain: str, query: str) -> List[str]:
        """Perform search with exponential backoff retry"""
        for attempt in range(self.MAX_RETRIES):
            try:
                # In a real implementation, this would use a search API
                # For now, we'll simulate with domain-specific URL patterns
                return await self._simulate_search(domain, query)
                
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise e
                
                delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                logger.warning(f"⚠️ Search attempt {attempt + 1} failed for {domain}, retrying in {delay}s")
                await asyncio.sleep(delay)
        
        return []
    
    async def _simulate_search(self, domain: str, query: str) -> List[str]:
        """Simulate search results (replace with real search API)"""
        # This is a placeholder - in production, integrate with search APIs
        # or web scraping following robots.txt
        
        base_urls = {
            "espncricinfo.com": "https://www.espncricinfo.com",
            "icc-cricket.com": "https://www.icc-cricket.com", 
            "bbc.com": "https://www.bbc.com/sport/cricket",
            "cricbuzz.com": "https://www.cricbuzz.com",
            "thecricketer.com": "https://www.thecricketer.com"
        }
        
        # Return simulated URLs for testing
        base_url = base_urls.get(domain, f"https://{domain}")
        return [
            f"{base_url}/article-1",
            f"{base_url}/article-2",
            f"{base_url}/article-3"
        ]
    
    async def _fetch_and_parse(self, url: str, domain: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse content from URL"""
        if not self.session:
            return None
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract content based on domain
                return self._parse_content_by_domain(soup, domain, url)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch {url}: {e}")
            return None
    
    def _parse_content_by_domain(self, soup: BeautifulSoup, domain: str, url: str) -> Dict[str, Any]:
        """Parse content with domain-specific selectors"""
        content = {
            "title": "",
            "excerpt": "",
            "published_at": None,
            "og_image": None
        }
        
        # Extract title
        title_selectors = [
            "h1",
            ".headline",
            ".article-title",
            "title"
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                content["title"] = title_elem.get_text(strip=True)
                break
        
        # Extract excerpt
        excerpt_selectors = [
            ".standfirst",
            ".article-lead",
            ".summary",
            "p"
        ]
        
        for selector in excerpt_selectors:
            excerpt_elem = soup.select_one(selector)
            if excerpt_elem:
                text = excerpt_elem.get_text(strip=True)
                content["excerpt"] = text[:300] + "..." if len(text) > 300 else text
                break
        
        # Extract published date
        time_elem = soup.select_one("time")
        if time_elem:
            datetime_attr = time_elem.get("datetime")
            if datetime_attr:
                content["published_at"] = datetime_attr
        
        # Extract Open Graph image
        og_image = soup.select_one('meta[property="og:image"]')
        if og_image:
            content["og_image"] = og_image.get("content")
        
        return content
    
    def _create_fact_and_source(self, content: Dict[str, Any], url: str, domain: str,
                               request: WebIntelRequest) -> Tuple[Optional[WebIntelFact], Optional[WebIntelSource]]:
        """Create fact and source objects from parsed content"""
        if not content.get("title") or not content.get("excerpt"):
            return None, None
        
        source_id = self._generate_source_id()
        
        # Create fact
        fact = WebIntelFact(
            title=content["title"],
            excerpt=content["excerpt"],
            entities=self._extract_entities(
                content["title"] + " " + content["excerpt"],
                request.teams,
                request.players
            ),
            published_at=content.get("published_at"),
            source_id=source_id,
            credibility=self._determine_credibility(content, domain)
        )
        
        # Create source
        source = WebIntelSource(
            id=source_id,
            url=url,
            publisher=self.ALLOWED_DOMAINS[domain],
            retrieved_at=datetime.utcnow().isoformat() + "Z"
        )
        
        return fact, source
    
    def _determine_credibility(self, content: Dict[str, Any], domain: str) -> CredibilityLevel:
        """Determine content credibility based on domain and content"""
        # All allowed domains are high credibility by default
        credibility = CredibilityLevel.HIGH
        
        # Check for opinion/blog markers
        title_lower = content.get("title", "").lower()
        excerpt_lower = content.get("excerpt", "").lower()
        
        opinion_markers = ["opinion", "blog", "comment", "editorial", "analysis"]
        
        for marker in opinion_markers:
            if marker in title_lower or marker in excerpt_lower:
                credibility = CredibilityLevel.MEDIUM
                break
        
        return credibility
    
    async def _get_player_photos(self, players: List[str], response: WebIntelResponse) -> None:
        """Get player headshots from ESPNcricinfo"""
        for player in players:
            try:
                photo = await self._get_espn_player_photo(player)
                if photo:
                    response.items.append(photo)
                    
                    # Add source if not already present
                    source_exists = any(s.id == photo.source_id for s in response.sources)
                    if not source_exists:
                        source = WebIntelSource(
                            id=photo.source_id,
                            url=photo.referer,
                            publisher="ESPNcricinfo",
                            retrieved_at=datetime.utcnow().isoformat() + "Z"
                        )
                        response.sources.append(source)
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to get photo for {player}: {e}")
                response.errors.append(WebIntelError(
                    where=f"photo:{player}",
                    message=f"Photo extraction failed: {str(e)}"
                ))
    
    async def _get_espn_player_photo(self, player: str) -> Optional[WebIntelPhoto]:
        """Get player headshot from ESPNcricinfo profile"""
        if not self.session:
            return None
        
        try:
            # Search for player profile (simulated)
            profile_url = f"https://www.espncricinfo.com/cricketers/{player.lower().replace(' ', '-')}-12345"
            
            async with self.session.get(profile_url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract Open Graph image
                og_image = soup.select_one('meta[property="og:image"]')
                if not og_image:
                    return None
                
                photo_url = og_image.get("content")
                if not photo_url:
                    return None
                
                # Validate image (basic checks)
                if not self._is_valid_headshot(photo_url):
                    return None
                
                source_id = self._generate_source_id()
                
                return WebIntelPhoto(
                    player=self._normalize_player_name(player),
                    photo_url=photo_url,
                    source_id=source_id,
                    license_note="ESPNcricinfo profile headshot — for internal analysis use",
                    referer=profile_url
                )
                
        except Exception as e:
            logger.warning(f"⚠️ ESPN photo extraction failed for {player}: {e}")
            return None
    
    def _is_valid_headshot(self, photo_url: str) -> bool:
        """Validate if image URL appears to be a valid headshot"""
        if not photo_url:
            return False
        
        # Check for common headshot indicators
        headshot_indicators = [
            "/PICTURES/",
            "/image/",
            "headshot",
            "profile",
            "player"
        ]
        
        url_lower = photo_url.lower()
        return any(indicator in url_lower for indicator in headshot_indicators)
    
    def _deduplicate_items(self, response: WebIntelResponse) -> None:
        """Remove duplicate items based on title similarity"""
        seen_titles = set()
        unique_items = []
        
        for item in response.items:
            if hasattr(item, 'title'):
                title_hash = hashlib.md5(item.title.lower().encode()).hexdigest()
                if title_hash not in seen_titles:
                    seen_titles.add(title_hash)
                    unique_items.append(item)
            else:
                unique_items.append(item)  # Photos don't have titles
        
        response.items = unique_items
    
    def _limit_results(self, response: WebIntelResponse, max_items: int) -> None:
        """Limit results to max_items"""
        if len(response.items) > max_items:
            response.items = response.items[:max_items]
    
    def _determine_final_status(self, response: WebIntelResponse) -> None:
        """Determine final response status"""
        if not response.items and not response.errors:
            response.status = WebIntelStatus.PARTIAL
            response.errors.append(WebIntelError(
                where="search",
                message="No items found on allowed domains"
            ))
        elif response.errors and response.items:
            response.status = WebIntelStatus.PARTIAL
        elif response.errors and not response.items:
            response.status = WebIntelStatus.ERROR


# Factory function for compatibility with existing agent patterns
def create_web_cricket_intelligence_agent(session: Optional[aiohttp.ClientSession] = None) -> WebCricketIntelligenceAgent:
    """
    Factory function to create Web Cricket Intelligence Agent
    
    Args:
        session: Optional aiohttp session
        
    Returns:
        WebCricketIntelligenceAgent instance
    """
    return WebCricketIntelligenceAgent(session=session)
