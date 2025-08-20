# Purpose: Enhanced cricket player image service using ESPNCricinfo and other cricket sources
# Author: Assistant, Last Modified: 2025-01-20

import requests
import json
import logging
from typing import Optional, Dict, List
from urllib.parse import quote_plus
import re
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnhancedCricketImageService:
    """
    Enhanced service for finding real cricket player photos from reliable sources
    """
    
    def __init__(self, cache_dir: str = "image_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # ESPNCricinfo base URLs for player searches
        self.espn_search_url = "https://www.espncricinfo.com/ci/engine/match/index.html"
        self.espn_player_search = "https://search-prod.espncricinfo.com/ci/content/site/search.json"
        
        # Backup image sources
        self.backup_sources = [
            "https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_320/lsci",
            "https://p.imgci.com/db/PICTURES/CMS",
            "https://img1.hscicdn.com/image/upload/f_auto/lsci"
        ]
        
        # Team color mappings for fallback avatars
        self.team_colors = {
            'Chennai Super Kings': 'fbbf24',
            'Mumbai Indians': '1e40af', 
            'Royal Challengers Bangalore': 'dc2626',
            'Royal Challengers Bengaluru': 'dc2626',
            'Kolkata Knight Riders': '7c3aed',
            'Delhi Capitals': '1e40af',
            'Rajasthan Royals': 'ec4899',
            'Sunrisers Hyderabad': 'ea580c',
            'Punjab Kings': 'dc2626',
            'Gujarat Titans': '1e40af',
            'Lucknow Super Giants': '06b6d4',
            'India': '1e40af',
            'Australia': 'fbbf24',
            'England': 'dc2626',
            'South Africa': '059669',
            'Pakistan': '059669',
            'West Indies': '7c2d12',
            'New Zealand': '000000',
            'Sri Lanka': '1e40af',
            'Bangladesh': '059669',
        }
    
    def search_espncricinfo_player(self, player_name: str) -> Optional[Dict]:
        """
        Search ESPNCricinfo for player information and image
        """
        try:
            # Clean player name for search
            search_name = player_name.strip().replace("'", "").replace(".", "")
            
            # Try multiple search patterns
            search_patterns = [
                search_name,
                search_name.replace(" ", "-"),
                search_name.lower().replace(" ", "-")
            ]
            
            for pattern in search_patterns:
                logger.info(f"🔍 Searching ESPNCricinfo for: {pattern}")
                
                # ESPNCricinfo player URLs often follow patterns like:
                # https://www.espncricinfo.com/player/virat-kohli-253802
                # https://www.espncricinfo.com/player/ms-dhoni-28081
                
                # Try to construct likely URLs
                likely_urls = [
                    f"https://www.espncricinfo.com/player/{pattern.lower()}",
                    f"https://www.espncricinfo.com/player/{pattern.lower()}-{i}" 
                    for i in range(100000, 999999, 50000)  # Common ID ranges
                ]
                
                for url in likely_urls[:5]:  # Limit attempts
                    try:
                        response = requests.get(url, timeout=5, headers={
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                        })
                        
                        if response.status_code == 200:
                            # Look for player image in the HTML
                            html_content = response.text
                            
                            # Common ESPNCricinfo image patterns
                            image_patterns = [
                                r'<img[^>]+src="([^"]*player[^"]*\.jpg)"',
                                r'<img[^>]+src="([^"]*headshot[^"]*\.jpg)"',
                                r'<img[^>]+src="([^"]*portrait[^"]*\.jpg)"',
                                r'<img[^>]+src="(https://img1\.hscicdn\.com[^"]*\.jpg)"',
                                r'<img[^>]+src="(https://p\.imgci\.com[^"]*\.jpg)"'
                            ]
                            
                            for pattern_regex in image_patterns:
                                matches = re.findall(pattern_regex, html_content)
                                if matches:
                                    image_url = matches[0]
                                    if self._validate_image_url(image_url):
                                        logger.info(f"✅ Found ESPNCricinfo image for {player_name}: {image_url}")
                                        return {
                                            'url': image_url,
                                            'source': 'ESPNCricinfo',
                                            'player_url': url
                                        }
                    
                    except Exception as e:
                        logger.debug(f"Failed to fetch {url}: {e}")
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"❌ ESPNCricinfo search failed for {player_name}: {e}")
            return None
    
    def search_cricinfo_images_api(self, player_name: str) -> Optional[str]:
        """
        Try to find player images using Cricinfo's image CDN patterns
        """
        try:
            # Common player image ID patterns on ESPNCricinfo
            name_variations = [
                player_name.lower().replace(" ", "-"),
                player_name.lower().replace(" ", ""),
                player_name.replace(" ", "-"),
                "-".join(player_name.lower().split())
            ]
            
            # ESPNCricinfo image CDN patterns
            cdn_patterns = [
                "https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_320/lsci/db/PICTURES/CMS/{}/{}",
                "https://p.imgci.com/db/PICTURES/CMS/{}/{}",
                "https://img1.hscicdn.com/image/upload/f_auto/lsci/db/PICTURES/CMS/{}/{}"
            ]
            
            # Common image file patterns
            image_patterns = [
                "{}.jpg",
                "{}_headshot.jpg", 
                "{}_portrait.jpg",
                "headshots/{}.jpg",
                "portraits/{}.jpg"
            ]
            
            for name_var in name_variations:
                for cdn_pattern in cdn_patterns:
                    for img_pattern in image_patterns:
                        # Try different folder structures
                        folders = ["headshots", "portraits", "players", ""]
                        
                        for folder in folders:
                            image_filename = img_pattern.format(name_var)
                            image_url = cdn_pattern.format(folder, image_filename)
                            
                            if self._validate_image_url(image_url):
                                logger.info(f"✅ Found Cricinfo CDN image: {image_url}")
                                return image_url
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cricinfo CDN search failed: {e}")
            return None
    
    def _validate_image_url(self, url: str) -> bool:
        """
        Validate that an image URL is accessible and returns an actual image
        """
        try:
            response = requests.head(url, timeout=3, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            
            return (response.status_code == 200 and 
                   'image' in response.headers.get('content-type', '').lower())
        
        except Exception:
            return False
    
    def get_cached_image(self, player_name: str) -> Optional[Dict]:
        """
        Get cached image data for a player
        """
        try:
            cache_file = self.cache_dir / f"{player_name.replace(' ', '_')}_image.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is still valid (30 days)
                cached_date = datetime.fromisoformat(cached_data['cached_date'])
                if (datetime.now() - cached_date).days < 30:
                    logger.info(f"📷 Using cached image for {player_name}")
                    return cached_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to read cache for {player_name}: {e}")
            return None
    
    def cache_image_data(self, player_name: str, image_data: Dict):
        """
        Cache image data for a player
        """
        try:
            cache_file = self.cache_dir / f"{player_name.replace(' ', '_')}_image.json"
            
            cache_data = {
                **image_data,
                'cached_date': datetime.now().isoformat(),
                'player_name': player_name
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"💾 Cached image data for {player_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache image for {player_name}: {e}")
    
    def generate_team_avatar(self, player_name: str, teams: Optional[List[str]] = None) -> str:
        """
        Generate a team-colored avatar as fallback
        """
        clean_name = player_name.replace(' ', '+').replace("'", "").replace('.', '')
        
        # Get team color
        bg_color = '1e40af'  # Default cricket blue
        if teams and len(teams) > 0:
            primary_team = teams[0]
            bg_color = self.team_colors.get(primary_team, '1e40af')
        
        return (f"https://ui-avatars.com/api/"
               f"?name={clean_name}"
               f"&background={bg_color}"
               f"&color=fff"
               f"&size=150"
               f"&font-size=0.4"
               f"&format=png"
               f"&rounded=true"
               f"&bold=true")
    
    def get_best_player_image(self, player_name: str, teams: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get the best available image for a cricket player
        
        Returns:
            Dict with 'url', 'source', and 'cached' keys
        """
        
        # 1. Check cache first
        cached_data = self.get_cached_image(player_name)
        if cached_data:
            return {
                'url': cached_data['url'],
                'source': cached_data.get('source', 'cached'),
                'cached': True
            }
        
        # 2. Try ESPNCricinfo search
        espn_result = self.search_espncricinfo_player(player_name)
        if espn_result:
            self.cache_image_data(player_name, espn_result)
            return {
                'url': espn_result['url'],
                'source': 'ESPNCricinfo',
                'cached': False
            }
        
        # 3. Try Cricinfo CDN patterns
        cdn_image = self.search_cricinfo_images_api(player_name)
        if cdn_image:
            image_data = {'url': cdn_image, 'source': 'Cricinfo_CDN'}
            self.cache_image_data(player_name, image_data)
            return {
                'url': cdn_image,
                'source': 'Cricinfo_CDN', 
                'cached': False
            }
        
        # 4. Fallback to team-colored avatar
        avatar_url = self.generate_team_avatar(player_name, teams)
        return {
            'url': avatar_url,
            'source': 'team_avatar',
            'cached': False
        }

# Global instance
cricket_image_service = EnhancedCricketImageService()

def get_enhanced_player_image(player_name: str, teams: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Convenience function to get enhanced player images
    """
    return cricket_image_service.get_best_player_image(player_name, teams)

if __name__ == "__main__":
    # Test the service
    logging.basicConfig(level=logging.INFO)
    
    test_players = ["Virat Kohli", "MS Dhoni", "Rohit Sharma", "Kane Williamson"]
    
    for player in test_players:
        print(f"\n🏏 Testing {player}:")
        result = get_enhanced_player_image(player, ["India"])
        print(f"   URL: {result['url']}")
        print(f"   Source: {result['source']}")
        print(f"   Cached: {result['cached']}")
