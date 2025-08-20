# Purpose: Cricket Player Image Service with team colors and OpenAI integration
# Author: Assistant, Last Modified: 2025-01-19

import os
import requests
import hashlib
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CricketImageService:
    """Service for generating and caching cricket player images"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.cache_dir = "player_images_cache"
        self.ensure_cache_dir()
        
        # Team color mappings
        self.team_colors = {
            'Chennai Super Kings': {'primary': 'fbbf24', 'secondary': '1e40af'},  # Yellow & Blue
            'Mumbai Indians': {'primary': '1e40af', 'secondary': 'fbbf24'},      # Blue & Gold
            'Royal Challengers Bangalore': {'primary': 'dc2626', 'secondary': 'fbbf24'}, # Red & Gold
            'Kolkata Knight Riders': {'primary': '7c3aed', 'secondary': 'fbbf24'}, # Purple & Gold
            'Delhi Capitals': {'primary': '1e40af', 'secondary': 'dc2626'},      # Blue & Red
            'Rajasthan Royals': {'primary': 'ec4899', 'secondary': '1e40af'},    # Pink & Blue
            'Sunrisers Hyderabad': {'primary': 'ea580c', 'secondary': '000000'}, # Orange & Black
            'Punjab Kings': {'primary': 'dc2626', 'secondary': 'fbbf24'},        # Red & Gold
            'Gujarat Titans': {'primary': '1e40af', 'secondary': 'fbbf24'},      # Blue & Gold
            'Lucknow Super Giants': {'primary': '06b6d4', 'secondary': 'ea580c'}, # Cyan & Orange
            'India': {'primary': '1e40af', 'secondary': 'ea580c'},               # Blue & Orange
            'Australia': {'primary': 'fbbf24', 'secondary': '059669'},           # Gold & Green
            'England': {'primary': 'dc2626', 'secondary': '1e40af'},             # Red & Blue
        }
    
    def ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_cache_key(self, player_name: str, image_type: str = "avatar") -> str:
        """Generate cache key for player image"""
        return hashlib.md5(f"{player_name}_{image_type}".encode()).hexdigest()
    
    def get_team_colors(self, teams: list) -> Dict[str, str]:
        """Get team colors for a player based on their teams"""
        if not teams:
            return {'primary': '1e40af', 'secondary': 'fbbf24'}  # Default cricket colors
        
        # Use the first team's colors
        primary_team = teams[0]
        return self.team_colors.get(primary_team, {'primary': '1e40af', 'secondary': 'fbbf24'})
    
    def generate_cricket_avatar(self, player_name: str, teams: Optional[list] = None, 
                              style: str = "modern") -> str:
        """Generate a cricket-themed avatar for the player"""
        
        # Clean name for URL
        clean_name = player_name.replace(' ', '+').replace("'", "").replace('.', '')
        
        # Get team-specific colors
        colors = self.get_team_colors(teams or [])
        
        # Different avatar styles
        if style == "modern":
            return (f"https://ui-avatars.com/api/"
                   f"?name={clean_name}"
                   f"&background={colors['primary']}"
                   f"&color=fff"
                   f"&size=150"
                   f"&font-size=0.4"
                   f"&format=png"
                   f"&rounded=true"
                   f"&bold=true")
        
        elif style == "classic":
            return (f"https://ui-avatars.com/api/"
                   f"?name={clean_name}"
                   f"&background={colors['secondary']}"
                   f"&color={colors['primary']}"
                   f"&size=150"
                   f"&font-size=0.5"
                   f"&format=png"
                   f"&rounded=false")
        
        else:  # simple
            return (f"https://ui-avatars.com/api/"
                   f"?name={clean_name}"
                   f"&background=6b7280"
                   f"&color=fff"
                   f"&size=150")
    
    def search_player_image_openai(self, player_name: str) -> Optional[str]:
        """
        Use OpenAI to search for cricket player images (Future implementation)
        """
        if not self.openai_api_key:
            logger.warning("OpenAI API key not provided for image search")
            return None
        
        # TODO: Implement OpenAI image search
        # This would use OpenAI's vision capabilities or image search APIs
        # to find appropriate cricket player photos
        
        search_prompts = [
            f"Professional headshot of cricket player {player_name}",
            f"Cricket player {player_name} in team uniform",
            f"{player_name} cricket player portrait photo",
            f"Official photo of cricketer {player_name}"
        ]
        
        # Placeholder for future OpenAI integration
        logger.info(f"Would search for {player_name} using prompts: {search_prompts}")
        return None
    
    def get_player_image_url(self, player_name: str, teams: Optional[list] = None, 
                           prefer_real_photo: bool = True) -> str:
        """
        Get the best available image URL for a cricket player
        
        Args:
            player_name: Name of the cricket player
            teams: List of teams the player has played for
            prefer_real_photo: Whether to try finding real photos first
        
        Returns:
            URL to player image
        """
        
        # Try to get real photo first (if enabled and API available)
        if prefer_real_photo and self.openai_api_key:
            real_photo_url = self.search_player_image_openai(player_name)
            if real_photo_url:
                return real_photo_url
        
        # Fallback to generated avatar with team colors
        return self.generate_cricket_avatar(player_name, teams, style="modern")
    
    def get_multiple_image_options(self, player_name: str, teams: Optional[list] = None) -> Dict[str, str]:
        """Get multiple image options for a player"""
        return {
            'modern': self.generate_cricket_avatar(player_name, teams, "modern"),
            'classic': self.generate_cricket_avatar(player_name, teams, "classic"),
            'simple': self.generate_cricket_avatar(player_name, teams, "simple"),
            'team_primary': self.generate_cricket_avatar(player_name, teams, "modern"),
            'team_secondary': self.generate_cricket_avatar(player_name, teams, "classic"),
        }

# Example usage
if __name__ == "__main__":
    # Initialize the service
    image_service = CricketImageService()
    
    # Test with different players
    test_players = [
        ("MS Dhoni", ["Chennai Super Kings", "India"]),
        ("Virat Kohli", ["Royal Challengers Bangalore", "India"]),
        ("Rohit Sharma", ["Mumbai Indians", "India"]),
        ("Unknown Player", [])
    ]
    
    for player_name, teams in test_players:
        print(f"\n{player_name}:")
        print(f"  Main Image: {image_service.get_player_image_url(player_name, teams)}")
        
        options = image_service.get_multiple_image_options(player_name, teams)
        for style, url in options.items():
            print(f"  {style}: {url}")
