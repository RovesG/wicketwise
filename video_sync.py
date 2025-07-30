# Purpose: Links ball metadata to video files using filename matching
# Author: Shamus Rae, Last Modified: July 17, 2025

import os
import re
from typing import Optional, Dict, List
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

def find_video_for_ball(ball_metadata: dict, video_directory: str) -> Optional[str]:
    """
    Find a video file that matches the given ball metadata.
    
    Args:
        ball_metadata: Dictionary containing ball information with keys:
            - batter: Name of the batter
            - bowler: Name of the bowler (optional)
            - over: Over number
            - ball: Ball number within over
            - match_date: Date of the match (optional)
        video_directory: Directory path to search for video files
    
    Returns:
        Full file path to the matched video file, or None if no match found
    
    Matching Priority:
        1. Batter name + over/ball number + date
        2. Batter name + over/ball number
        3. Batter name + date
        4. Batter name only
        5. Over/ball number + date
        6. Over/ball number only
    """
    if not os.path.exists(video_directory):
        logger.warning(f"Video directory does not exist: {video_directory}")
        return None
    
    # Get all video files in the directory
    video_files = []
    for file in os.listdir(video_directory):
        if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            video_files.append(file)
    
    if not video_files:
        logger.info(f"No video files found in directory: {video_directory}")
        return None
    
    # Extract metadata components
    batter = ball_metadata.get('batter', '').strip()
    bowler = ball_metadata.get('bowler', '').strip()
    over = ball_metadata.get('over')
    ball = ball_metadata.get('ball')
    match_date = ball_metadata.get('match_date')
    
    # Normalize batter name for matching (remove spaces, convert to lowercase)
    batter_normalized = normalize_name(batter)
    bowler_normalized = normalize_name(bowler) if bowler else ''
    
    # Create search patterns with different priority levels
    search_patterns = create_search_patterns(
        batter_normalized, bowler_normalized, over, ball, match_date
    )
    
    # Try to match files using priority-based patterns
    for priority, patterns in search_patterns.items():
        logger.debug(f"Trying priority {priority} patterns: {patterns}")
        
        for pattern in patterns:
            for video_file in video_files:
                if match_pattern_in_filename(pattern, video_file):
                    full_path = os.path.join(video_directory, video_file)
                    logger.info(f"Found match: {video_file} for pattern: {pattern}")
                    return full_path
    
    logger.info(f"No video file found for ball metadata: {ball_metadata}")
    return None

def normalize_name(name: str) -> str:
    """
    Normalize a player name for matching.
    
    Args:
        name: Player name to normalize
    
    Returns:
        Normalized name (lowercase, no spaces, no special characters)
    """
    if not name:
        return ''
    
    # Convert to lowercase, remove spaces and special characters
    normalized = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
    return normalized

def create_search_patterns(batter: str, bowler: str, over: int, ball: int, match_date: str) -> Dict[int, List[str]]:
    """
    Create search patterns with different priority levels.
    
    Args:
        batter: Normalized batter name
        bowler: Normalized bowler name
        over: Over number
        ball: Ball number
        match_date: Match date string
    
    Returns:
        Dictionary with priority levels as keys and pattern lists as values
    """
    patterns = {
        1: [],  # Highest priority: batter + over/ball + date
        2: [],  # High priority: batter + over/ball
        3: [],  # Medium priority: batter + date
        4: [],  # Low priority: batter only
        5: [],  # Lower priority: over/ball + date
        6: []   # Lowest priority: over/ball only
    }
    
    # Format over/ball information
    over_ball = f"{over}{ball}" if over is not None and ball is not None else None
    over_ball_alt = f"over{over}ball{ball}" if over is not None and ball is not None else None
    
    # Format date information
    date_formatted = format_date_for_matching(match_date) if match_date else None
    
    # Priority 1: Batter + over/ball + date
    if batter and over_ball and date_formatted:
        patterns[1].extend([
            f"{batter}.*{over_ball}.*{date_formatted}",
            f"{batter}.*{over_ball_alt}.*{date_formatted}",
            f"{date_formatted}.*{batter}.*{over_ball}",
            f"{date_formatted}.*{batter}.*{over_ball_alt}"
        ])
    
    # Priority 2: Batter + over/ball
    if batter and over_ball:
        patterns[2].extend([
            f"{batter}.*{over_ball}",
            f"{batter}.*{over_ball_alt}",
            f"{over_ball}.*{batter}",
            f"{over_ball_alt}.*{batter}"
        ])
    
    # Priority 3: Batter + date
    if batter and date_formatted:
        patterns[3].extend([
            f"{batter}.*{date_formatted}",
            f"{date_formatted}.*{batter}"
        ])
    
    # Priority 4: Batter only
    if batter:
        patterns[4].append(batter)
    
    # Priority 5: Over/ball + date
    if over_ball and date_formatted:
        patterns[5].extend([
            f"{over_ball}.*{date_formatted}",
            f"{over_ball_alt}.*{date_formatted}",
            f"{date_formatted}.*{over_ball}",
            f"{date_formatted}.*{over_ball_alt}"
        ])
    
    # Priority 6: Over/ball only
    if over_ball:
        patterns[6].extend([over_ball, over_ball_alt])
    
    # Also try bowler patterns if available
    if bowler:
        if over_ball and date_formatted:
            patterns[5].extend([
                f"{bowler}.*{over_ball}.*{date_formatted}",
                f"{bowler}.*{over_ball_alt}.*{date_formatted}"
            ])
        if over_ball:
            patterns[6].extend([
                f"{bowler}.*{over_ball}",
                f"{bowler}.*{over_ball_alt}"
            ])
    
    # Remove empty pattern lists
    return {k: v for k, v in patterns.items() if v}

def format_date_for_matching(date_string: str) -> str:
    """
    Format date string for filename matching.
    
    Args:
        date_string: Date in various formats
    
    Returns:
        Formatted date string for matching
    """
    if not date_string:
        return ''
    
    # Try to parse common date formats
    date_formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y%m%d',
        '%d-%m-%Y',
        '%m-%d-%Y'
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_string, fmt)
            # Return multiple format options for matching
            return f"{parsed_date.strftime('%Y%m%d')}|{parsed_date.strftime('%Y-%m-%d')}|{parsed_date.strftime('%d%m%Y')}"
        except ValueError:
            continue
    
    # If parsing fails, return the original string cleaned
    return re.sub(r'[^0-9]', '', date_string)

def match_pattern_in_filename(pattern: str, filename: str) -> bool:
    """
    Check if a pattern matches in a filename.
    
    Args:
        pattern: Regex pattern to match
        filename: Filename to check
    
    Returns:
        True if pattern matches, False otherwise
    """
    # Normalize filename for matching
    filename_normalized = normalize_name(filename)
    
    # Handle multiple date formats separated by |
    if '|' in pattern:
        date_options = pattern.split('|')
        for date_option in date_options:
            if re.search(date_option, filename_normalized, re.IGNORECASE):
                return True
        return False
    
    # Regular pattern matching
    try:
        return bool(re.search(pattern, filename_normalized, re.IGNORECASE))
    except re.error:
        logger.warning(f"Invalid regex pattern: {pattern}")
        return False

def get_video_files_for_match(video_directory: str, match_identifier: str) -> List[str]:
    """
    Get all video files that might belong to a specific match.
    
    Args:
        video_directory: Directory to search for videos
        match_identifier: String to identify the match (date, teams, etc.)
    
    Returns:
        List of video file paths that match the identifier
    """
    if not os.path.exists(video_directory):
        return []
    
    matching_files = []
    normalized_identifier = normalize_name(match_identifier)
    
    for file in os.listdir(video_directory):
        if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            if normalized_identifier in normalize_name(file):
                matching_files.append(os.path.join(video_directory, file))
    
    return matching_files

def validate_ball_metadata(ball_metadata: dict) -> bool:
    """
    Validate that ball metadata has required fields.
    
    Args:
        ball_metadata: Dictionary to validate
    
    Returns:
        True if metadata is valid, False otherwise
    """
    required_fields = ['batter']
    
    for field in required_fields:
        if field not in ball_metadata or not ball_metadata[field]:
            logger.warning(f"Missing required field: {field}")
            return False
    
    return True

# Example usage and testing functions
if __name__ == "__main__":
    # Example ball metadata
    sample_ball = {
        'batter': 'Virat Kohli',
        'bowler': 'Jasprit Bumrah',
        'over': 15,
        'ball': 3,
        'match_date': '2024-03-15'
    }
    
    # Example video directory (would be a real directory in practice)
    video_dir = './test_videos'
    
    # Test the function
    result = find_video_for_ball(sample_ball, video_dir)
    if result:
        print(f"Found video: {result}")
    else:
        print("No matching video found") 