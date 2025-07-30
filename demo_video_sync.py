# Purpose: Demonstration of video_sync.py functionality with real cricket video files
# Author: Shamus Rae, Last Modified: July 17, 2025

import os
import sys
from video_sync import find_video_for_ball, get_video_files_for_match, validate_ball_metadata

def demo_video_sync():
    """Demonstrate video_sync functionality with real cricket video files."""
    
    print("🏏 Cricket Video Sync Demonstration")
    print("=" * 50)
    
    # Define the video directory path
    video_directory = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /SA(Kumar)_Yadav_Batter_Balls'
    
    # Check if directory exists
    if not os.path.exists(video_directory):
        print(f"❌ Video directory not found: {video_directory}")
        return
    
    # Get all video files in the directory
    video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]
    print(f"📁 Found {len(video_files)} video files in directory")
    print(f"📂 Directory: {video_directory}")
    print()
    
    # Show first few video files as examples
    print("📺 Sample video files:")
    for i, file in enumerate(video_files[:5]):
        print(f"  {i+1}. {file}")
    if len(video_files) > 5:
        print(f"  ... and {len(video_files) - 5} more files")
    print()
    
    # Test cases for ball metadata matching
    test_cases = [
        {
            'name': 'Match 1, Over 7, Ball 2',
            'metadata': {
                'batter': 'SA(Kumar) Yadav',
                'over': 7,
                'ball': 2,
                'match_date': '2024-07-01'
            }
        },
        {
            'name': 'Match 1, Over 8, Ball 1',
            'metadata': {
                'batter': 'SA(Kumar) Yadav',
                'over': 8,
                'ball': 1
            }
        },
        {
            'name': 'Match 1, Over 9, Ball 4',
            'metadata': {
                'batter': 'SA(Kumar) Yadav',
                'over': 9,
                'ball': 4
            }
        },
        {
            'name': 'Match 1, Over 10, Ball 5',
            'metadata': {
                'batter': 'SA(Kumar) Yadav',
                'over': 10,
                'ball': 5
            }
        },
        {
            'name': 'Match 1, Over 11, Ball 3',
            'metadata': {
                'batter': 'SA(Kumar) Yadav',
                'over': 11,
                'ball': 3
            }
        },
        {
            'name': 'Non-existent ball (Over 99, Ball 99)',
            'metadata': {
                'batter': 'SA(Kumar) Yadav',
                'over': 99,
                'ball': 99
            }
        },
        {
            'name': 'Different player name',
            'metadata': {
                'batter': 'Virat Kohli',
                'over': 7,
                'ball': 2
            }
        }
    ]
    
    print("🎯 Testing ball metadata matching:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Metadata: {test_case['metadata']}")
        
        # Validate metadata
        if validate_ball_metadata(test_case['metadata']):
            print("   ✅ Metadata validation: PASSED")
        else:
            print("   ❌ Metadata validation: FAILED")
            continue
        
        # Find video for ball
        result = find_video_for_ball(test_case['metadata'], video_directory)
        
        if result:
            filename = os.path.basename(result)
            print(f"   🎥 Found video: {filename}")
            print(f"   📍 Full path: {result}")
            
            # Check if file actually exists
            if os.path.exists(result):
                file_size = os.path.getsize(result)
                print(f"   📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            else:
                print("   ⚠️  File path returned but file doesn't exist")
        else:
            print("   ❌ No matching video found")
    
    print("\n" + "=" * 50)
    print("🔍 Testing match-wide video search:")
    print("-" * 50)
    
    # Test getting all videos for a match
    match_videos = get_video_files_for_match(video_directory, 'Match_1_1')
    print(f"📹 Found {len(match_videos)} videos for Match 1:")
    for i, video in enumerate(match_videos[:10]):  # Show first 10
        filename = os.path.basename(video)
        print(f"   {i+1}. {filename}")
    if len(match_videos) > 10:
        print(f"   ... and {len(match_videos) - 10} more videos")
    
    print("\n" + "=" * 50)
    print("📈 Summary:")
    print("-" * 50)
    
    # Count successful matches
    successful_matches = sum(1 for test_case in test_cases 
                           if find_video_for_ball(test_case['metadata'], video_directory) is not None)
    
    print(f"✅ Successful matches: {successful_matches}/{len(test_cases)}")
    print(f"📁 Total video files: {len(video_files)}")
    print(f"🎥 Video format: MP4")
    print(f"🏏 Player: SA(Kumar) Yadav")
    print(f"🏟️ Match format: Match_<match>_<innings>_<over>_<ball>.mp4")
    
    print("\n🎯 Video Sync Demo Complete!")

def demo_pattern_analysis():
    """Analyze video file naming patterns."""
    print("\n🔍 Video File Pattern Analysis")
    print("=" * 50)
    
    video_directory = '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /SA(Kumar)_Yadav_Batter_Balls'
    
    if not os.path.exists(video_directory):
        print(f"❌ Video directory not found: {video_directory}")
        return
    
    video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]
    
    # Analyze patterns
    patterns = {}
    overs = set()
    balls = set()
    
    for file in video_files:
        parts = file.replace('.mp4', '').split('_')
        if len(parts) >= 5:
            try:
                match_num = parts[1]
                innings = parts[2]
                over = int(parts[3])
                ball = int(parts[4])
                
                pattern = f"Match_{match_num}_{innings}_<over>_<ball>.mp4"
                patterns[pattern] = patterns.get(pattern, 0) + 1
                
                overs.add(over)
                balls.add(ball)
            except (ValueError, IndexError):
                pass
    
    print(f"📊 Pattern Analysis:")
    for pattern, count in patterns.items():
        print(f"   {pattern}: {count} files")
    
    print(f"\n🎯 Over Range: {min(overs) if overs else 'N/A'} - {max(overs) if overs else 'N/A'}")
    print(f"🎯 Ball Range: {min(balls) if balls else 'N/A'} - {max(balls) if balls else 'N/A'}")
    print(f"🎯 Total Overs: {len(overs)}")
    print(f"🎯 Total Balls: {len(balls)}")

if __name__ == "__main__":
    try:
        demo_video_sync()
        demo_pattern_analysis()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        sys.exit(1) 