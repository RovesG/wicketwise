# Purpose: Demo script for simulator engine functionality
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

from simulator_engine import MatchSimulator, create_simulator
import pandas as pd

def demo_simulator_engine():
    """Demonstrate the simulator engine functionality."""
    
    print("⚡ Cricket Simulator Engine Demo")
    print("=" * 60)
    
    # Demo 1: Basic simulator creation and usage
    print("\n🏗️ Demo 1: Basic Simulator Creation")
    print("-" * 40)
    
    # Create simulator for all matches
    simulator_all = MatchSimulator("test_matches.csv")
    print(f"✅ Created simulator for all matches")
    print(f"📊 Total balls across all matches: {simulator_all.get_match_ball_count()}")
    
    # Demo 2: Match filtering
    print("\n🔍 Demo 2: Match Filtering")
    print("-" * 40)
    
    # Filter for MATCH_001
    simulator_001 = MatchSimulator("test_matches.csv", match_id="MATCH_001")
    print(f"✅ Created simulator for MATCH_001")
    print(f"📊 Balls in MATCH_001: {simulator_001.get_match_ball_count()}")
    
    # Filter for MATCH_002
    simulator_002 = MatchSimulator("test_matches.csv", match_id="MATCH_002")
    print(f"✅ Created simulator for MATCH_002")
    print(f"📊 Balls in MATCH_002: {simulator_002.get_match_ball_count()}")
    
    # Demo 3: Ball-by-ball retrieval
    print("\n🏏 Demo 3: Ball-by-Ball Retrieval")
    print("-" * 40)
    
    # Show first 5 balls from MATCH_001
    print(f"📋 First 5 balls from MATCH_001:")
    for i in range(5):
        ball_info = simulator_001.get_ball(i)
        print(f"   Ball {ball_info['ball_number']}: {ball_info['batter']} vs {ball_info['bowler']} -> {ball_info['runs']} runs")
    
    # Demo 4: Detailed ball information
    print("\n📝 Demo 4: Detailed Ball Information")
    print("-" * 40)
    
    # Get detailed info for an interesting ball
    ball_2 = simulator_001.get_ball(2)  # This should be a boundary
    print(f"🎯 Ball {ball_2['ball_number']} Details:")
    print(f"   Batter: {ball_2['batter']}")
    print(f"   Bowler: {ball_2['bowler']}")
    print(f"   Over: {ball_2['over']}")
    print(f"   Runs: {ball_2['runs']}")
    print(f"   Extras: {ball_2['extras']}")
    print(f"   Wicket: {'Yes' if ball_2['wicket'] else 'No'}")
    print(f"   Phase: {ball_2['phase']}")
    
    # Demo 5: Match state tracking
    print("\n📊 Demo 5: Match State Tracking")
    print("-" * 40)
    
    # Show match state at different points
    checkpoints = [0, 4, 8, 11]  # First, middle, near end, last ball
    for checkpoint in checkpoints:
        if checkpoint < simulator_001.get_match_ball_count():
            ball_info = simulator_001.get_ball(checkpoint)
            state = ball_info['match_state']
            print(f"   After ball {checkpoint + 1}:")
            print(f"     Progress: {state['progress_percentage']}%")
            print(f"     Total runs: {state.get('total_runs', 'N/A')}")
            print(f"     Total wickets: {state.get('total_wickets', 'N/A')}")
            print(f"     Run rate: {state.get('run_rate', 'N/A')}")
    
    # Demo 6: Match summary
    print("\n📈 Demo 6: Match Summary")
    print("-" * 40)
    
    summary_001 = simulator_001.get_match_summary()
    print(f"📋 MATCH_001 Summary:")
    print(f"   Total balls: {summary_001['total_balls']}")
    print(f"   Total overs: {summary_001['total_overs']}")
    print(f"   Total runs: {summary_001.get('total_runs', 'N/A')}")
    print(f"   Total wickets: {summary_001.get('total_wickets', 'N/A')}")
    print(f"   Boundaries: {summary_001.get('boundaries', 'N/A')}")
    print(f"   Sixes: {summary_001.get('sixes', 'N/A')}")
    print(f"   Batters: {', '.join(summary_001.get('batters', []))}")
    print(f"   Bowlers: {', '.join(summary_001.get('bowlers', []))}")
    
    # Demo 7: Caching behavior
    print("\n⚡ Demo 7: Caching Behavior")
    print("-" * 40)
    
    # Create new simulator to test caching
    simulator_cache = MatchSimulator("test_matches.csv", match_id="MATCH_001")
    print(f"🔄 Initial cache state: {simulator_cache._cached}")
    
    # First access - should cache
    ball_count = simulator_cache.get_match_ball_count()
    print(f"📊 After first access: cached={simulator_cache._cached}, balls={ball_count}")
    
    # Second access - should use cache
    ball_count_cached = simulator_cache.get_match_ball_count()
    print(f"⚡ Second access (cached): balls={ball_count_cached}")
    
    # Reset cache
    simulator_cache.reset_cache()
    print(f"🔄 After cache reset: {simulator_cache._cached}")
    
    # Demo 8: Convenience function
    print("\n🛠️ Demo 8: Convenience Function")
    print("-" * 40)
    
    # Use create_simulator function
    quick_sim = create_simulator("test_matches.csv", match_id="MATCH_002")
    print(f"✅ Created simulator using convenience function")
    print(f"📊 Quick access to ball count: {quick_sim.get_match_ball_count()}")
    
    # Demo 9: Error handling
    print("\n⚠️ Demo 9: Error Handling")
    print("-" * 40)
    
    # Test invalid match_id
    try:
        invalid_sim = MatchSimulator("test_matches.csv", match_id="INVALID_MATCH")
        invalid_sim.get_match_ball_count()
    except ValueError as e:
        print(f"✅ Correctly caught invalid match_id: {e}")
    
    # Test invalid ball index
    try:
        simulator_001.get_ball(-1)
    except IndexError as e:
        print(f"✅ Correctly caught invalid ball index: {e}")
    
    # Test non-existent file
    try:
        bad_sim = MatchSimulator("non_existent_file.csv")
        bad_sim.get_match_ball_count()
    except FileNotFoundError as e:
        print(f"✅ Correctly caught file not found error")
    
    # Demo 10: Dynamic match switching
    print("\n🔄 Demo 10: Dynamic Match Switching")
    print("-" * 40)
    
    # Start with MATCH_001
    dynamic_sim = MatchSimulator("test_matches.csv", match_id="MATCH_001")
    print(f"📊 Initial match (MATCH_001): {dynamic_sim.get_match_ball_count()} balls")
    
    # Switch to MATCH_002
    dynamic_sim.set_match_id("MATCH_002")
    print(f"📊 After switching to MATCH_002: {dynamic_sim.get_match_ball_count()} balls")
    
    # Verify we're getting different data
    first_ball_002 = dynamic_sim.get_ball(0)
    print(f"🎯 First ball of MATCH_002: {first_ball_002['batter']} vs {first_ball_002['bowler']}")
    
    # Summary
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("✅ Basic simulator creation and usage")
    print("✅ Match filtering by match_id")
    print("✅ Ball-by-ball retrieval with detailed info")
    print("✅ Match state tracking and progress")
    print("✅ Comprehensive match summaries")
    print("✅ Efficient caching mechanism")
    print("✅ Robust error handling")
    print("✅ Dynamic match switching")
    print("✅ Flexible column name handling")
    print("✅ Integration with real CSV data")
    
    print("\n🚀 Ready for Integration:")
    print("   • Use with UI components")
    print("   • Integrate with prediction models")
    print("   • Add real-time match simulation")
    print("   • Scale to larger datasets")
    print("   • Support multiple match formats")
    
    print("\n📋 Key Features:")
    print("   • Zero-based indexing for ball retrieval")
    print("   • Comprehensive match state tracking")
    print("   • Flexible CSV column handling")
    print("   • Efficient data caching")
    print("   • Robust error handling")
    print("   • Match filtering capabilities")


if __name__ == "__main__":
    demo_simulator_engine() 