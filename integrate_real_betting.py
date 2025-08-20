# Purpose: Integration example - Connect real betting intelligence to dashboard
# Author: WicketWise Team, Last Modified: August 19, 2024

"""
INTEGRATION GUIDE: How to replace mock betting data with real KG + GNN calculations

Current State: Mock data in wicketwise_dashboard.html
Target State: Real calculations using your Knowledge Graph + GNN system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_betting_intelligence import RealBettingIntelligenceEngine, get_real_betting_intelligence
from crickformers.chat.unified_kg_query_engine import UnifiedKGQueryEngine
import pandas as pd

def demo_real_betting_integration():
    """
    Demonstration of how to integrate real betting intelligence
    """
    print("🎯 WicketWise Real Betting Intelligence Integration Demo")
    print("=" * 60)
    
    # Step 1: Load your existing systems
    print("📊 Loading Knowledge Graph...")
    try:
        # This uses your existing KG system
        kg_path = "models/unified_knowledge_graph.pkl"
        if os.path.exists(kg_path):
            import pickle
            with open(kg_path, 'rb') as f:
                knowledge_graph = pickle.load(f)
            
            kg_query_engine = UnifiedKGQueryEngine(knowledge_graph)
            print(f"✅ Loaded KG: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        else:
            print("⚠️  KG file not found, using mock query engine")
            kg_query_engine = MockKGQueryEngine()
    
    except Exception as e:
        print(f"⚠️  Using mock KG query engine: {e}")
        kg_query_engine = MockKGQueryEngine()
    
    # Step 2: Load player index
    print("👥 Loading player index...")
    try:
        people_csv_path = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/people.csv"
        if os.path.exists(people_csv_path):
            player_index = pd.read_csv(people_csv_path)
            print(f"✅ Loaded {len(player_index)} players for search")
        else:
            print("⚠️  Using mock player index")
            player_index = pd.DataFrame({'name': ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma']})
    except Exception as e:
        print(f"⚠️  Using mock player index: {e}")
        player_index = pd.DataFrame({'name': ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma']})
    
    # Step 3: Initialize real betting intelligence engine
    print("🧠 Initializing Real Betting Intelligence Engine...")
    gnn_predictor = MockGNNPredictor()  # Replace with your real GNN
    
    betting_engine = RealBettingIntelligenceEngine(
        kg_query_engine=kg_query_engine,
        gnn_predictor=gnn_predictor,
        player_index=player_index
    )
    
    # Step 4: Generate real betting intelligence
    print("\n🎯 Generating Real Betting Intelligence for Virat Kohli...")
    print("-" * 50)
    
    situation = {
        'bowling_type': 'pace',
        'phase': 'middle',
        'venue': 'Wankhede Stadium',
        'pressure_level': 'medium'
    }
    
    intelligence = betting_engine.calculate_runs_probability('Virat Kohli', 30.5, situation)
    
    # Step 5: Display results in the same format as dashboard
    print(f"""
🎰 REAL Betting Intelligence for {intelligence.player_name}:

Value Opportunity: Runs Over 30.5                    [+EV {intelligence.expected_value:+.1f}%]

Market Odds: {intelligence.market_odds:.2f} ({intelligence.market_probability:.1%})     Model Odds: {intelligence.model_odds:.2f} ({intelligence.model_probability:.1%})
Bookmaker implied probability              Our calculated probability

Confidence: {intelligence.confidence:.0%} (σ={intelligence.risk_assessment['volatility']:.1f}, n={intelligence.sample_size} similar situations)
Reasoning: Form trend ({intelligence.reasoning.get('form_trend', 0):+.0f}%), matchup advantage ({intelligence.reasoning.get('matchup_advantage', 0):+.0f}%), venue factor ({intelligence.reasoning.get('venue_factor', 0):+.0f}%)
Risk Level: {intelligence.risk_assessment['risk_level']} (volatility σ={intelligence.volatility:.1f}, consistency {intelligence.consistency:.0%})
    """)
    
    print("\n" + "=" * 60)
    print("🔧 INTEGRATION STEPS:")
    print("1. Replace mock data in wicketwise_dashboard.html")
    print("2. Add API endpoint in enhanced_dashboard_api.py")
    print("3. Connect to your real KG + GNN systems")
    print("4. Add betting API integration for market odds")
    print("=" * 60)

class MockKGQueryEngine:
    """Mock KG query engine for demonstration"""
    def query_player_comprehensive(self, player_name):
        return {
            'recent_innings': [67, 45, 23, 89, 34, 56, 78, 41, 29, 92],
            'batting_avg': 42.5,
            'strike_rate': 135.2,
            'powerplay_strike_rate': 142.0,
            'death_overs_strike_rate': 158.0,
            'vs_pace_average': 45.2,
            'vs_spin_average': 38.7,
            'pressure_performance': 8.9,
            'venue_stats': {'Wankhede Stadium': 0.15, 'Eden Gardens': 0.08},
            'form_trend': 0.23,
            'consistency': 0.87
        }

class MockGNNPredictor:
    """Mock GNN predictor for demonstration"""
    def predict_runs_probability(self, **kwargs):
        return 0.658  # 65.8% probability

def create_real_api_endpoint():
    """
    Example of how to add this to enhanced_dashboard_api.py
    """
    api_code = '''
@app.route('/api/enhanced/real-betting-intelligence', methods=['POST'])
def get_real_betting_intelligence_endpoint():
    """Get real betting intelligence using KG + GNN"""
    try:
        data = request.json
        player_name = data.get('player_name', 'Virat Kohli')
        threshold = data.get('threshold', 30.5)
        situation = data.get('situation', {})
        
        # Use real betting intelligence engine
        intelligence = betting_engine.calculate_runs_probability(
            player_name, threshold, situation
        )
        
        return jsonify({
            'success': True,
            'intelligence': {
                'market_odds': f"{intelligence.market_odds:.2f}",
                'market_probability': f"{intelligence.market_probability:.1%}",
                'model_odds': f"{intelligence.model_odds:.2f}",
                'model_probability': f"{intelligence.model_probability:.1%}",
                'expected_value': f"{intelligence.expected_value:+.1f}%",
                'confidence': f"{intelligence.confidence:.0%}",
                'reasoning': intelligence.reasoning,
                'risk_assessment': intelligence.risk_assessment,
                'sample_size': intelligence.sample_size
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    '''
    
    print("📝 API Endpoint Code:")
    print(api_code)

if __name__ == "__main__":
    demo_real_betting_integration()
    print("\n" + "=" * 60)
    create_real_api_endpoint()
