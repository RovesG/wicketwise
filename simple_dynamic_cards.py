# Purpose: Simplified Dynamic Player Cards that actually works
# Author: WicketWise Team, Last Modified: August 19, 2024

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import logging
from datetime import datetime
import random
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global player data
player_data = None

def load_players():
    """Load player data from CSV"""
    global player_data
    try:
        people_csv = "/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data/people.csv"
        player_data = pd.read_csv(people_csv)
        logger.info(f"✅ Loaded {len(player_data)} players")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading players: {e}")
        # Create mock data
        player_data = pd.DataFrame({
            'identifier': ['virat_kohli', 'ms_dhoni', 'rohit_sharma', 'kl_rahul', 'hardik_pandya'],
            'name': ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Hardik Pandya'],
            'unique_name': ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Hardik Pandya']
        })
        logger.info(f"✅ Using mock data with {len(player_data)} players")
        return False

def generate_mock_card_data(player_name):
    """Generate mock card data for a player"""
    # Use player name hash for consistent data
    seed = hash(player_name) % 1000
    random.seed(seed)
    
    batting_avg = round(random.uniform(25.0, 55.0), 1)
    strike_rate = round(random.uniform(110.0, 160.0), 1)
    form_rating = round(random.uniform(5.0, 9.5), 1)
    
    form_descriptions = ["Poor Form", "Average Form", "Good Form", "Hot Form", "Exceptional Form"]
    form_index = min(int(form_rating / 2), len(form_descriptions) - 1)
    
    # Generate recent matches
    teams = ['MI', 'CSK', 'SRH', 'KKR', 'RR', 'DC', 'RCB', 'PBKS', 'GT', 'LSG']
    recent_matches = []
    for i in range(5):
        score = random.randint(15, 95)
        is_not_out = i == 0 and random.random() > 0.7
        opponent = random.choice(teams)
        recent_matches.append({
            'score': f"{score}{'*' if is_not_out else ''}",
            'opponent': opponent,
            'days_ago': (i * 4 + 3)
        })
    
    # Generate live data
    ball_outcomes = ['1', '2', '4', '6', '0', 'W', '.']
    last_6_balls = [random.choice(ball_outcomes) for _ in range(6)]
    
    # Betting intelligence
    market_odds = round(random.uniform(1.4, 2.5), 2)
    model_odds = round(random.uniform(1.3, 2.3), 2)
    expected_value = round(((1/model_odds * market_odds) - 1) * 100, 1)
    
    return {
        'player_name': player_name,
        'batting_avg': batting_avg,
        'strike_rate': strike_rate,
        'recent_form': form_descriptions[form_index],
        'form_rating': form_rating,
        'powerplay_sr': round(random.uniform(110.0, 150.0), 1),
        'death_overs_sr': round(random.uniform(130.0, 180.0), 1),
        'vs_pace_avg': round(random.uniform(28.0, 45.0), 1),
        'vs_spin_avg': round(random.uniform(25.0, 42.0), 1),
        'pressure_rating': round(random.uniform(5.0, 9.5), 1),
        'recent_matches': recent_matches,
        'current_match_status': random.choice(['Not Playing', 'Batting', 'Bowling', 'Fielding']),
        'last_6_balls': last_6_balls,
        'betting_odds': {
            'market_odds': market_odds,
            'model_odds': model_odds,
            'expected_value': expected_value
        },
        'value_opportunities': [{
            'market': 'Runs Over 30.5',
            'market_odds': market_odds,
            'model_odds': model_odds,
            'expected_value': expected_value,
            'confidence': random.randint(65, 90)
        }] if abs(expected_value) > 5 else [],
        'profile_image_url': f"https://via.placeholder.com/150x150?text={player_name.replace(' ', '+')}&bg=1f2937&color=ffffff",
        'last_updated': datetime.now().isoformat(),
        'data_sources': ['Mock', 'Consistent_Seed']
    }

@app.route('/api/cards/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Simple Dynamic Player Cards API',
        'players_available': len(player_data) if player_data is not None else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/cards/autocomplete', methods=['GET'])
def autocomplete():
    """Get autocomplete suggestions for player names"""
    try:
        partial = request.args.get('partial', '').strip()
        limit = int(request.args.get('limit', 5))
        
        if not partial or len(partial) < 2:
            return jsonify({
                'success': True,
                'partial': partial,
                'suggestions': [],
                'count': 0
            })
        
        # Filter players whose names start with the partial name (case insensitive)
        mask = player_data['name'].str.lower().str.startswith(partial.lower())
        suggestions = player_data[mask]['name'].head(limit).tolist()
        
        logger.info(f"🔍 Autocomplete '{partial}' -> {len(suggestions)} results")
        
        return jsonify({
            'success': True,
            'partial': partial,
            'suggestions': suggestions,
            'count': len(suggestions)
        })
        
    except Exception as e:
        logger.error(f"Error in autocomplete: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'suggestions': [],
            'count': 0
        }), 500

@app.route('/api/cards/search', methods=['GET'])
def search_players():
    """Search for players"""
    try:
        query = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400
        
        # Fuzzy matching on player names
        mask = player_data['name'].str.contains(query, case=False, na=False)
        matches = player_data[mask].head(limit)
        
        results = []
        for _, player in matches.iterrows():
            results.append({
                'identifier': player['identifier'],
                'name': player['name'],
                'unique_name': player['unique_name'],
                'display_name': player['name']
            })
        
        logger.info(f"🔍 Search '{query}' -> {len(results)} results")
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in player search: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/generate', methods=['POST'])
def generate_card():
    """Generate a dynamic player card"""
    try:
        data = request.json
        player_name = data.get('player_name', '').strip()
        persona = data.get('persona', 'betting')
        
        if not player_name:
            return jsonify({'error': 'player_name is required'}), 400
        
        logger.info(f"🎴 Generating card for '{player_name}' (persona: {persona})")
        
        # Generate the card data
        card_data = generate_mock_card_data(player_name)
        
        return jsonify({
            'success': True,
            'player_name': player_name,
            'persona': persona,
            'card_data': card_data
        })
        
    except Exception as e:
        logger.error(f"Error generating card: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/popular', methods=['GET'])
def get_popular_players():
    """Get popular players"""
    try:
        popular_names = ['Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Hardik Pandya']
        
        popular_players = []
        for name in popular_names:
            # Try to find the player in our data
            mask = player_data['name'].str.contains(name, case=False, na=False)
            matches = player_data[mask]
            
            if not matches.empty:
                player = matches.iloc[0]
                popular_players.append({
                    'identifier': player['identifier'],
                    'name': player['name'],
                    'unique_name': player['unique_name']
                })
            else:
                # Add as fallback
                popular_players.append({
                    'identifier': hashlib.md5(name.encode()).hexdigest()[:8],
                    'name': name,
                    'unique_name': name
                })
        
        return jsonify({
            'success': True,
            'popular_players': popular_players,
            'count': len(popular_players)
        })
        
    except Exception as e:
        logger.error(f"Error getting popular players: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cards/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'success': True,
        'stats': {
            'total_players': len(player_data) if player_data is not None else 0,
            'api_version': '1.0.0',
            'status': 'operational'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Simple Dynamic Player Cards API...")
    
    # Load player data
    if load_players():
        print(f"✅ Successfully loaded {len(player_data)} players")
    else:
        print("⚠️ Using mock player data")
    
    print("\n📡 Available API Endpoints:")
    print("  GET  /api/cards/health - Health check")
    print("  GET  /api/cards/autocomplete?partial=<text> - Autocomplete")
    print("  GET  /api/cards/search?q=<query> - Search players")
    print("  POST /api/cards/generate - Generate single card")
    print("  GET  /api/cards/popular - Get popular players")
    print("  GET  /api/cards/stats - System statistics")
    
    print("\n🌐 Server starting on http://127.0.0.1:5003")
    print("🎯 Ready to serve dynamic player cards!")
    
    app.run(host='127.0.0.1', port=5003, debug=False, threaded=True)
