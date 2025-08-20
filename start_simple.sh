#!/bin/bash

# Simple WicketWise Startup Script
echo "🏏 Starting WicketWise Services..."

# Kill existing processes
pkill -f "real_dynamic_cards_api" 2>/dev/null || true
pkill -f "http.server 8000" 2>/dev/null || true
lsof -ti:5005 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo "✅ Cleaned up existing processes"

# Activate virtual environment
source .venv/bin/activate

# Start HTTP server
echo "🌐 Starting HTTP server on port 8000..."
nohup python -m http.server 8000 > http.log 2>&1 &
HTTP_PID=$!
echo "✅ HTTP Server started (PID: $HTTP_PID)"

# Start API server
echo "🚀 Starting API server on port 5005..."
nohup python real_dynamic_cards_api_v2.py > api.log 2>&1 &
API_PID=$!
echo "✅ API Server started (PID: $API_PID)"

# Wait a moment for startup
sleep 3

# Test connections
echo "🧪 Testing services..."
if curl -s http://127.0.0.1:8000 > /dev/null; then
    echo "✅ HTTP Server responding"
else
    echo "❌ HTTP Server not responding"
fi

if curl -s http://127.0.0.1:5005/api/cards/health > /dev/null; then
    echo "✅ API Server responding"
else
    echo "❌ API Server not responding"
fi

echo ""
echo "🎯 Services Started:"
echo "  📊 Dashboard: http://127.0.0.1:8000/wicketwise_dashboard.html"
echo "  🎴 Cards:     http://127.0.0.1:8000/enhanced_player_cards_ui.html"
echo "  🔧 API:       http://127.0.0.1:5005/api/cards/health"
echo ""
echo "📝 Logs:"
echo "  HTTP: http.log"
echo "  API:  api.log"
echo ""
echo "🛑 To stop: pkill -f 'real_dynamic_cards_api' && pkill -f 'http.server'"
echo ""
echo "✅ Startup complete!"
