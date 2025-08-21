#!/bin/bash

# WicketWise Match Enrichment Demo Startup Script
echo "🚀 WicketWise Match Enrichment Demo"
echo "======================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate
echo "✅ Virtual environment activated"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OpenAI API key not found in environment"
    echo "   You can set it through the admin interface or export OPENAI_API_KEY=your_key_here"
else
    echo "✅ OpenAI API key configured"
fi

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
pkill -f "python.*admin_backend.py" 2>/dev/null || true
pkill -f "python.*http.server.*8000" 2>/dev/null || true

# Wait for processes to stop
sleep 2

# Start HTTP server for static files
echo "🌐 Starting static file server on port 8000..."
python -m http.server 8000 > /dev/null 2>&1 &
HTTP_PID=$!

# Wait for HTTP server to start
sleep 2

# Start admin backend
echo "🔧 Starting admin backend on port 5001..."
python admin_backend.py > admin_backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Health check
echo "🏥 Checking service health..."
if curl -s http://127.0.0.1:8000/wicketwise_admin_simple.html > /dev/null; then
    echo "✅ Static server running on http://127.0.0.1:8000"
else
    echo "❌ Static server failed to start"
fi

if curl -s http://127.0.0.1:5001/api/health > /dev/null; then
    echo "✅ Admin backend running on http://127.0.0.1:5001"
else
    echo "❌ Admin backend failed to start"
fi

echo ""
echo "🎯 MATCH ENRICHMENT DEMO READY!"
echo "======================================"
echo "Admin Interface: http://127.0.0.1:8000/wicketwise_admin_simple.html"
echo ""
echo "📋 Quick Start Guide:"
echo "1. Open the admin interface above"
echo "2. Go to 'API Keys' tab and enter your OpenAI API key"
echo "3. Go to 'Match Enrichment' tab"
echo "4. Configure settings (start with 10-20 matches for testing)"
echo "5. Click 'Start Match Enrichment'"
echo ""
echo "💰 Cost Estimate:"
echo "• 10 matches = ~$0.20"
echo "• 50 matches = ~$1.00"
echo "• Full dataset (3,987 matches) = ~$80"
echo ""
echo "📊 What You'll Get:"
echo "• Weather data for each match"
echo "• Team squads and player roles"
echo "• Venue coordinates and timezones"
echo "• Match context (toss, format, timing)"
echo ""
echo "📁 Output Files:"
echo "• enriched_data/enriched_betting_matches.json"
echo "• enriched_data/enrichment_summary.txt"
echo ""
echo "Press Ctrl+C to stop all services"
echo "Logs: admin_backend.log"

# Create a function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $HTTP_PID 2>/dev/null || true
    kill $BACKEND_PID 2>/dev/null || true
    pkill -f "python.*admin_backend.py" 2>/dev/null || true
    pkill -f "python.*http.server.*8000" 2>/dev/null || true
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running
wait
