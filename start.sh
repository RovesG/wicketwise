#!/bin/bash

# WicketWise Enhanced Cricket Dashboard Startup Script
# Purpose: Clean startup of all services with proper cleanup and health checks
# Author: Assistant, Last Modified: 2025-01-19

set -e  # Exit on any error

echo "🏏 WicketWise Enhanced Cricket Dashboard Startup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_PORT=5005
HTTP_PORT=8000
API_SCRIPT="real_dynamic_cards_api_v2.py"
VENV_PATH=".venv"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✅]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠️ ]${NC} $1"
}

print_error() {
    echo -e "${RED}[❌]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ️ ]${NC} $1"
}

# Function to kill processes on specific ports
kill_port() {
    local port=$1
    local service_name=$2
    
    print_info "Checking for existing $service_name on port $port..."
    
    # Find and kill processes using the port
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ ! -z "$pids" ]; then
        print_warning "Found existing $service_name processes. Cleaning up..."
        echo $pids | xargs kill -9 2>/dev/null || true
        sleep 2
        print_status "Cleaned up existing $service_name processes"
    else
        print_status "No existing $service_name processes found"
    fi
}

# Function to kill processes by name pattern
kill_by_name() {
    local pattern=$1
    local service_name=$2
    
    print_info "Checking for existing $service_name processes..."
    
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ ! -z "$pids" ]; then
        print_warning "Found existing $service_name processes. Cleaning up..."
        pkill -f "$pattern" 2>/dev/null || true
        sleep 2
        print_status "Cleaned up existing $service_name processes"
    else
        print_status "No existing $service_name processes found"
    fi
}

# Function to check if virtual environment exists and activate it
setup_venv() {
    if [ -d "$VENV_PATH" ]; then
        print_status "Activating virtual environment..."
        source "$VENV_PATH/bin/activate"
        print_status "Virtual environment activated"
    else
        print_warning "Virtual environment not found at $VENV_PATH"
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
        print_status "Virtual environment created and activated"
        
        if [ -f "requirements.txt" ]; then
            print_info "Installing requirements..."
            pip install -r requirements.txt
            print_status "Requirements installed"
        fi
    fi
}

# Function to check if required files exist
check_requirements() {
    print_info "Checking required files..."
    
    local missing_files=()
    
    if [ ! -f "$API_SCRIPT" ]; then
        missing_files+=("$API_SCRIPT")
    fi
    
    if [ ! -f "wicketwise_dashboard.html" ]; then
        missing_files+=("wicketwise_dashboard.html")
    fi
    
    if [ ! -f "models/unified_cricket_kg.pkl" ]; then
        missing_files+=("models/unified_cricket_kg.pkl")
    fi
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_error "Missing required files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
    
    print_status "All required files found"
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=10
    local attempt=1
    
    print_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within $max_attempts seconds"
    return 1
}

# Function to test API health
test_api_health() {
    print_info "Testing API health..."
    
    local health_response=$(curl -s "http://127.0.0.1:$API_PORT/api/cards/health" 2>/dev/null || echo "failed")
    
    if [ "$health_response" = "failed" ] || [ -z "$health_response" ]; then
        print_warning "API health check couldn't connect - but API may still be starting"
        print_info "You can manually check: http://127.0.0.1:$API_PORT/api/cards/health"
        return 0  # Don't fail the startup for this
    elif echo "$health_response" | grep -q '"status":"healthy"'; then
        print_status "API health check passed"
        
        # Extract KG status
        if echo "$health_response" | grep -q '"kg_available":true'; then
            print_status "Knowledge Graph loaded successfully"
        else
            print_warning "Knowledge Graph not available - using mock data"
        fi
    else
        print_warning "API returned unexpected response, but may be working"
        print_info "Response: $health_response"
    fi
}

# Main startup sequence
main() {
    echo "🧹 Phase 1: Cleanup existing services"
    echo "------------------------------------"
    
    # Kill existing services
    kill_by_name "$API_SCRIPT" "API Server"
    kill_by_name "http.server" "HTTP Server"
    kill_by_name "admin_backend.py" "Admin Backend"
    kill_port $API_PORT "API Server"
    kill_port $HTTP_PORT "HTTP Server"
    kill_port 5001 "Admin Backend"
    
    echo ""
    echo "🔧 Phase 2: Environment setup"
    echo "-----------------------------"
    
    # Setup virtual environment
    setup_venv
    
    # Check requirements
    check_requirements
    
    echo ""
    echo "🚀 Phase 3: Starting services"
    echo "-----------------------------"
    
    # Start HTTP server for static files
    print_info "Starting HTTP server on port $HTTP_PORT..."
    python -m http.server $HTTP_PORT > /dev/null 2>&1 &
    HTTP_PID=$!
    
    # Wait for HTTP server
    if wait_for_service "http://127.0.0.1:$HTTP_PORT" "HTTP Server"; then
        print_status "HTTP Server started (PID: $HTTP_PID)"
    else
        print_error "Failed to start HTTP Server"
        exit 1
    fi
    
    # Start API server
    print_info "Starting Enhanced API server on port $API_PORT..."
    python "$API_SCRIPT" > api.log 2>&1 &
    API_PID=$!
    
    # Start admin backend server (for match enrichment)
    print_info "Starting Admin Backend server on port 5001..."
    python admin_backend.py > admin_backend.log 2>&1 &
    ADMIN_PID=$!
    
    # Wait for API server
    if wait_for_service "http://127.0.0.1:$API_PORT/api/cards/health" "API Server"; then
        print_status "API Server started (PID: $API_PID)"
    else
        print_error "Failed to start API Server"
        print_info "Check api.log for details"
        exit 1
    fi
    
    # Wait for admin backend server
    if wait_for_service "http://127.0.0.1:5001/api/health" "Admin Backend"; then
        print_status "Admin Backend started (PID: $ADMIN_PID)"
    else
        print_error "Failed to start Admin Backend"
        print_info "Check admin_backend.log for details"
        exit 1
    fi
    
    echo ""
    echo "🧪 Phase 4: Health checks"
    echo "-------------------------"
    
    # Test API health
    test_api_health
    
    echo ""
    echo "✨ Phase 5: Ready to use!"
    echo "========================"
    
    print_status "WicketWise Enhanced Cricket Dashboard is ready!"
    echo ""
    echo "📊 URLs:"
    echo "  • Main Dashboard:    http://127.0.0.1:$HTTP_PORT/wicketwise_dashboard.html"
    echo "  • Admin Panel:       http://127.0.0.1:$HTTP_PORT/wicketwise_admin_simple.html"
    echo "  • Standalone Cards:  http://127.0.0.1:$HTTP_PORT/enhanced_player_cards_ui.html"
    echo "  • API Health:        http://127.0.0.1:$API_PORT/api/cards/health"
    echo ""
    echo "🔧 Service Info:"
    echo "  • HTTP Server PID:   $HTTP_PID"
    echo "  • API Server PID:    $API_PID"
    echo "  • Admin Backend PID: $ADMIN_PID"
    echo "  • API Logs:          api.log"
    echo "  • Admin Logs:        admin_backend.log"
    echo ""
    echo "🎯 Features Available:"
    echo "  • Real Knowledge Graph data (11,997 nodes)"
    echo "  • Comprehensive player statistics"
    echo "  • Format-specific analytics"
    echo "  • Persona-based analysis"
    echo "  • Dynamic player cards"
    echo "  • OpenAI Match Enrichment (weather, teams, venues)"
    echo ""
    echo "🛑 To stop services:"
    echo "  kill $HTTP_PID $API_PID $ADMIN_PID"
    echo "  # Or use: ./stop.sh"
    echo ""
    
    # Save PIDs for stop script
    echo "$HTTP_PID" > .http_server.pid
    echo "$API_PID" > .api_server.pid
    echo "$ADMIN_PID" > .admin_backend.pid
    
    print_status "Startup complete! 🎉"
}

# Trap to cleanup on script exit
cleanup() {
    if [ ! -z "$HTTP_PID" ] && kill -0 $HTTP_PID 2>/dev/null; then
        print_warning "Cleaning up HTTP server..."
        kill $HTTP_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$API_PID" ] && kill -0 $API_PID 2>/dev/null; then
        print_warning "Cleaning up API server..."
        kill $API_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$ADMIN_PID" ] && kill -0 $ADMIN_PID 2>/dev/null; then
        print_warning "Cleaning up Admin Backend..."
        kill $ADMIN_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

# Run main function
main

# Keep script running to maintain services
print_info "Services are running. Press Ctrl+C to stop all services."
wait