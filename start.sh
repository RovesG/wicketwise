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
API_GATEWAY_PORT=5005
ADMIN_BACKEND_PORT=5001
HTTP_PORT=8000
API_GATEWAY_SCRIPT="real_dynamic_cards_api_v2.py"
ADMIN_BACKEND_SCRIPT="admin_backend.py"
VENV_PATH=".venv"
ENVIRONMENT="development"

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
            pip install --upgrade pip
            pip install -r requirements.txt
            
            # Install additional dependencies from refactoring
            print_info "Installing additional dependencies..."
            pip install fastapi uvicorn aiofiles httpx pytest-asyncio pytest-benchmark bandit safety psutil
            pip install pyjwt cryptography python-multipart
            
            print_status "All requirements installed"
        fi
    fi
}

# Function to check if required files exist
check_requirements() {
    print_info "Checking required files..."
    
    local missing_files=()
    local optional_files=()
    
    # Critical files
    if [ ! -f "$API_GATEWAY_SCRIPT" ]; then
        missing_files+=("$API_GATEWAY_SCRIPT")
    fi
    
    if [ ! -f "$ADMIN_BACKEND_SCRIPT" ]; then
        missing_files+=("$ADMIN_BACKEND_SCRIPT")
    fi
    
    if [ ! -f "wicketwise_dashboard.html" ]; then
        missing_files+=("wicketwise_dashboard.html")
    fi
    
    if [ ! -f "wicketwise_admin_redesigned.html" ]; then
        missing_files+=("wicketwise_admin_redesigned.html")
    fi
    
    if [ ! -f "security_framework.py" ]; then
        missing_files+=("security_framework.py")
    fi
    
    if [ ! -f "unified_configuration.py" ]; then
        missing_files+=("unified_configuration.py")
    fi
    
    # Optional files (warn but don't fail)
    if [ ! -f "models/unified_cricket_kg.pkl" ]; then
        optional_files+=("models/unified_cricket_kg.pkl (will use mock data)")
    fi
    
    if [ ! -f "service_container.py" ]; then
        optional_files+=("service_container.py (microservices disabled)")
    fi
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_error "Missing critical files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
    
    if [ ${#optional_files[@]} -ne 0 ]; then
        print_warning "Optional files missing (system will still work):"
        for file in "${optional_files[@]}"; do
            echo "  - $file"
        done
    fi
    
    print_status "All critical files found"
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
    print_info "Testing API Gateway health..."
    
    local health_response=$(curl -s "http://127.0.0.1:$API_GATEWAY_PORT/api/cards/health" 2>/dev/null || echo "failed")
    
    if [ "$health_response" = "failed" ] || [ -z "$health_response" ]; then
        print_warning "API Gateway health check couldn't connect - but API may still be starting"
        print_info "You can manually check: http://127.0.0.1:$API_GATEWAY_PORT/api/health"
        return 0  # Don't fail the startup for this
    elif echo "$health_response" | grep -q '"status":"healthy"'; then
        print_status "API Gateway health check passed"
        
        # Extract service status
        if echo "$health_response" | grep -q '"services"'; then
            print_status "Microservices architecture active"
        fi
    else
        print_warning "API Gateway returned unexpected response, but may be working"
        print_info "Response: $health_response"
    fi
    
    # Test Admin Backend health
    print_info "Testing Admin Backend health..."
    local admin_health=$(curl -s "http://127.0.0.1:$ADMIN_BACKEND_PORT/api/health" 2>/dev/null || echo "failed")
    
    if echo "$admin_health" | grep -q '"status":"healthy"'; then
        print_status "Admin Backend health check passed"
    else
        print_warning "Admin Backend may not be fully ready yet"
    fi
}

# Main startup sequence
main() {
    echo "🧹 Phase 1: Cleanup existing services"
    echo "------------------------------------"
    
    # Kill existing services
    kill_by_name "$API_GATEWAY_SCRIPT" "API Gateway"
    kill_by_name "$ADMIN_BACKEND_SCRIPT" "Admin Backend"
    kill_by_name "http.server" "HTTP Server"
    kill_by_name "uvicorn" "FastAPI Server"
    kill_port $API_GATEWAY_PORT "API Gateway"
    kill_port $ADMIN_BACKEND_PORT "Admin Backend"
    kill_port $HTTP_PORT "HTTP Server"
    
    # Set environment variables
    export WICKETWISE_ENV="$ENVIRONMENT"
    export WICKETWISE_JWT_SECRET="dev-secret-key-$(date +%s)-$(openssl rand -hex 16 2>/dev/null || echo 'fallback-secret')"
    
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
    
    # Start API Gateway (FastAPI with modern architecture)
    print_info "Starting API Gateway on port $API_GATEWAY_PORT..."
    python "$API_GATEWAY_SCRIPT" > api_gateway.log 2>&1 &
    API_GATEWAY_PID=$!
    
    # Start Admin Backend server (for match enrichment and admin operations)
    print_info "Starting Admin Backend server on port $ADMIN_BACKEND_PORT..."
    python "$ADMIN_BACKEND_SCRIPT" > admin_backend.log 2>&1 &
    ADMIN_BACKEND_PID=$!
    
    # Wait for API Gateway
    if wait_for_service "http://127.0.0.1:$API_GATEWAY_PORT/api/cards/health" "API Gateway"; then
        print_status "API Gateway started (PID: $API_GATEWAY_PID)"
    else
        print_error "Failed to start API Gateway"
        print_info "Check api_gateway.log for details"
        exit 1
    fi
    
    # Wait for Admin Backend server
    if wait_for_service "http://127.0.0.1:$ADMIN_BACKEND_PORT/api/health" "Admin Backend"; then
        print_status "Admin Backend started (PID: $ADMIN_BACKEND_PID)"
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
    echo "  • Admin Panel:       http://127.0.0.1:$HTTP_PORT/wicketwise_admin_redesigned.html"
    echo "  • Standalone Cards:  http://127.0.0.1:$HTTP_PORT/enhanced_player_cards_ui.html"
    echo "  • API Gateway:       http://127.0.0.1:$API_GATEWAY_PORT/api/health"
    echo "  • Admin Backend:     http://127.0.0.1:$ADMIN_BACKEND_PORT/api/health"
    echo "  • API Documentation: http://127.0.0.1:$API_GATEWAY_PORT/docs"
    echo ""
    echo "🔧 Service Info:"
    echo "  • HTTP Server PID:     $HTTP_PID"
    echo "  • API Gateway PID:     $API_GATEWAY_PID"
    echo "  • Admin Backend PID:   $ADMIN_BACKEND_PID"
    echo "  • API Gateway Logs:    api_gateway.log"
    echo "  • Admin Backend Logs:  admin_backend.log"
    echo "  • Environment:         $ENVIRONMENT"
    echo ""
    echo "🎯 Features Available:"
    echo "  • Modern FastAPI Gateway with authentication"
    echo "  • Security Framework (JWT, Rate Limiting, Input Validation)"
    echo "  • Real Knowledge Graph data (11,997+ nodes)"
    echo "  • Comprehensive player statistics"
    echo "  • Format-specific analytics"
    echo "  • Persona-based analysis"
    echo "  • Dynamic player cards"
    echo "  • OpenAI Match Enrichment (weather, teams, venues)"
    echo "  • Performance monitoring and benchmarks"
    echo "  • Microservices-ready architecture"
    echo ""
    echo "🛑 To stop services:"
    echo "  kill $HTTP_PID $API_GATEWAY_PID $ADMIN_BACKEND_PID"
    echo "  # Or use: ./stop.sh"
    echo ""
    
    # Save PIDs for stop script
    echo "$HTTP_PID" > .http_server.pid
    echo "$API_GATEWAY_PID" > .api_gateway.pid
    echo "$ADMIN_BACKEND_PID" > .admin_backend.pid
    
    print_status "Startup complete! 🎉"
}

# Trap to cleanup on script exit
cleanup() {
    print_warning "Shutting down services..."
    
    if [ ! -z "$HTTP_PID" ] && kill -0 $HTTP_PID 2>/dev/null; then
        print_warning "Cleaning up HTTP server..."
        kill $HTTP_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$API_GATEWAY_PID" ] && kill -0 $API_GATEWAY_PID 2>/dev/null; then
        print_warning "Cleaning up API Gateway..."
        kill $API_GATEWAY_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$ADMIN_BACKEND_PID" ] && kill -0 $ADMIN_BACKEND_PID 2>/dev/null; then
        print_warning "Cleaning up Admin Backend..."
        kill $ADMIN_BACKEND_PID 2>/dev/null || true
    fi
    
    # Clean up PID files
    rm -f .http_server.pid .api_gateway.pid .admin_backend.pid
    
    print_status "All services stopped"
}

trap cleanup EXIT INT TERM

# Run main function
main

# Keep script running to maintain services
print_info "Services are running. Press Ctrl+C to stop all services."
wait