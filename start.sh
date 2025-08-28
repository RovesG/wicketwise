#!/bin/bash

# WicketWise Complete System Startup Script
# Author: WicketWise AI
# Last Modified: January 2025

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DGL_PORT=${DGL_PORT:-8001}
ENHANCED_API_PORT=${ENHANCED_API_PORT:-5001}
STATIC_SERVER_PORT=${STATIC_SERVER_PORT:-8000}
AGENT_UI_PORT=${AGENT_UI_PORT:-3001}
ENVIRONMENT=${ENVIRONMENT:-development}

echo -e "${BLUE}🏏 WicketWise Complete System Startup${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

# Function to start static file server
start_static_server() {
    print_info "Starting Static File Server on port $STATIC_SERVER_PORT..."
    
    if ! check_port $STATIC_SERVER_PORT; then
        print_status "Static server already running on port $STATIC_SERVER_PORT"
        return 0
    fi
    
    # Start Python HTTP server in background
    print_info "Launching static file server..."
    nohup python3 -m http.server $STATIC_SERVER_PORT > logs/static_server.log 2>&1 &
    STATIC_PID=$!
    echo $STATIC_PID > pids/static_server.pid
    
    # Wait for service to be ready
    if wait_for_service "http://localhost:$STATIC_SERVER_PORT" "Static File Server"; then
        print_status "Static File Server started successfully (PID: $STATIC_PID)"
        return 0
    else
        return 1
    fi
}

# Function to start DGL service
start_dgl_service() {
    print_info "Starting DGL Service on port $DGL_PORT..."
    
    if ! check_port $DGL_PORT; then
        print_warning "Port $DGL_PORT is already in use, attempting to free it..."
        lsof -ti:$DGL_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    cd services/dgl
    
    # Check if virtual environment exists
    if [ ! -d "../../.venv" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv ../../.venv
    fi
    
    # Activate virtual environment
    source ../../.venv/bin/activate
    
    # Install dependencies
    print_info "Installing DGL dependencies..."
    pip install -q fastapi uvicorn pydantic pydantic-settings httpx asyncio psutil
    pip install -q hypothesis freezegun pytest
    
    # Start DGL service in background
    print_info "Launching DGL FastAPI service..."
    nohup uvicorn app:app --host 0.0.0.0 --port $DGL_PORT --reload > ../../logs/dgl.log 2>&1 &
    DGL_PID=$!
    echo $DGL_PID > ../../pids/dgl.pid
    
    cd ../..
    
    # Wait for service to be ready
    if wait_for_service "http://localhost:$DGL_PORT/healthz" "DGL Service"; then
        print_status "DGL Service started successfully (PID: $DGL_PID)"
        return 0
    else
        return 1
    fi
}

# Function to start Player Cards API
start_player_cards_api() {
    print_info "Starting Player Cards API on port 5004..."
    
    if ! check_port 5004; then
        print_warning "Port 5004 is already in use, attempting to free it..."
        lsof -ti:5004 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Start Player Cards API in background
    print_info "Launching Player Cards API..."
    nohup python real_dynamic_cards_api.py > logs/player_cards_api.log 2>&1 &
    PLAYER_CARDS_PID=$!
    echo $PLAYER_CARDS_PID > pids/player_cards_api.pid
    
    # Wait for service to be ready
    if wait_for_service "http://localhost:5004/api/match-context" "Player Cards API"; then
        print_status "Player Cards API started successfully (PID: $PLAYER_CARDS_PID)"
        return 0
    else
        return 1
    fi
}

# Function to start Enhanced Dashboard API
start_enhanced_api() {
    print_info "Starting Enhanced Dashboard API on port $ENHANCED_API_PORT..."
    
    if ! check_port $ENHANCED_API_PORT; then
        print_warning "Port $ENHANCED_API_PORT is already in use, attempting to free it..."
        lsof -ti:$ENHANCED_API_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install dependencies
    print_info "Installing Enhanced API dependencies..."
    pip install -q flask flask-cors flask-socketio requests pandas numpy
    pip install -q networkx scikit-learn plotly
    
    # Start Admin Backend service in background
    print_info "Launching Admin Backend API with Agent UI support..."
    nohup python admin_backend.py > logs/admin_backend.log 2>&1 &
    ENHANCED_API_PID=$!
    echo $ENHANCED_API_PID > pids/enhanced_api.pid
    
    # Wait for service to be ready
    if wait_for_service "http://localhost:$ENHANCED_API_PORT/api/health" "Admin Backend API"; then
        print_status "Admin Backend API started successfully (PID: $ENHANCED_API_PID)"
        return 0
    else
        return 1
    fi
}

# Function to check Agent UI availability
check_agent_ui() {
    print_info "Checking Agent UI availability..."
    
    # Agent UI is now a static HTML file served by the static server
    if [ -f "wicketwise_agent_ui.html" ]; then
        print_status "Agent UI (HTML) is available at http://localhost:$STATIC_SERVER_PORT/wicketwise_agent_ui.html"
        return 0
    else
        print_warning "Agent UI HTML file not found"
        return 1
    fi
}

# Function to show system status
show_system_status() {
    echo ""
    print_info "WicketWise Complete System Status:"
    echo "=================================="
    
    # Check Static File Server
    if curl -s "http://localhost:$STATIC_SERVER_PORT" >/dev/null 2>&1; then
        print_status "Static File Server: Running on http://localhost:$STATIC_SERVER_PORT"
    else
        print_error "Static File Server: Not running"
    fi
    
    # Check DGL Service
    if curl -s "http://localhost:$DGL_PORT/healthz" >/dev/null 2>&1; then
        print_status "DGL Service: Running on http://localhost:$DGL_PORT"
    else
        print_error "DGL Service: Not running"
    fi
    
    # Check Enhanced Dashboard API
    if curl -s "http://localhost:$ENHANCED_API_PORT/api/health" >/dev/null 2>&1; then
        print_status "Admin Backend API: Running on http://localhost:$ENHANCED_API_PORT"
    else
        print_error "Admin Backend API: Not running"
    fi
    
    # Check Agent UI (HTML file)
    if [ -f "wicketwise_agent_ui.html" ]; then
        print_status "Agent UI: Available at http://localhost:$STATIC_SERVER_PORT/wicketwise_agent_ui.html"
    else
        print_error "Agent UI: HTML file not found"
    fi
    
    # Check Player Cards API
    if curl -s "http://localhost:5004/api/match-context" >/dev/null 2>&1; then
        print_status "Player Cards API: Running on http://localhost:5004"
    else
        print_error "Player Cards API: Not running"
    fi
    
    echo ""
    print_info "🎯 Main URLs:"
    echo "  🏏 Main Dashboard: http://localhost:$STATIC_SERVER_PORT/wicketwise_dashboard.html"
    echo "  🤖 Agent UI (NEW): http://localhost:$STATIC_SERVER_PORT/wicketwise_agent_ui.html"
    echo "  ⚙️  Legacy Admin Panel: http://localhost:$STATIC_SERVER_PORT/wicketwise_admin_redesigned.html"
    echo "  🛡️  DGL API: http://localhost:$DGL_PORT"
    echo "  📊 Admin Backend API: http://localhost:$ENHANCED_API_PORT"
    
    echo ""
    print_info "🔗 Key API Endpoints:"
    echo "  📊 Holdout Matches: GET http://localhost:$ENHANCED_API_PORT/api/simulation/holdout-matches"
    echo "  🎯 Run Simulation: POST http://localhost:$ENHANCED_API_PORT/api/simulation/run"
    echo "  🏏 Player Search: GET http://localhost:$ENHANCED_API_PORT/api/enhanced/search-players"
    echo "  🎴 Enhanced Player Cards: POST http://localhost:5004/api/cards/enhanced"
    echo "  🤖 Agent UI WebSocket: ws://localhost:$ENHANCED_API_PORT/agent_ui"
    echo ""
    
    print_info "🚀 Agent UI Features:"
    echo "  • System Map - Real-time agent visualization"
    echo "  • Flowline Explorer - Timeline-based event analysis"
    echo "  • Advanced Debug Tools - Breakpoints, watch expressions, performance analytics"
    echo "  • Cricket Intelligence - Match-aware betting decision explainability"
    echo ""
}

# Function to stop all services
stop_services() {
    print_info "Stopping WicketWise services..."
    
    # Stop services using PID files
    local pid_files=("pids/dgl.pid" "pids/enhanced_api.pid" "pids/static_server.pid" "pids/player_cards_api.pid" "pids/agent_ui.pid")
    
    for pid_file in "${pid_files[@]}"; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                print_info "Stopping process $pid..."
                kill "$pid"
                rm "$pid_file"
            fi
        fi
    done
    
    # Also kill by port
    for port in $DGL_PORT $ENHANCED_API_PORT $STATIC_SERVER_PORT $AGENT_UI_PORT 5004; do
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
    done
    
    print_status "All services stopped"
}

# Function to setup directories
setup_directories() {
    print_info "Setting up directories..."
    
    # Create necessary directories
    mkdir -p logs pids
    
    # Create log files
    touch logs/dgl.log logs/enhanced_api.log logs/static_server.log logs/agent_ui.log logs/agent_ui_install.log
    
    print_status "Directories setup complete"
}

# Function to run quick system test
run_quick_test() {
    print_info "Running quick system test..."
    
    # Test Static Server
    if curl -s "http://localhost:$STATIC_SERVER_PORT" >/dev/null 2>&1; then
        print_status "Static File Server: ✓"
    else
        print_error "Static File Server: ✗"
    fi
    
    # Test DGL health
    if curl -s "http://localhost:$DGL_PORT/healthz" >/dev/null 2>&1; then
        print_status "DGL Service: ✓"
    else
        print_error "DGL Service: ✗"
    fi
    
    # Test Enhanced API health
    if curl -s "http://localhost:$ENHANCED_API_PORT/api/enhanced/health" >/dev/null 2>&1; then
        print_status "Enhanced API: ✓"
    else
        print_error "Enhanced API: ✗"
    fi
    
    # Test simulation endpoints
    print_info "Testing simulation endpoints..."
    if curl -s "http://localhost:$ENHANCED_API_PORT/api/simulation/holdout-matches" >/dev/null 2>&1; then
        print_status "Simulation API: ✓"
    else
        print_error "Simulation API: ✗"
    fi
}

# Function to open browser
open_browser() {
    local url="http://localhost:$STATIC_SERVER_PORT/wicketwise_dashboard.html"
    
    if command -v open >/dev/null 2>&1; then
        # macOS
        open "$url"
    elif command -v xdg-open >/dev/null 2>&1; then
        # Linux
        xdg-open "$url"
    elif command -v start >/dev/null 2>&1; then
        # Windows
        start "$url"
    else
        print_info "Please open $url in your browser"
    fi
}

# Main execution
main() {
    local command=${1:-"start"}
    
    case $command in
        "start")
            setup_directories
            
            print_info "Starting WicketWise Complete System..."
            echo ""
            
            # Start static file server
            if start_static_server; then
                print_status "Static File Server started successfully"
            else
                print_warning "Static File Server failed to start (continuing...)"
            fi
            
            # Start DGL service
            if start_dgl_service; then
                print_status "DGL Service started successfully"
            else
                print_error "Failed to start DGL Service"
                exit 1
            fi
            
            # Start Enhanced Dashboard API
            if start_enhanced_api; then
                print_status "Enhanced Dashboard API started successfully"
            else
                print_warning "Enhanced Dashboard API failed to start (continuing...)"
            fi
            
            # Start Player Cards API
            if start_player_cards_api; then
                print_status "Player Cards API started successfully"
            else
                print_warning "Player Cards API failed to start (continuing...)"
            fi
            
            # Check Agent UI availability
            if check_agent_ui; then
                print_status "Agent UI is available"
            else
                print_warning "Agent UI HTML file not found (continuing...)"
            fi
            
            show_system_status
            
            print_status "WicketWise Complete System startup complete! 🚀"
            print_info "Opening Agent UI in browser..."
            
            # Open Agent UI (HTML file)
            local agent_ui_url="http://localhost:$STATIC_SERVER_PORT/wicketwise_agent_ui.html"
            
            if command -v open >/dev/null 2>&1; then
                # macOS
                open "$agent_ui_url"
            elif command -v xdg-open >/dev/null 2>&1; then
                # Linux
                xdg-open "$agent_ui_url"
            elif command -v start >/dev/null 2>&1; then
                # Windows
                start "$agent_ui_url"
            else
                print_info "Please open $agent_ui_url in your browser"
            fi
            ;;
            
        "stop")
            stop_services
            ;;
            
        "restart")
            stop_services
            sleep 3
            main "start"
            ;;
            
        "status")
            show_system_status
            ;;
            
        "test")
            run_quick_test
            ;;
            
        "logs")
            local service=${2:-"dgl"}
            if [ -f "logs/$service.log" ]; then
                tail -f "logs/$service.log"
            else
                print_error "Log file not found: logs/$service.log"
                print_info "Available logs: $(ls logs/ 2>/dev/null || echo 'none')"
            fi
            ;;
            
        "help"|"-h"|"--help")
            echo "WicketWise Complete System Control Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  start     Start all WicketWise services (default)"
            echo "  stop      Stop all WicketWise services"
            echo "  restart   Restart all WicketWise services"
            echo "  status    Show system status"
            echo "  test      Run quick system test"
            echo "  logs      Show logs (specify service: dgl, enhanced_api, static_server, agent_ui)"
            echo "  help      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  DGL_PORT=$DGL_PORT"
            echo "  ENHANCED_API_PORT=$ENHANCED_API_PORT"
            echo "  STATIC_SERVER_PORT=$STATIC_SERVER_PORT"
            echo "  AGENT_UI_PORT=$AGENT_UI_PORT"
            echo "  ENVIRONMENT=$ENVIRONMENT"
            ;;
            
        *)
            print_error "Unknown command: $command"
            print_info "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'print_info "Received interrupt signal, stopping services..."; stop_services; exit 0' INT

# Run main function
main "$@"