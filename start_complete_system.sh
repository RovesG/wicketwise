#!/bin/bash

# WicketWise Complete System Startup Script
# Author: WicketWise AI
# Last Modified: December 2024

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    local max_attempts=15
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
    
    print_warning "$service_name took longer than expected (continuing...)"
    return 1
}

# Setup directories
setup_directories() {
    print_info "Setting up directories..."
    mkdir -p logs pids
    print_status "Directories setup complete"
}

# Start Static Server for Main UI
start_static_server() {
    print_info "Starting Static Server for Main Betting UI (Port 8000)..."
    
    if ! check_port 8000; then
        print_warning "Port 8000 is already in use"
        return 0
    fi
    
    # Start static server in background
    nohup python -m http.server 8000 > logs/static.log 2>&1 &
    STATIC_PID=$!
    echo $STATIC_PID > pids/static.pid
    
    if wait_for_service "http://localhost:8000" "Static Server"; then
        print_status "Static Server started successfully (PID: $STATIC_PID)"
        return 0
    else
        return 1
    fi
}

# Start Admin Backend
start_admin_backend() {
    print_info "Starting Admin Backend with Cricsheet Integration (Port 5001)..."
    
    if ! check_port 5001; then
        print_warning "Port 5001 is already in use"
        return 0
    fi
    
    # Activate virtual environment
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Start admin backend in background
    nohup python admin_backend.py > logs/admin_backend.log 2>&1 &
    ADMIN_PID=$!
    echo $ADMIN_PID > pids/admin_backend.pid
    
    if wait_for_service "http://localhost:5001/api/health" "Admin Backend"; then
        print_status "Admin Backend started successfully (PID: $ADMIN_PID)"
        return 0
    else
        return 1
    fi
}

# Start Enhanced Dashboard API
start_enhanced_api() {
    print_info "Starting Enhanced Dashboard API (Port 5002)..."
    
    if ! check_port 5002; then
        print_warning "Port 5002 is already in use"
        return 0
    fi
    
    # Activate virtual environment
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Start enhanced API in background
    nohup python enhanced_dashboard_api.py > logs/enhanced_api.log 2>&1 &
    ENHANCED_PID=$!
    echo $ENHANCED_PID > pids/enhanced_api.pid
    
    if wait_for_service "http://localhost:5002/api/enhanced/health" "Enhanced API"; then
        print_status "Enhanced Dashboard API started successfully (PID: $ENHANCED_PID)"
        return 0
    else
        return 1
    fi
}

# Start DGL Backend
start_dgl_backend() {
    print_info "Starting DGL Backend (Port 8001)..."
    
    if ! check_port 8001; then
        print_warning "Port 8001 is already in use"
        return 0
    fi
    
    cd services/dgl
    
    # Activate virtual environment
    if [ -f "../../.venv/bin/activate" ]; then
        source ../../.venv/bin/activate
    fi
    
    # Install dependencies
    pip install -q fastapi uvicorn pydantic pydantic-settings
    
    # Start DGL service in background
    nohup uvicorn app:app --host 0.0.0.0 --port 8001 --reload > ../../logs/dgl.log 2>&1 &
    DGL_PID=$!
    echo $DGL_PID > ../../pids/dgl.pid
    
    cd ../..
    
    if wait_for_service "http://localhost:8001/healthz" "DGL Backend"; then
        print_status "DGL Backend started successfully (PID: $DGL_PID)"
        return 0
    else
        return 1
    fi
}

# Start DGL Streamlit UI
start_dgl_ui() {
    print_info "Starting DGL Streamlit UI (Port 8501)..."
    
    if ! check_port 8501; then
        print_warning "Port 8501 is already in use"
        return 0
    fi
    
    cd services/dgl
    
    # Activate virtual environment
    if [ -f "../../.venv/bin/activate" ]; then
        source ../../.venv/bin/activate
    fi
    
    # Install Streamlit dependencies
    pip install -q streamlit plotly pandas numpy
    
    # Start Streamlit in background
    nohup streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > ../../logs/streamlit.log 2>&1 &
    STREAMLIT_PID=$!
    echo $STREAMLIT_PID > ../../pids/streamlit.pid
    
    cd ../..
    
    if wait_for_service "http://localhost:8501" "DGL Streamlit UI"; then
        print_status "DGL Streamlit UI started successfully (PID: $STREAMLIT_PID)"
        return 0
    else
        return 1
    fi
}

# Function to show system status
show_system_status() {
    echo ""
    print_info "🏏 WicketWise Complete System Status:"
    echo "====================================="
    
    # Check Main UI
    if curl -s "http://localhost:8000" >/dev/null 2>&1; then
        print_status "📊 Main Betting UI: http://localhost:8000/wicketwise_dashboard.html"
    else
        print_error "📊 Main Betting UI: Not running"
    fi
    
    # Check Admin Panel
    if curl -s "http://localhost:8000/wicketwise_admin_redesigned.html" >/dev/null 2>&1; then
        print_status "⚙️  Admin Panel: http://localhost:8000/wicketwise_admin_redesigned.html"
    else
        print_error "⚙️  Admin Panel: Not accessible"
    fi
    
    # Check Admin Backend
    if curl -s "http://localhost:5001/api/health" >/dev/null 2>&1; then
        print_status "🔧 Admin Backend: http://localhost:5001 (Cricsheet Integration)"
    else
        print_error "🔧 Admin Backend: Not running"
    fi
    
    # Check Enhanced API
    if curl -s "http://localhost:5002/api/enhanced/health" >/dev/null 2>&1; then
        print_status "🧠 Enhanced API: http://localhost:5002 (Cricket Intelligence)"
    else
        print_error "🧠 Enhanced API: Not running"
    fi
    
    # Check DGL Backend
    if curl -s "http://localhost:8001/healthz" >/dev/null 2>&1; then
        print_status "🛡️  DGL Backend: http://localhost:8001/docs (Risk Management)"
    else
        print_error "🛡️  DGL Backend: Not running"
    fi
    
    # Check DGL UI
    if curl -s "http://localhost:8501" >/dev/null 2>&1; then
        print_status "🎛️  DGL UI: http://localhost:8501 (Governance Dashboard)"
    else
        print_error "🎛️  DGL UI: Not running"
    fi
    
    echo ""
    print_info "🎯 Your Complete WicketWise Architecture:"
    echo ""
    echo "  📊 MAIN BETTING UI DASHBOARD:"
    echo "     http://localhost:8000/wicketwise_dashboard.html"
    echo "     • Live betting intelligence with KG queries"
    echo "     • Player cards with GNN insights (17K+ players)"
    echo "     • Real-time match analysis and predictions"
    echo ""
    echo "  ⚙️  ADMIN PANEL (Cricsheet Integration):"
    echo "     http://localhost:8000/wicketwise_admin_redesigned.html"
    echo "     • Auto-check Cricsheet for new JSON files"
    echo "     • Build/update Knowledge Graph (32K+ nodes)"
    echo "     • Train GNN embeddings and Crickformer models"
    echo ""
    echo "  🛡️  DGL BETTING INFRASTRUCTURE (Deterministic Controls):"
    echo "     • API: http://localhost:8001/docs"
    echo "     • UI: http://localhost:8501"
    echo "     • Sub-millisecond risk decisions (1.08ms avg)"
    echo "     • AI-independent safety controls"
    echo "     • Prevents LLM orchestrator from 'losing the plot'"
    echo ""
}

# Function to stop all services
stop_services() {
    print_info "Stopping all WicketWise services..."
    
    # Stop services using PID files
    local pid_files=("pids/static.pid" "pids/admin_backend.pid" "pids/enhanced_api.pid" "pids/dgl.pid" "pids/streamlit.pid")
    
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
    
    print_status "All services stopped"
}

# Main execution
main() {
    local command=${1:-"start"}
    
    case $command in
        "start")
            setup_directories
            
            print_info "Starting Complete WicketWise System..."
            echo ""
            
            # 1. Start Static Server for Main UI
            start_static_server
            
            # 2. Start Admin Backend (Cricsheet Integration)
            start_admin_backend
            
            # 3. Start Enhanced Dashboard API
            start_enhanced_api
            
            # 4. Start DGL Backend (Risk Management)
            start_dgl_backend
            
            # 5. Start DGL Streamlit UI
            start_dgl_ui
            
            show_system_status
            
            print_status "🚀 WicketWise Complete System startup complete!"
            print_info "🎯 Ready for cricket betting with full AI intelligence and risk controls!"
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
            echo "  help      Show this help message"
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
