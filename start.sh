#!/bin/bash

# WicketWise DGL System Startup Script
# Author: WicketWise AI
# Last Modified: December 2024

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DGL_PORT=${DGL_PORT:-8001}
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
BACKEND_PORT=${BACKEND_PORT:-5001}
ENVIRONMENT=${ENVIRONMENT:-development}

echo -e "${BLUE}🏏 WicketWise DGL System Startup${NC}"
echo -e "${BLUE}=================================${NC}"
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

# Function to start DGL service
start_dgl_service() {
    print_info "Starting DGL Service on port $DGL_PORT..."
    
    if ! check_port $DGL_PORT; then
        print_warning "Port $DGL_PORT is already in use"
        return 1
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
    pip install -q streamlit plotly pandas numpy
    pip install -q hypothesis freezegun pytest
    
    # Start DGL service in background
    print_info "Launching DGL FastAPI service..."
    nohup uvicorn app:app --host 0.0.0.0 --port $DGL_PORT --reload > ../../logs/dgl.log 2>&1 &
    DGL_PID=$!
    echo $DGL_PID > ../../pids/dgl.pid
    
    # Wait for service to be ready
    if wait_for_service "http://localhost:$DGL_PORT/healthz" "DGL Service"; then
        print_status "DGL Service started successfully (PID: $DGL_PID)"
        return 0
    else
        return 1
    fi
}

# Function to start Streamlit dashboard
start_streamlit_dashboard() {
    print_info "Starting Streamlit Dashboard on port $STREAMLIT_PORT..."
    
    if ! check_port $STREAMLIT_PORT; then
        print_warning "Port $STREAMLIT_PORT is already in use"
        return 1
    fi
    
    cd services/dgl/ui
    
    # Start Streamlit in background
    print_info "Launching Streamlit dashboard..."
    nohup streamlit run streamlit_app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0 > ../../../logs/streamlit.log 2>&1 &
    STREAMLIT_PID=$!
    echo $STREAMLIT_PID > ../../../pids/streamlit.pid
    
    # Wait for service to be ready
    if wait_for_service "http://localhost:$STREAMLIT_PORT" "Streamlit Dashboard"; then
        print_status "Streamlit Dashboard started successfully (PID: $STREAMLIT_PID)"
        return 0
    else
        return 1
    fi
}

# Function to start backend service
start_backend_service() {
    print_info "Starting Backend Service on port $BACKEND_PORT..."
    
    if ! check_port $BACKEND_PORT; then
        print_warning "Port $BACKEND_PORT is already in use"
        return 1
    fi
    
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "../.venv" ]; then
        print_info "Creating Python virtual environment for backend..."
        python3 -m venv ../.venv
    fi
    
    # Activate virtual environment
    source ../.venv/bin/activate
    
    # Install dependencies
    print_info "Installing backend dependencies..."
    pip install -q -r requirements.txt
    
    # Start backend service
    print_info "Launching backend service..."
    nohup python app.py > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../pids/backend.pid
    
    # Wait for service to be ready
    if wait_for_service "http://localhost:$BACKEND_PORT/health" "Backend Service"; then
        print_status "Backend Service started successfully (PID: $BACKEND_PID)"
        return 0
    else
        return 1
    fi
}

# Function to start frontend service
start_frontend_service() {
    print_info "Starting Frontend Service on port $FRONTEND_PORT..."
    
    if ! check_port $FRONTEND_PORT; then
        print_warning "Port $FRONTEND_PORT is already in use"
        return 1
    fi
    
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        print_info "Installing frontend dependencies..."
        npm install
    fi
    
    # Start frontend service
    print_info "Launching frontend service..."
    nohup npm start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../pids/frontend.pid
    
    # Wait for service to be ready
    if wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend Service"; then
        print_status "Frontend Service started successfully (PID: $FRONTEND_PID)"
        return 0
    else
        return 1
    fi
}

# Function to run system tests
run_system_tests() {
    print_info "Running WicketWise DGL System Tests..."
    
    cd services/dgl
    source ../../.venv/bin/activate
    
    # Run all sprint tests
    local test_files=(
        "tests/test_sprint_g0.py"
        "tests/test_sprint_g1.py" 
        "tests/test_sprint_g2.py"
        "tests/test_sprint_g3.py"
        "tests/test_sprint_g4.py"
        "tests/test_sprint_g5.py"
        "tests/test_sprint_g6.py"
        "tests/test_sprint_g7.py"
        "tests/test_sprint_g8.py"
        "tests/test_sprint_g9.py"
    )
    
    local passed_tests=0
    local total_tests=${#test_files[@]}
    
    for test_file in "${test_files[@]}"; do
        if [ -f "$test_file" ]; then
            print_info "Running $(basename $test_file)..."
            if python "$test_file" > /dev/null 2>&1; then
                print_status "$(basename $test_file) - PASSED"
                ((passed_tests++))
            else
                print_error "$(basename $test_file) - FAILED"
            fi
        else
            print_warning "Test file not found: $test_file"
        fi
    done
    
    echo ""
    print_info "Test Results: $passed_tests/$total_tests tests passed"
    
    if [ $passed_tests -eq $total_tests ]; then
        print_status "All system tests passed! ✨"
        return 0
    else
        print_warning "Some tests failed. Check logs for details."
        return 1
    fi
}

# Function to show system status
show_system_status() {
    echo ""
    print_info "WicketWise DGL System Status:"
    echo "================================"
    
    # Check DGL Service
    if curl -s "http://localhost:$DGL_PORT/healthz" >/dev/null 2>&1; then
        print_status "DGL Service: Running on http://localhost:$DGL_PORT"
    else
        print_error "DGL Service: Not running"
    fi
    
    # Check Streamlit Dashboard
    if curl -s "http://localhost:$STREAMLIT_PORT" >/dev/null 2>&1; then
        print_status "Streamlit Dashboard: Running on http://localhost:$STREAMLIT_PORT"
    else
        print_error "Streamlit Dashboard: Not running"
    fi
    
    # Check Backend Service
    if curl -s "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1; then
        print_status "Backend Service: Running on http://localhost:$BACKEND_PORT"
    else
        print_error "Backend Service: Not running"
    fi
    
    # Check Frontend Service
    if curl -s "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1; then
        print_status "Frontend Service: Running on http://localhost:$FRONTEND_PORT"
    else
        print_error "Frontend Service: Not running"
    fi
    
    echo ""
    print_info "Key URLs:"
    echo "  🛡️  DGL API: http://localhost:$DGL_PORT"
    echo "  📊 DGL Dashboard: http://localhost:$STREAMLIT_PORT"
    echo "  🏏 WicketWise Dashboard: http://localhost:$FRONTEND_PORT"
    echo "  ⚙️  Backend API: http://localhost:$BACKEND_PORT"
    echo ""
}

# Function to stop all services
stop_services() {
    print_info "Stopping WicketWise services..."
    
    # Stop services using PID files
    local pid_files=("pids/dgl.pid" "pids/streamlit.pid" "pids/backend.pid" "pids/frontend.pid")
    
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

# Function to setup directories
setup_directories() {
    print_info "Setting up directories..."
    
    # Create necessary directories
    mkdir -p logs pids
    
    # Create log files
    touch logs/dgl.log logs/streamlit.log logs/backend.log logs/frontend.log
    
    print_status "Directories setup complete"
}

# Main execution
main() {
    local command=${1:-"start"}
    
    case $command in
        "start")
            setup_directories
            
            print_info "Starting WicketWise DGL System..."
            echo ""
            
            # Start core DGL service
            if start_dgl_service; then
                print_status "DGL Service started successfully"
            else
                print_error "Failed to start DGL Service"
                exit 1
            fi
            
            # Start Streamlit dashboard
            if start_streamlit_dashboard; then
                print_status "Streamlit Dashboard started successfully"
            else
                print_warning "Streamlit Dashboard failed to start (continuing...)"
            fi
            
            # Start backend service (if exists)
            if [ -d "backend" ]; then
                if start_backend_service; then
                    print_status "Backend Service started successfully"
                else
                    print_warning "Backend Service failed to start (continuing...)"
                fi
            fi
            
            # Start frontend service (if exists)
            if [ -d "frontend" ]; then
                if start_frontend_service; then
                    print_status "Frontend Service started successfully"
                else
                    print_warning "Frontend Service failed to start (continuing...)"
                fi
            fi
            
            show_system_status
            
            print_status "WicketWise DGL System startup complete! 🚀"
            ;;
            
        "stop")
            stop_services
            ;;
            
        "restart")
            stop_services
            sleep 2
            main "start"
            ;;
            
        "status")
            show_system_status
            ;;
            
        "test")
            run_system_tests
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
            echo "WicketWise DGL System Control Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  start     Start all WicketWise services (default)"
            echo "  stop      Stop all WicketWise services"
            echo "  restart   Restart all WicketWise services"
            echo "  status    Show system status"
            echo "  test      Run system tests"
            echo "  logs      Show logs (specify service: dgl, streamlit, backend, frontend)"
            echo "  help      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  DGL_PORT=$DGL_PORT"
            echo "  STREAMLIT_PORT=$STREAMLIT_PORT"
            echo "  FRONTEND_PORT=$FRONTEND_PORT"
            echo "  BACKEND_PORT=$BACKEND_PORT"
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