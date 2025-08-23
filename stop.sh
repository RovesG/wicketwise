#!/bin/bash

# WicketWise Enhanced Cricket Dashboard Stop Script
# Purpose: Clean shutdown of all services
# Author: Assistant, Last Modified: 2025-01-19

echo "🛑 WicketWise Enhanced Cricket Dashboard Shutdown"
echo "==============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✅]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠️ ]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ️ ]${NC} $1"
}

# Kill services by PID files
if [ -f ".http_server.pid" ]; then
    HTTP_PID=$(cat .http_server.pid)
    if kill -0 $HTTP_PID 2>/dev/null; then
        print_info "Stopping HTTP server (PID: $HTTP_PID)..."
        kill $HTTP_PID
        print_status "HTTP server stopped"
    else
        print_warning "HTTP server process not found"
    fi
    rm -f .http_server.pid
fi

if [ -f ".api_gateway.pid" ]; then
    API_GATEWAY_PID=$(cat .api_gateway.pid)
    if kill -0 $API_GATEWAY_PID 2>/dev/null; then
        print_info "Stopping API Gateway (PID: $API_GATEWAY_PID)..."
        kill $API_GATEWAY_PID
        print_status "API Gateway stopped"
    else
        print_warning "API Gateway process not found"
    fi
    rm -f .api_gateway.pid
fi

if [ -f ".admin_backend.pid" ]; then
    ADMIN_BACKEND_PID=$(cat .admin_backend.pid)
    if kill -0 $ADMIN_BACKEND_PID 2>/dev/null; then
        print_info "Stopping Admin Backend (PID: $ADMIN_BACKEND_PID)..."
        kill $ADMIN_BACKEND_PID
        print_status "Admin Backend stopped"
    else
        print_warning "Admin Backend process not found"
    fi
    rm -f .admin_backend.pid
fi

# Fallback: kill by process name
print_info "Cleaning up any remaining processes..."
pkill -f "modern_api_gateway.py" 2>/dev/null || true
pkill -f "admin_backend.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "http.server 8000" 2>/dev/null || true

# Kill by port as final cleanup
print_info "Final port cleanup..."
lsof -ti:5005 | xargs kill -9 2>/dev/null || true
lsof -ti:5001 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

print_status "All WicketWise services stopped"
echo ""
print_info "To restart, run: ./start.sh"
