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

if [ -f ".api_server.pid" ]; then
    API_PID=$(cat .api_server.pid)
    if kill -0 $API_PID 2>/dev/null; then
        print_info "Stopping API server (PID: $API_PID)..."
        kill $API_PID
        print_status "API server stopped"
    else
        print_warning "API server process not found"
    fi
    rm -f .api_server.pid
fi

# Fallback: kill by process name
print_info "Cleaning up any remaining processes..."
pkill -f "real_dynamic_cards_api" 2>/dev/null || true
pkill -f "http.server 8000" 2>/dev/null || true

# Kill by port as final cleanup
lsof -ti:5005 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

print_status "All WicketWise services stopped"
echo ""
print_info "To restart, run: ./start.sh"
