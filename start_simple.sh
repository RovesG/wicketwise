#!/bin/bash

# Simple DGL startup script
# Author: WicketWise AI

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🛡️  WicketWise DGL - Simple Startup${NC}"
echo "=================================="

# Configuration
DGL_PORT=${DGL_PORT:-8001}

echo -e "${BLUE}Starting DGL service on port $DGL_PORT...${NC}"

# Check if port is available
if lsof -Pi :$DGL_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Port $DGL_PORT is already in use${NC}"
    echo "To stop existing service: kill \$(lsof -t -i:$DGL_PORT)"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f "../../.venv/bin/activate" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source ../../.venv/bin/activate
fi

# Install basic dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -q fastapi uvicorn pydantic pydantic-settings

# Start the service
echo -e "${BLUE}Starting DGL FastAPI service...${NC}"
echo -e "${GREEN}✅ DGL service starting at: http://localhost:$DGL_PORT${NC}"
echo -e "${GREEN}✅ API documentation: http://localhost:$DGL_PORT/docs${NC}"
echo -e "${GREEN}✅ Health check: http://localhost:$DGL_PORT/healthz${NC}"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Run the service (foreground)
uvicorn app:app --host 0.0.0.0 --port $DGL_PORT --reload