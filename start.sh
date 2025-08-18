#!/usr/bin/env bash
# Purpose: Start WicketWise backend (Flask) and serve Figma UI
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Activate venv
if [ -d .venv ]; then
  source .venv/bin/activate
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip -q install --upgrade pip
  pip -q install -r requirements.txt
fi

# Start backend (port 5001)
echo "Starting backend on :5001..."
pkill -f "admin_backend.py" >/dev/null 2>&1 || true
nohup bash -lc 'FLASK_ENV=production PYTHONUNBUFFERED=1 PORT=5001 python admin_backend.py' \
  > backend.log 2>&1 &

# Start static server for UI (port 8000)
echo "Starting static UI on :8000..."
pkill -f "python3 -m http.server 8000" >/dev/null 2>&1 || true
nohup python3 -m http.server 8000 > static.log 2>&1 &

sleep 2

echo "Backend health: http://127.0.0.1:5001/api/health"
echo "UI (Figma):     http://127.0.0.1:8000/wicketwise_dashboard.html"

exit 0

#!/bin/bash
echo "Starting WicketWise..."
# future: add commands to run the application 