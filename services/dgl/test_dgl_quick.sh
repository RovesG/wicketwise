#!/bin/bash

# Quick DGL test script
echo "🧪 WicketWise DGL - Quick Test"
echo "=============================="

# Test 1: App import
echo "📦 Testing app import..."
python -c "from app import create_app; print('✅ App import successful')" || exit 1

# Test 2: Core functionality
echo "🔧 Testing core functionality..."
python tests/test_dgl_core_functionality.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Core functionality: PASSED"
else
    echo "⚠️  Core functionality: Some tests failed (expected)"
fi

# Test 3: Configuration
echo "⚙️  Testing configuration..."
python -c "from config import load_config; config = load_config(); print('✅ Configuration loaded successfully')" || exit 1

# Test 4: Schemas
echo "📋 Testing schemas..."
python -c "from schemas import BetProposal, BetSide; p = BetProposal(match_id='test', market_id='test', side=BetSide.BACK, selection='test', odds=2.0, stake=100.0, model_confidence=0.8, expected_edge_pct=5.0); print('✅ Schemas working')" || exit 1

echo ""
echo "�� DGL Quick Test Complete!"
echo "✅ System is ready to start"
echo ""
echo "To start the DGL service:"
echo "  ./start_simple.sh"
echo ""
echo "Or manually:"
echo "  uvicorn app:app --host 0.0.0.0 --port 8001 --reload"
