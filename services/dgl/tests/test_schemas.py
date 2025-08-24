# Purpose: Unit tests for DGL Pydantic schemas
# Author: WicketWise AI, Last Modified: 2024

import pytest
from datetime import datetime
from decimal import Decimal
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Add services directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schemas import (
    BetProposal, GovernanceDecision, DecisionType, BetSide, RuleId,
    GovernanceState, BetAmendment, ExposureSnapshot, AuditRecord,
    HealthResponse, VersionResponse, LiquidityInfo, MarketDepth
)


class TestBetProposal:
    """Test suite for BetProposal schema"""
    
    def test_bet_proposal_creation(self):
        """Test creating a valid bet proposal"""
        proposal = BetProposal(
            market_id="betfair:1.234567",
            match_id="MI_vs_CSK_2025-05-12",
            side=BetSide.BACK,
            selection="TeamA_Win",
            odds=1.78,
            stake=500.0,
            currency="GBP",
            model_confidence=0.83,
            fair_odds=1.62,
            expected_edge_pct=9.9,
            explain="Edge from mispricing engine"
        )
        
        assert proposal.market_id == "betfair:1.234567"
        assert proposal.match_id == "MI_vs_CSK_2025-05-12"
        assert proposal.side == BetSide.BACK
        assert proposal.selection == "TeamA_Win"
        assert proposal.odds == 1.78
        assert proposal.stake == 500.0
        assert proposal.currency == "GBP"
        assert proposal.model_confidence == 0.83
        assert proposal.fair_odds == 1.62
        assert proposal.expected_edge_pct == 9.9
        assert proposal.explain == "Edge from mispricing engine"
        
        # Check auto-generated fields
        assert proposal.proposal_id is not None
        assert isinstance(proposal.timestamp, datetime)
    
    def test_bet_proposal_with_liquidity(self):
        """Test bet proposal with liquidity information"""
        liquidity = LiquidityInfo(
            available=25000.0,
            market_depth=[
                MarketDepth(odds=1.78, size=1200.0),
                MarketDepth(odds=1.79, size=800.0)
            ]
        )
        
        proposal = BetProposal(
            market_id="betfair:1.234567",
            match_id="MI_vs_CSK_2025-05-12",
            side=BetSide.LAY,
            selection="TeamB_Win",
            odds=2.5,
            stake=1000.0,
            model_confidence=0.75,
            fair_odds=2.2,
            expected_edge_pct=12.5,
            liquidity=liquidity,
            correlation_group="MI_vs_CSK_match_win"
        )
        
        assert proposal.liquidity.available == 25000.0
        assert len(proposal.liquidity.market_depth) == 2
        assert proposal.liquidity.market_depth[0].odds == 1.78
        assert proposal.liquidity.market_depth[0].size == 1200.0
        assert proposal.correlation_group == "MI_vs_CSK_match_win"
    
    def test_bet_proposal_validation_errors(self):
        """Test bet proposal validation errors"""
        
        # Invalid currency format
        with pytest.raises(ValueError, match="Currency must be a 3-letter uppercase code"):
            BetProposal(
                market_id="test",
                match_id="test",
                side=BetSide.BACK,
                selection="test",
                odds=1.5,
                stake=100.0,
                model_confidence=0.5,
                fair_odds=1.4,
                expected_edge_pct=5.0,
                currency="gb"  # Invalid - not 3 letters uppercase
            )
        
        # Invalid odds range
        with pytest.raises(ValueError):
            BetProposal(
                market_id="test",
                match_id="test",
                side=BetSide.BACK,
                selection="test",
                odds=0.5,  # Invalid - too low
                stake=100.0,
                model_confidence=0.5,
                fair_odds=1.4,
                expected_edge_pct=5.0
            )
        
        # Invalid stake (negative)
        with pytest.raises(ValueError):
            BetProposal(
                market_id="test",
                match_id="test",
                side=BetSide.BACK,
                selection="test",
                odds=1.5,
                stake=-100.0,  # Invalid - negative
                model_confidence=0.5,
                fair_odds=1.4,
                expected_edge_pct=5.0
            )
        
        # Invalid model confidence (out of range)
        with pytest.raises(ValueError):
            BetProposal(
                market_id="test",
                match_id="test",
                side=BetSide.BACK,
                selection="test",
                odds=1.5,
                stake=100.0,
                model_confidence=1.5,  # Invalid - > 1.0
                fair_odds=1.4,
                expected_edge_pct=5.0
            )


class TestGovernanceDecision:
    """Test suite for GovernanceDecision schema"""
    
    def test_approval_decision(self):
        """Test creating an approval decision"""
        decision = GovernanceDecision(
            proposal_id="test-proposal-123",
            decision=DecisionType.APPROVE,
            rule_ids_triggered=[],
            human_message="All governance rules satisfied",
            state=GovernanceState.LIVE,
            audit_ref="audit:2025-08-24:abcd123"
        )
        
        assert decision.proposal_id == "test-proposal-123"
        assert decision.decision == DecisionType.APPROVE
        assert decision.rule_ids_triggered == []
        assert decision.human_message == "All governance rules satisfied"
        assert decision.state == GovernanceState.LIVE
        assert decision.ttl_seconds == 5  # Default value
        assert decision.audit_ref == "audit:2025-08-24:abcd123"
        assert decision.amendment is None
    
    def test_rejection_decision(self):
        """Test creating a rejection decision"""
        decision = GovernanceDecision(
            proposal_id="test-proposal-456",
            decision=DecisionType.REJECT,
            rule_ids_triggered=[RuleId.BANKROLL_MAX_EXPOSURE, RuleId.PNL_DAILY_LOSS_LIMIT],
            human_message="Exposure limits exceeded",
            state=GovernanceState.LIVE,
            audit_ref="audit:2025-08-24:def456",
            ttl_seconds=10
        )
        
        assert decision.decision == DecisionType.REJECT
        assert len(decision.rule_ids_triggered) == 2
        assert RuleId.BANKROLL_MAX_EXPOSURE in decision.rule_ids_triggered
        assert RuleId.PNL_DAILY_LOSS_LIMIT in decision.rule_ids_triggered
        assert decision.ttl_seconds == 10
    
    def test_amendment_decision(self):
        """Test creating an amendment decision"""
        amendment = BetAmendment(stake=350.0, odds=1.76)
        
        decision = GovernanceDecision(
            proposal_id="test-proposal-789",
            decision=DecisionType.AMEND,
            amendment=amendment,
            rule_ids_triggered=[RuleId.EXPO_PER_MATCH_MAX],
            human_message="Reduced stake due to exposure cap",
            state=GovernanceState.LIVE,
            audit_ref="audit:2025-08-24:ghi789"
        )
        
        assert decision.decision == DecisionType.AMEND
        assert decision.amendment is not None
        assert decision.amendment.stake == 350.0
        assert decision.amendment.odds == 1.76
        assert RuleId.EXPO_PER_MATCH_MAX in decision.rule_ids_triggered


class TestBetAmendment:
    """Test suite for BetAmendment schema"""
    
    def test_amendment_creation(self):
        """Test creating bet amendments"""
        # Stake only amendment
        amendment1 = BetAmendment(stake=250.0)
        assert amendment1.stake == 250.0
        assert amendment1.odds is None
        
        # Odds only amendment
        amendment2 = BetAmendment(odds=1.85)
        assert amendment2.stake is None
        assert amendment2.odds == 1.85
        
        # Both stake and odds
        amendment3 = BetAmendment(stake=300.0, odds=1.90)
        assert amendment3.stake == 300.0
        assert amendment3.odds == 1.90
    
    def test_amendment_validation(self):
        """Test amendment validation"""
        # Invalid odds range
        with pytest.raises(ValueError):
            BetAmendment(odds=0.5)
        
        # Invalid stake (negative)
        with pytest.raises(ValueError):
            BetAmendment(stake=-100.0)


class TestExposureSnapshot:
    """Test suite for ExposureSnapshot schema"""
    
    def test_exposure_snapshot_creation(self):
        """Test creating exposure snapshot"""
        snapshot = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=4500.0,
            daily_pnl=-250.0,
            session_pnl=150.0,
            per_match_exposure={"MI_vs_CSK": 1500.0, "RCB_vs_KKR": 2000.0},
            per_market_exposure={"betfair:1.234": 800.0, "betfair:1.567": 700.0},
            per_correlation_group={"match_winners": 3000.0}
        )
        
        assert snapshot.bankroll == 100000.0
        assert snapshot.open_exposure == 4500.0
        assert snapshot.daily_pnl == -250.0
        assert snapshot.session_pnl == 150.0
        assert len(snapshot.per_match_exposure) == 2
        assert snapshot.per_match_exposure["MI_vs_CSK"] == 1500.0
        assert len(snapshot.per_market_exposure) == 2
        assert len(snapshot.per_correlation_group) == 1
        assert isinstance(snapshot.snapshot_time, datetime)
        assert isinstance(snapshot.session_start, datetime)


class TestAuditRecord:
    """Test suite for AuditRecord schema"""
    
    def test_audit_record_creation(self):
        """Test creating audit record"""
        exposure_snapshot = ExposureSnapshot(
            bankroll=100000.0,
            open_exposure=5000.0,
            daily_pnl=0.0,
            session_pnl=0.0
        )
        
        audit_record = AuditRecord(
            proposal_id="test-proposal-123",
            decision=DecisionType.APPROVE,
            rule_ids=[RuleId.BANKROLL_MAX_EXPOSURE],
            snapshot=exposure_snapshot,
            user_id="user123",
            session_id="session456"
        )
        
        assert audit_record.proposal_id == "test-proposal-123"
        assert audit_record.decision == DecisionType.APPROVE
        assert len(audit_record.rule_ids) == 1
        assert RuleId.BANKROLL_MAX_EXPOSURE in audit_record.rule_ids
        assert audit_record.snapshot.bankroll == 100000.0
        assert audit_record.user_id == "user123"
        assert audit_record.session_id == "session456"
        assert audit_record.entity == "DGL"  # Default value
        assert audit_record.audit_id is not None
        assert isinstance(audit_record.timestamp, datetime)


class TestHealthResponse:
    """Test suite for HealthResponse schema"""
    
    def test_health_response_creation(self):
        """Test creating health response"""
        health = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0,
            components={"rule_engine": "healthy", "audit_system": "healthy"},
            metrics={"total_decisions": 150, "avg_processing_time_ms": 2.5}
        )
        
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.uptime_seconds == 3600.0
        assert len(health.components) == 2
        assert health.components["rule_engine"] == "healthy"
        assert len(health.metrics) == 2
        assert health.metrics["total_decisions"] == 150
        assert isinstance(health.timestamp, datetime)


class TestVersionResponse:
    """Test suite for VersionResponse schema"""
    
    def test_version_response_creation(self):
        """Test creating version response"""
        version = VersionResponse(
            version="1.0.0",
            build_time="2025-08-24T12:00:00Z",
            git_commit="abc123def456",
            config_version="1.0.0"
        )
        
        assert version.service == "DGL"  # Default value
        assert version.version == "1.0.0"
        assert version.build_time == "2025-08-24T12:00:00Z"
        assert version.git_commit == "abc123def456"
        assert version.config_version == "1.0.0"


def run_schema_tests():
    """Run all schema tests"""
    print("📋 Running DGL Schema Tests")
    print("=" * 40)
    
    test_classes = [
        ("BetProposal", TestBetProposal),
        ("GovernanceDecision", TestGovernanceDecision),
        ("BetAmendment", TestBetAmendment),
        ("ExposureSnapshot", TestExposureSnapshot),
        ("AuditRecord", TestAuditRecord),
        ("HealthResponse", TestHealthResponse),
        ("VersionResponse", TestVersionResponse)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for class_name, test_class in test_classes:
        print(f"\n📊 {class_name}")
        print("-" * 30)
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        class_passed = 0
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance and run method
                test_instance = test_class()
                method = getattr(test_instance, test_method)
                method()
                
                print(f"  ✅ {test_method}")
                passed_tests += 1
                class_passed += 1
                
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Class Results: {class_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Schema Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_schema_tests()
    exit(0 if success else 1)
