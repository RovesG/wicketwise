# Purpose: Comprehensive unit tests for mispricing detection engine
# Author: WicketWise Team, Last Modified: 2025-08-23

import pytest
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from crickformers.betting.mispricing_engine import (
    BetType,
    BookmakerType,
    OddsData,
    ImpliedProbability,
    ValueOpportunity,
    OddsConverter,
    MispricingEngine
)


class TestOddsConverter:
    """Test OddsConverter utility functions"""
    
    def test_decimal_to_probability(self):
        """Test decimal odds to probability conversion"""
        assert abs(OddsConverter.decimal_to_probability(2.0) - 0.5) < 1e-10
        assert abs(OddsConverter.decimal_to_probability(4.0) - 0.25) < 1e-10
        assert abs(OddsConverter.decimal_to_probability(1.5) - 0.6667) < 1e-3
        
        # Test invalid odds
        with pytest.raises(ValueError):
            OddsConverter.decimal_to_probability(0.5)
        with pytest.raises(ValueError):
            OddsConverter.decimal_to_probability(1.0)
    
    def test_probability_to_decimal(self):
        """Test probability to decimal odds conversion"""
        assert abs(OddsConverter.probability_to_decimal(0.5) - 2.0) < 1e-10
        assert abs(OddsConverter.probability_to_decimal(0.25) - 4.0) < 1e-10
        assert abs(OddsConverter.probability_to_decimal(0.8) - 1.25) < 1e-10
        
        # Test invalid probabilities
        with pytest.raises(ValueError):
            OddsConverter.probability_to_decimal(0.0)
        with pytest.raises(ValueError):
            OddsConverter.probability_to_decimal(1.0)
        with pytest.raises(ValueError):
            OddsConverter.probability_to_decimal(1.5)
    
    def test_american_to_decimal(self):
        """Test American odds to decimal conversion"""
        assert abs(OddsConverter.american_to_decimal(100) - 2.0) < 1e-10
        assert abs(OddsConverter.american_to_decimal(200) - 3.0) < 1e-10
        assert abs(OddsConverter.american_to_decimal(-200) - 1.5) < 1e-10
        assert abs(OddsConverter.american_to_decimal(-100) - 2.0) < 1e-10
    
    def test_fractional_to_decimal(self):
        """Test fractional odds to decimal conversion"""
        assert abs(OddsConverter.fractional_to_decimal(1, 1) - 2.0) < 1e-10  # 1/1 = 2.0
        assert abs(OddsConverter.fractional_to_decimal(3, 1) - 4.0) < 1e-10  # 3/1 = 4.0
        assert abs(OddsConverter.fractional_to_decimal(1, 2) - 1.5) < 1e-10  # 1/2 = 1.5
        assert abs(OddsConverter.fractional_to_decimal(5, 2) - 3.5) < 1e-10  # 5/2 = 3.5
    
    def test_calculate_overround(self):
        """Test overround calculation"""
        # Perfect market (no overround)
        probs = [0.5, 0.5]
        assert abs(OddsConverter.calculate_overround(probs) - 0.0) < 1e-10
        
        # Market with overround
        probs = [0.55, 0.55]  # Total = 1.1, overround = 0.1
        assert abs(OddsConverter.calculate_overround(probs) - 0.1) < 1e-10
        
        # Three-way market
        probs = [0.4, 0.35, 0.3]  # Total = 1.05, overround = 0.05
        assert abs(OddsConverter.calculate_overround(probs) - 0.05) < 1e-10
    
    def test_remove_overround(self):
        """Test overround removal"""
        # Market with overround
        probs = [0.55, 0.55]  # Total = 1.1
        true_probs = OddsConverter.remove_overround(probs)
        
        assert abs(sum(true_probs) - 1.0) < 1e-10
        assert abs(true_probs[0] - 0.5) < 1e-10
        assert abs(true_probs[1] - 0.5) < 1e-10
        
        # Market without overround
        probs_fair = [0.5, 0.5]
        true_probs_fair = OddsConverter.remove_overround(probs_fair)
        assert true_probs_fair == probs_fair


class TestOddsData:
    """Test OddsData dataclass"""
    
    def test_basic_creation(self):
        """Test basic odds data creation"""
        odds = OddsData(
            bookmaker_id="bet365",
            bookmaker_type=BookmakerType.REGULATED,
            bet_type=BetType.MATCH_WINNER,
            market_description="Match Winner",
            odds=2.5,
            stake_limit=1000.0,
            timestamp=datetime.now(),
            match_id="match_123",
            selection="Team A"
        )
        
        assert odds.bookmaker_id == "bet365"
        assert odds.bookmaker_type == BookmakerType.REGULATED
        assert odds.bet_type == BetType.MATCH_WINNER
        assert odds.odds == 2.5
        assert odds.selection == "Team A"
        assert len(odds.metadata) == 0
    
    def test_with_metadata(self):
        """Test odds data with metadata"""
        metadata = {"region": "UK", "currency": "GBP"}
        
        odds = OddsData(
            bookmaker_id="betfair",
            bookmaker_type=BookmakerType.EXCHANGE,
            bet_type=BetType.TOTAL_RUNS,
            market_description="Total Runs Over/Under",
            odds=1.9,
            stake_limit=5000.0,
            timestamp=datetime.now(),
            match_id="match_456",
            selection="Over 160.5",
            metadata=metadata
        )
        
        assert odds.metadata == metadata
        assert odds.bookmaker_type == BookmakerType.EXCHANGE


class TestMispricingEngine:
    """Test MispricingEngine core functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create mispricing engine for testing"""
        config = {
            "min_edge_threshold": 0.05,
            "max_kelly_fraction": 0.25,
            "confidence_threshold": 0.4,  # Lower threshold for testing
            "max_odds_age_minutes": 10
        }
        return MispricingEngine(config)
    
    @pytest.fixture
    def sample_odds(self):
        """Create sample odds data"""
        base_time = datetime.now()
        
        return [
            # Team A odds from multiple bookmakers
            OddsData(
                bookmaker_id="bet365",
                bookmaker_type=BookmakerType.REGULATED,
                bet_type=BetType.MATCH_WINNER,
                market_description="Match Winner",
                odds=2.2,
                stake_limit=1000.0,
                timestamp=base_time,
                match_id="match_123",
                selection="Team A"
            ),
            OddsData(
                bookmaker_id="betfair",
                bookmaker_type=BookmakerType.EXCHANGE,
                bet_type=BetType.MATCH_WINNER,
                market_description="Match Winner",
                odds=2.4,
                stake_limit=5000.0,
                timestamp=base_time,
                match_id="match_123",
                selection="Team A"
            ),
            OddsData(
                bookmaker_id="sportsbet",
                bookmaker_type=BookmakerType.SOFT,
                bet_type=BetType.MATCH_WINNER,
                market_description="Match Winner",
                odds=2.1,
                stake_limit=800.0,
                timestamp=base_time,
                match_id="match_123",
                selection="Team A"
            ),
            # Team B odds from multiple bookmakers
            OddsData(
                bookmaker_id="pinnacle",
                bookmaker_type=BookmakerType.SHARP,
                bet_type=BetType.MATCH_WINNER,
                market_description="Match Winner",
                odds=1.8,
                stake_limit=2000.0,
                timestamp=base_time,
                match_id="match_123",
                selection="Team B"
            ),
            OddsData(
                bookmaker_id="bet365",
                bookmaker_type=BookmakerType.REGULATED,
                bet_type=BetType.MATCH_WINNER,
                market_description="Match Winner",
                odds=1.75,
                stake_limit=1000.0,
                timestamp=base_time,
                match_id="match_123",
                selection="Team B"
            ),
            OddsData(
                bookmaker_id="betfair",
                bookmaker_type=BookmakerType.EXCHANGE,
                bet_type=BetType.MATCH_WINNER,
                market_description="Match Winner",
                odds=1.85,
                stake_limit=5000.0,
                timestamp=base_time,
                match_id="match_123",
                selection="Team B"
            )
        ]
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.min_edge_threshold == 0.05
        assert engine.max_kelly_fraction == 0.25
        assert engine.confidence_threshold == 0.4
        assert engine.max_odds_age_minutes == 10
        assert len(engine.odds_history) == 0
        assert len(engine.mispricing_history) == 0
    
    def test_default_initialization(self):
        """Test engine initialization with default config"""
        engine = MispricingEngine()
        
        assert engine.min_edge_threshold == 0.05
        assert engine.max_kelly_fraction == 0.25
        assert engine.confidence_threshold == 0.7
        assert engine.max_odds_age_minutes == 10
    
    def test_add_odds_data(self, engine, sample_odds):
        """Test adding odds data to engine"""
        engine.add_odds_data(sample_odds)
        
        assert len(engine.odds_history) == 6
        
        # All odds should be in history
        assert all(odds in engine.odds_history for odds in sample_odds)
    
    def test_add_stale_odds_filtering(self, engine):
        """Test filtering of stale odds data"""
        current_time = datetime.now()
        
        # Create mix of fresh and stale odds
        fresh_odds = OddsData(
            bookmaker_id="bet365",
            bookmaker_type=BookmakerType.REGULATED,
            bet_type=BetType.MATCH_WINNER,
            market_description="Match Winner",
            odds=2.0,
            stake_limit=1000.0,
            timestamp=current_time - timedelta(minutes=5),  # Fresh
            match_id="match_123",
            selection="Team A"
        )
        
        stale_odds = OddsData(
            bookmaker_id="betfair",
            bookmaker_type=BookmakerType.EXCHANGE,
            bet_type=BetType.MATCH_WINNER,
            market_description="Match Winner",
            odds=2.1,
            stake_limit=1000.0,
            timestamp=current_time - timedelta(minutes=15),  # Stale
            match_id="match_123",
            selection="Team A"
        )
        
        engine.add_odds_data([fresh_odds, stale_odds])
        
        # Only fresh odds should be added
        assert len(engine.odds_history) == 1
        assert engine.odds_history[0] == fresh_odds
    
    def test_calculate_implied_probabilities(self, engine, sample_odds):
        """Test implied probability calculation"""
        implied_probs = engine.calculate_implied_probabilities(sample_odds)
        
        # Should have probabilities for both teams
        assert "Team A" in implied_probs
        assert "Team B" in implied_probs
        
        # Check Team A probability (weighted average of 2.2, 2.4, and 2.1 odds)
        team_a_prob = implied_probs["Team A"]
        assert isinstance(team_a_prob, ImpliedProbability)
        assert 0.4 < team_a_prob.probability < 0.5  # Should be around 1/2.25
        assert team_a_prob.sample_size == 3
        
        # Check Team B probability (weighted average of 1.8, 1.75, and 1.85 odds)
        team_b_prob = implied_probs["Team B"]
        assert isinstance(team_b_prob, ImpliedProbability)
        assert 0.5 < team_b_prob.probability < 0.6  # Should be around 1/1.8
        assert team_b_prob.sample_size == 3
    
    def test_detect_mispricing_value_opportunity(self, engine, sample_odds):
        """Test detection of value opportunities"""
        engine.add_odds_data(sample_odds)
        
        # Model predicts Team A has 60% chance (higher than market ~43%)
        model_predictions = {
            "Team A": 0.6,
            "Team B": 0.4
        }
        
        opportunities = engine.detect_mispricing(model_predictions, sample_odds, "match_123")
        
        # Should find value opportunity for Team A
        assert len(opportunities) >= 1
        
        team_a_opp = next((opp for opp in opportunities if opp.selection == "Team A"), None)
        assert team_a_opp is not None
        assert team_a_opp.expected_value > 0.05  # Should exceed minimum threshold
        assert team_a_opp.kelly_fraction > 0
        assert team_a_opp.confidence_score >= 0.4
    
    def test_detect_mispricing_no_value(self, engine, sample_odds):
        """Test when no value opportunities exist"""
        engine.add_odds_data(sample_odds)
        
        # Model predictions align exactly with adjusted market probabilities
        market_probs = engine.calculate_implied_probabilities(sample_odds)
        model_predictions = {
            "Team A": float(market_probs["Team A"].true_probability),
            "Team B": float(market_probs["Team B"].true_probability)
        }
        
        opportunities = engine.detect_mispricing(model_predictions, sample_odds, "match_123")
        
        # Should find no significant value opportunities
        assert len(opportunities) == 0
    
    def test_kelly_fraction_calculation(self, engine):
        """Test Kelly criterion calculation"""
        # Positive Kelly scenario
        kelly = engine._calculate_kelly_fraction(0.6, 2.0)  # 60% prob, 2.0 odds
        assert kelly > 0
        assert kelly < 1
        
        # No edge scenario
        kelly_no_edge = engine._calculate_kelly_fraction(0.5, 2.0)  # 50% prob, 2.0 odds
        assert kelly_no_edge == 0.0
        
        # Negative edge scenario
        kelly_negative = engine._calculate_kelly_fraction(0.4, 2.0)  # 40% prob, 2.0 odds
        assert kelly_negative == 0.0  # Should not bet negative Kelly
    
    def test_expected_value_calculation(self, engine):
        """Test expected value calculation"""
        # Positive EV scenario
        ev_positive = engine._calculate_expected_value(0.6, 2.0)  # 60% prob, 2.0 odds
        expected_ev = (0.6 * 1.0) - (0.4 * 1.0)  # (prob * profit) - (prob_lose * stake)
        assert abs(ev_positive - expected_ev) < 1e-10
        assert ev_positive > 0
        
        # Negative EV scenario
        ev_negative = engine._calculate_expected_value(0.4, 2.0)  # 40% prob, 2.0 odds
        assert ev_negative < 0
        
        # Break-even scenario
        ev_breakeven = engine._calculate_expected_value(0.5, 2.0)  # 50% prob, 2.0 odds
        assert abs(ev_breakeven) < 1e-10
    
    def test_confidence_score_calculation(self, engine):
        """Test confidence score calculation"""
        # High confidence: Large probability difference, many bookmakers
        high_conf = engine._calculate_confidence_score(0.7, 0.4, 5)
        assert high_conf > 0.8
        
        # Low confidence: Small probability difference, few bookmakers
        low_conf = engine._calculate_confidence_score(0.52, 0.5, 1)
        assert low_conf < 0.3
        
        # Medium confidence
        med_conf = engine._calculate_confidence_score(0.6, 0.45, 3)
        assert 0.3 < med_conf < 0.8
    
    def test_risk_level_assessment(self, engine):
        """Test risk level assessment"""
        # Low risk: High confidence, low Kelly
        risk_low = engine._assess_risk_level(0.1, 0.05, 0.9)
        assert risk_low == "low"
        
        # High risk: Low confidence or high Kelly
        risk_high_conf = engine._assess_risk_level(0.1, 0.05, 0.5)
        assert risk_high_conf == "high"
        
        risk_high_kelly = engine._assess_risk_level(0.1, 0.2, 0.9)
        assert risk_high_kelly == "high"
        
        # Medium risk
        risk_medium = engine._assess_risk_level(0.1, 0.1, 0.75)
        assert risk_medium == "medium"
    
    def test_find_best_odds(self, engine, sample_odds):
        """Test finding best odds for a selection"""
        # Team A has odds of 2.2, 2.4, and 2.1, best should be 2.4
        best_team_a = engine._find_best_odds(sample_odds, "Team A")
        assert best_team_a is not None
        assert best_team_a.odds == 2.4
        assert best_team_a.bookmaker_id == "betfair"
        
        # Team B has odds of 1.8, 1.75, and 1.85, best should be 1.85
        best_team_b = engine._find_best_odds(sample_odds, "Team B")
        assert best_team_b is not None
        assert best_team_b.odds == 1.85
        
        # Non-existent selection
        best_none = engine._find_best_odds(sample_odds, "Team C")
        assert best_none is None
    
    def test_detect_arbitrage_opportunities(self, engine):
        """Test arbitrage detection"""
        # Create arbitrage scenario
        current_time = datetime.now()
        
        arbitrage_odds = [
            OddsData(
                bookmaker_id="bookmaker1",
                bookmaker_type=BookmakerType.SOFT,
                bet_type=BetType.MATCH_WINNER,
                market_description="Match Winner",
                odds=2.1,  # Team A
                stake_limit=1000.0,
                timestamp=current_time,
                match_id="arb_match",
                selection="Team A"
            ),
            OddsData(
                bookmaker_id="bookmaker2", 
                bookmaker_type=BookmakerType.SOFT,
                bet_type=BetType.MATCH_WINNER,
                market_description="Match Winner",
                odds=2.1,  # Team B
                stake_limit=1000.0,
                timestamp=current_time,
                match_id="arb_match",
                selection="Team B"
            )
        ]
        
        # Total implied probability = 1/2.1 + 1/2.1 = 0.952 < 1.0 (arbitrage!)
        arb_opportunities = engine.detect_arbitrage_opportunities(arbitrage_odds)
        
        assert len(arb_opportunities) == 1
        arb_opp = arb_opportunities[0]
        
        assert arb_opp["match_id"] == "arb_match"
        assert arb_opp["profit_margin"] > 0
        assert arb_opp["total_implied_prob"] < 0.98
        assert "optimal_stakes" in arb_opp
        assert "Team A" in arb_opp["optimal_stakes"]
        assert "Team B" in arb_opp["optimal_stakes"]
    
    def test_arbitrage_detection(self, engine, sample_odds):
        """Test arbitrage detection with sample odds"""
        # Market with arbitrage opportunity (total implied prob < 1.0)
        arb_opportunities = engine.detect_arbitrage_opportunities(sample_odds)
        
        # Should find arbitrage opportunity
        assert len(arb_opportunities) >= 1
        
        # Check arbitrage details
        arb = arb_opportunities[0]
        assert arb["profit_margin"] > 0
        assert "optimal_stakes" in arb
        assert len(arb["optimal_stakes"]) == 2  # Two outcomes
    
    def test_no_arbitrage_detection(self):
        """Test when no arbitrage opportunities exist"""
        engine = MispricingEngine()
        
        current_time = datetime.now()
        
        # Create odds with high overround (no arbitrage)
        high_overround_odds = [
            OddsData("bet365", BookmakerType.REGULATED, BetType.MATCH_WINNER,
                    "Match Winner", 1.8, 1000.0, current_time, "match_456", "Team A"),
            OddsData("sportsbet", BookmakerType.SOFT, BetType.MATCH_WINNER,
                    "Match Winner", 2.0, 500.0, current_time, "match_456", "Team B")
        ]
        
        # Total implied prob: 1/1.8 + 1/2.0 = 0.556 + 0.5 = 1.056 > 1.0 (no arbitrage)
        arb_opportunities = engine.detect_arbitrage_opportunities(high_overround_odds)
        
        # Should find no arbitrage
        assert len(arb_opportunities) == 0
    
    def test_analyze_market_efficiency(self, engine, sample_odds):
        """Test market efficiency analysis"""
        engine.add_odds_data(sample_odds)
        
        analysis = engine.analyze_market_efficiency("match_123")
        
        assert "match_id" in analysis
        assert analysis["match_id"] == "match_123"
        assert "bet_types" in analysis
        assert "overall_efficiency" in analysis
        
        # Should have analysis for match winner bet type
        bet_types = analysis["bet_types"]
        assert "match_winner" in bet_types
        
        match_winner_analysis = bet_types["match_winner"]
        assert "num_bookmakers" in match_winner_analysis
        assert "odds_spread" in match_winner_analysis
        assert "avg_overround" in match_winner_analysis
        assert "market_efficiency_score" in match_winner_analysis
        assert "implied_probabilities" in match_winner_analysis
        
        # Check efficiency score is between 0 and 1
        efficiency = match_winner_analysis["market_efficiency_score"]
        assert 0 <= efficiency <= 1
    
    def test_analyze_market_efficiency_no_data(self, engine):
        """Test market efficiency analysis with no data"""
        analysis = engine.analyze_market_efficiency("nonexistent_match")
        
        assert "error" in analysis
        assert analysis["error"] == "No odds data found for match"
    
    def test_performance_summary_empty(self, engine):
        """Test performance summary with no data"""
        summary = engine.get_performance_summary()
        
        assert "message" in summary
        assert summary["message"] == "No mispricing opportunities tracked yet"
    
    def test_performance_summary_with_data(self, engine):
        """Test performance summary with opportunity data"""
        # Create mock opportunities
        opportunities = [
            ValueOpportunity(
                opportunity_id="opp_1",
                match_id="match_1",
                bet_type=BetType.MATCH_WINNER,
                selection="Team A",
                bookmaker_odds=2.5,
                model_probability=0.5,
                implied_probability=0.4,
                expected_value=0.1,
                kelly_fraction=0.08,
                confidence_score=0.8,
                risk_level="low",
                max_stake=1000.0,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=15)
            ),
            ValueOpportunity(
                opportunity_id="opp_2",
                match_id="match_2", 
                bet_type=BetType.TOTAL_RUNS,
                selection="Over 160.5",
                bookmaker_odds=1.9,
                model_probability=0.6,
                implied_probability=0.53,
                expected_value=0.07,
                kelly_fraction=0.05,
                confidence_score=0.75,
                risk_level="medium",
                max_stake=2000.0,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=15)
            )
        ]
        
        engine.mispricing_history.extend(opportunities)
        
        summary = engine.get_performance_summary()
        
        assert summary["total_opportunities"] == 2
        assert "expected_value_stats" in summary
        assert "kelly_fraction_stats" in summary
        assert "confidence_stats" in summary
        assert "risk_distribution" in summary
        assert "bet_type_distribution" in summary
        
        # Check statistics
        ev_stats = summary["expected_value_stats"]
        assert ev_stats["mean"] == 0.085  # (0.1 + 0.07) / 2
        assert ev_stats["min"] == 0.07
        assert ev_stats["max"] == 0.1
        
        # Check risk distribution
        risk_dist = summary["risk_distribution"]
        assert risk_dist["low"] == 1
        assert risk_dist["medium"] == 1
    
    def test_clear_expired_opportunities(self, engine):
        """Test clearing expired opportunities"""
        current_time = datetime.now()
        
        # Create mix of expired and valid opportunities
        valid_opp = ValueOpportunity(
            opportunity_id="valid",
            match_id="match_1",
            bet_type=BetType.MATCH_WINNER,
            selection="Team A",
            bookmaker_odds=2.0,
            model_probability=0.6,
            implied_probability=0.5,
            expected_value=0.1,
            kelly_fraction=0.08,
            confidence_score=0.8,
            risk_level="low",
            max_stake=1000.0,
            timestamp=current_time,
            expires_at=current_time + timedelta(minutes=30)  # Valid
        )
        
        expired_opp = ValueOpportunity(
            opportunity_id="expired",
            match_id="match_2",
            bet_type=BetType.MATCH_WINNER,
            selection="Team B",
            bookmaker_odds=1.8,
            model_probability=0.7,
            implied_probability=0.56,
            expected_value=0.12,
            kelly_fraction=0.1,
            confidence_score=0.85,
            risk_level="low",
            max_stake=1000.0,
            timestamp=current_time - timedelta(minutes=30),
            expires_at=current_time - timedelta(minutes=10)  # Expired
        )
        
        engine.mispricing_history = [valid_opp, expired_opp]
        
        removed_count = engine.clear_expired_opportunities()
        
        assert removed_count == 1
        assert len(engine.mispricing_history) == 1
        assert engine.mispricing_history[0] == valid_opp
    
    def test_efficiency_score_calculation(self, engine):
        """Test market efficiency score calculation"""
        # High efficiency: Low spread, low overround
        high_eff = engine._calculate_efficiency_score(0.1, 0.02)
        assert high_eff > 0.8
        
        # Low efficiency: High spread, high overround
        low_eff = engine._calculate_efficiency_score(1.5, 0.15)
        assert low_eff < 0.3
        
        # Medium efficiency
        med_eff = engine._calculate_efficiency_score(0.5, 0.08)
        assert 0.3 < med_eff < 0.8


class TestIntegrationScenarios:
    """Test realistic betting scenarios"""
    
    def test_ipl_match_winner_scenario(self):
        """Test IPL match winner betting scenario"""
        engine = MispricingEngine()
        
        # Realistic IPL match odds
        current_time = datetime.now()
        
        odds_data = [
            # RCB vs MI - Multiple bookmakers
            OddsData("bet365", BookmakerType.REGULATED, BetType.MATCH_WINNER, 
                    "Match Winner", 1.85, 1000.0, current_time, "ipl_rcb_mi", "RCB"),
            OddsData("betfair", BookmakerType.EXCHANGE, BetType.MATCH_WINNER,
                    "Match Winner", 1.90, 5000.0, current_time, "ipl_rcb_mi", "RCB"),
            OddsData("pinnacle", BookmakerType.SHARP, BetType.MATCH_WINNER,
                    "Match Winner", 1.88, 2000.0, current_time, "ipl_rcb_mi", "RCB"),
            
            OddsData("bet365", BookmakerType.REGULATED, BetType.MATCH_WINNER,
                    "Match Winner", 2.05, 1000.0, current_time, "ipl_rcb_mi", "MI"),
            OddsData("betfair", BookmakerType.EXCHANGE, BetType.MATCH_WINNER,
                    "Match Winner", 2.10, 5000.0, current_time, "ipl_rcb_mi", "MI"),
            OddsData("pinnacle", BookmakerType.SHARP, BetType.MATCH_WINNER,
                    "Match Winner", 2.08, 2000.0, current_time, "ipl_rcb_mi", "MI")
        ]
        
        # AI model predictions (RCB slightly favored)
        model_predictions = {
            "RCB": 0.58,  # Model thinks RCB has 58% chance
            "MI": 0.42    # Market implies ~48% for RCB, ~52% for MI
        }
        
        opportunities = engine.detect_mispricing(model_predictions, odds_data, "ipl_rcb_mi")
        
        # Should find value opportunity for RCB (model higher than market)
        rcb_opportunities = [opp for opp in opportunities if opp.selection == "RCB"]
        assert len(rcb_opportunities) >= 1
        
        if rcb_opportunities:
            rcb_opp = rcb_opportunities[0]
            assert rcb_opp.expected_value > 0
            assert rcb_opp.kelly_fraction > 0
            assert rcb_opp.bookmaker_odds == 1.90  # Best odds (Betfair)
    
    def test_total_runs_betting_scenario(self):
        """Test total runs over/under betting scenario"""
        engine = MispricingEngine({"confidence_threshold": 0.4})  # Lower threshold for testing
        
        current_time = datetime.now()
        
        # Total runs market: Over/Under 165.5 with more bookmakers for confidence
        odds_data = [
            OddsData("sportsbet", BookmakerType.SOFT, BetType.TOTAL_RUNS,
                    "Total Runs", 1.95, 500.0, current_time, "match_total", "Over 165.5"),
            OddsData("bet365", BookmakerType.REGULATED, BetType.TOTAL_RUNS,
                    "Total Runs", 1.92, 1000.0, current_time, "match_total", "Over 165.5"),
            OddsData("pinnacle", BookmakerType.SHARP, BetType.TOTAL_RUNS,
                    "Total Runs", 1.98, 2000.0, current_time, "match_total", "Over 165.5"),
            
            OddsData("sportsbet", BookmakerType.SOFT, BetType.TOTAL_RUNS,
                    "Total Runs", 1.85, 500.0, current_time, "match_total", "Under 165.5"),
            OddsData("bet365", BookmakerType.REGULATED, BetType.TOTAL_RUNS,
                    "Total Runs", 1.88, 1000.0, current_time, "match_total", "Under 165.5"),
            OddsData("pinnacle", BookmakerType.SHARP, BetType.TOTAL_RUNS,
                    "Total Runs", 1.90, 2000.0, current_time, "match_total", "Under 165.5")
        ]
        
        # Model predicts 70% chance of over 165.5 runs
        model_predictions = {
            "Over 165.5": 0.70,
            "Under 165.5": 0.30
        }
        
        opportunities = engine.detect_mispricing(model_predictions, odds_data, "match_total")
        
        # Should find strong value on Over 165.5
        over_opportunities = [opp for opp in opportunities if "Over" in opp.selection]
        assert len(over_opportunities) >= 1
        
        if over_opportunities:
            over_opp = over_opportunities[0]
            assert over_opp.expected_value > 0.15  # Should be significant edge
            assert over_opp.bookmaker_odds == 1.98  # Best available odds (Pinnacle)
    
    def test_multi_market_analysis(self):
        """Test analysis across multiple markets"""
        engine = MispricingEngine()
        
        current_time = datetime.now()
        
        # Multiple markets for same match
        multi_market_odds = [
            # Match winner
            OddsData("bet365", BookmakerType.REGULATED, BetType.MATCH_WINNER,
                    "Match Winner", 1.8, 1000.0, current_time, "multi_match", "Team A"),
            OddsData("bet365", BookmakerType.REGULATED, BetType.MATCH_WINNER,
                    "Match Winner", 2.0, 1000.0, current_time, "multi_match", "Team B"),
            
            # Total runs
            OddsData("bet365", BookmakerType.REGULATED, BetType.TOTAL_RUNS,
                    "Total Runs", 1.9, 1000.0, current_time, "multi_match", "Over 160.5"),
            OddsData("bet365", BookmakerType.REGULATED, BetType.TOTAL_RUNS,
                    "Total Runs", 1.9, 1000.0, current_time, "multi_match", "Under 160.5"),
            
            # Player runs
            OddsData("bet365", BookmakerType.REGULATED, BetType.PLAYER_RUNS,
                    "Player Runs", 2.5, 500.0, current_time, "multi_match", "Kohli Over 30.5")
        ]
        
        engine.add_odds_data(multi_market_odds)
        
        analysis = engine.analyze_market_efficiency("multi_match")
        
        assert len(analysis["bet_types"]) == 3  # Three different bet types
        assert "match_winner" in analysis["bet_types"]
        assert "total_runs" in analysis["bet_types"]
        assert "player_runs" in analysis["bet_types"]
        
        # Overall efficiency should be calculated
        assert 0 <= analysis["overall_efficiency"] <= 1
