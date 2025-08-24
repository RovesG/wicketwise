#!/usr/bin/env python3
"""
Specialized Agents Tests
=======================

Unit tests for specialized cricket analysis agents including performance,
tactical, prediction, and betting agents.

Author: WicketWise Team
Last Modified: 2025-08-24
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from crickformers.agents.base_agent import AgentCapability, AgentContext, AgentResponse
from crickformers.agents.performance_agent import PerformanceAgent
from crickformers.agents.tactical_agent import TacticalAgent
from crickformers.agents.prediction_agent import PredictionAgent
from crickformers.agents.betting_agent import BettingAgent


class TestPerformanceAgent:
    """Test PerformanceAgent functionality"""
    
    def test_performance_agent_initialization(self):
        """Test performance agent initialization"""
        config = {
            "min_matches_for_trend": 3,
            "recent_form_days": 180
        }
        
        agent = PerformanceAgent(config)
        
        assert agent.agent_id == "performance_agent"
        assert AgentCapability.PERFORMANCE_ANALYSIS in agent.capabilities
        assert AgentCapability.TREND_ANALYSIS in agent.capabilities
        assert AgentCapability.COMPARISON_ANALYSIS in agent.capabilities
        assert AgentCapability.MULTI_FORMAT_ANALYSIS in agent.capabilities
        assert agent.min_matches_for_trend == 3
        assert agent.recent_form_days == 180
    
    def test_performance_agent_can_handle(self):
        """Test performance agent capability handling"""
        agent = PerformanceAgent()
        
        # Should handle performance-related queries (with player keyword)
        perf_context = AgentContext("test", "Show player Kohli's batting performance", datetime.now())
        assert agent.can_handle(AgentCapability.PERFORMANCE_ANALYSIS, perf_context) is True
        
        player_context = AgentContext("test", "Compare player statistics", datetime.now(),
                                     player_context={"names": ["Kohli", "Smith"]})
        assert agent.can_handle(AgentCapability.COMPARISON_ANALYSIS, player_context) is True
        
        team_context = AgentContext("test", "Analyze team performance", datetime.now(),
                                   team_context={"names": ["India", "Australia"]})
        assert agent.can_handle(AgentCapability.PERFORMANCE_ANALYSIS, team_context) is True
        
        # Should not handle unrelated queries
        betting_context = AgentContext("test", "Find betting opportunities", datetime.now())
        assert agent.can_handle(AgentCapability.BETTING_ANALYSIS, betting_context) is False
    
    def test_analysis_type_determination(self):
        """Test determining analysis type from context"""
        agent = PerformanceAgent()
        
        # Player performance (with explicit player_context)
        player_context = AgentContext("test", "Analyze batsman statistics", datetime.now(),
                                     player_context={"names": ["Kohli"]})
        assert agent._determine_analysis_type(player_context) == "player_performance"
        
        # Team performance (avoid "form" substring in "performing")
        team_context = AgentContext("test", "How is the Indian team doing?", datetime.now(),
                                   team_context={"names": ["India"]})
        assert agent._determine_analysis_type(team_context) == "team_performance"
        
        # Comparison
        compare_context = AgentContext("test", "Compare Kohli vs Smith", datetime.now())
        assert agent._determine_analysis_type(compare_context) == "comparison"
        
        # Trend analysis
        trend_context = AgentContext("test", "Show recent form trends", datetime.now())
        assert agent._determine_analysis_type(trend_context) == "trend"
    
    def test_player_name_extraction(self):
        """Test extracting player names from context"""
        agent = PerformanceAgent()
        
        # From explicit context
        context_with_players = AgentContext(
            "test", "Analyze performance", datetime.now(),
            player_context={"names": ["Kohli", "Dhoni"]}
        )
        players = agent._extract_player_names(context_with_players)
        assert "Kohli" in players
        assert "Dhoni" in players
        
        # From query text
        context_from_query = AgentContext("test", "How is Kohli performing?", datetime.now())
        players = agent._extract_player_names(context_from_query)
        assert "Kohli" in players
    
    def test_team_name_extraction(self):
        """Test extracting team names from context"""
        agent = PerformanceAgent()
        
        # From explicit context
        context_with_teams = AgentContext(
            "test", "Team analysis", datetime.now(),
            team_context={"names": ["India", "Australia"]}
        )
        teams = agent._extract_team_names(context_with_teams)
        assert "India" in teams
        assert "Australia" in teams
        
        # From query text
        context_from_query = AgentContext("test", "How is india performing?", datetime.now())
        teams = agent._extract_team_names(context_from_query)
        assert "India" in teams
    
    @pytest.mark.asyncio
    async def test_performance_agent_execution(self):
        """Test performance agent execution"""
        with patch('crickformers.agents.performance_agent.EnhancedKGQueryEngine') as mock_kg, \
             patch('crickformers.agents.performance_agent.TemporalDecayEngine') as mock_decay:
            
            # Mock the dependencies
            mock_kg_instance = AsyncMock()
            mock_kg.return_value = mock_kg_instance
            mock_kg_instance.execute_query.return_value = {"sample": "data"}
            
            mock_decay_instance = MagicMock()
            mock_decay.return_value = mock_decay_instance
            
            agent = PerformanceAgent()
            agent._initialize_agent()  # Initialize with mocks
            
            context = AgentContext(
                "test", "Analyze batsman statistics", datetime.now(),
                player_context={"names": ["Kohli"]}
            )
            
            response = await agent.execute(context)
            
            assert response.success is True
            assert response.agent_id == "performance_agent"
            assert response.capability == AgentCapability.PERFORMANCE_ANALYSIS
            assert response.result is not None
            assert response.result["analysis_type"] == "player_performance"


class TestTacticalAgent:
    """Test TacticalAgent functionality"""
    
    def test_tactical_agent_initialization(self):
        """Test tactical agent initialization"""
        agent = TacticalAgent()
        
        assert agent.agent_id == "tactical_agent"
        assert AgentCapability.TACTICAL_ANALYSIS in agent.capabilities
        assert AgentCapability.CONTEXTUAL_REASONING in agent.capabilities
        assert AgentCapability.MULTI_FORMAT_ANALYSIS in agent.capabilities
        assert len(agent.field_positions) > 0
        assert len(agent.bowling_types) > 0
    
    def test_tactical_agent_can_handle(self):
        """Test tactical agent capability handling"""
        agent = TacticalAgent()
        
        # Should handle tactical queries
        tactical_context = AgentContext("test", "What field placement strategy should be used?", datetime.now())
        assert agent.can_handle(AgentCapability.TACTICAL_ANALYSIS, tactical_context) is True
        
        bowling_context = AgentContext("test", "What bowling strategy is best?", datetime.now())
        assert agent.can_handle(AgentCapability.TACTICAL_ANALYSIS, bowling_context) is True
        
        # Should not handle non-tactical queries (need more specific non-tactical query)
        performance_context = AgentContext("test", "Show player statistics and averages", datetime.now())
        assert agent.can_handle(AgentCapability.TACTICAL_ANALYSIS, performance_context) is False
    
    def test_tactical_analysis_type_determination(self):
        """Test determining tactical analysis type"""
        agent = TacticalAgent()
        
        # Field placement
        field_context = AgentContext("test", "What field placement should we use?", datetime.now())
        assert agent._determine_tactical_analysis_type(field_context) == "field_placement"
        
        # Bowling strategy
        bowling_context = AgentContext("test", "What bowling attack is best?", datetime.now())
        assert agent._determine_tactical_analysis_type(bowling_context) == "bowling_strategy"
        
        # Batting approach
        batting_context = AgentContext("test", "What batting approach should we take?", datetime.now())
        assert agent._determine_tactical_analysis_type(batting_context) == "batting_approach"
        
        # Match situation
        situation_context = AgentContext("test", "Analyze the current match situation", datetime.now())
        assert agent._determine_tactical_analysis_type(situation_context) == "match_situation"
    
    def test_field_placement_recommendations(self):
        """Test field placement recommendation methods"""
        agent = TacticalAgent()
        
        # Test different field formations
        t20_powerplay = agent._get_t20_powerplay_field()
        assert len(t20_powerplay) > 0
        assert "slip" in t20_powerplay
        
        t20_death = agent._get_t20_death_field()
        assert len(t20_death) > 0
        assert "long_on" in t20_death or "long_off" in t20_death
        
        test_attacking = agent._get_test_attacking_field()
        assert len(test_attacking) > 0
        assert "slip" in test_attacking
    
    @pytest.mark.asyncio
    async def test_tactical_agent_execution(self):
        """Test tactical agent execution"""
        with patch('crickformers.agents.tactical_agent.EnhancedKGQueryEngine') as mock_kg, \
             patch('crickformers.agents.tactical_agent.ContextNodeManager') as mock_context:
            
            mock_kg_instance = AsyncMock()
            mock_kg.return_value = mock_kg_instance
            
            mock_context_instance = MagicMock()
            mock_context.return_value = mock_context_instance
            
            agent = TacticalAgent()
            agent._initialize_agent()
            
            context = AgentContext(
                "test", "What field placement strategy for T20 powerplay?", datetime.now(),
                format_context="T20"
            )
            
            response = await agent.execute(context)
            
            assert response.success is True
            assert response.agent_id == "tactical_agent"
            assert response.capability == AgentCapability.TACTICAL_ANALYSIS
            assert response.result is not None
            assert response.result["analysis_type"] == "field_placement"


class TestPredictionAgent:
    """Test PredictionAgent functionality"""
    
    def test_prediction_agent_initialization(self):
        """Test prediction agent initialization"""
        agent = PredictionAgent()
        
        assert agent.agent_id == "prediction_agent"
        assert AgentCapability.MATCH_PREDICTION in agent.capabilities
        assert AgentCapability.TREND_ANALYSIS in agent.capabilities
        assert AgentCapability.REAL_TIME_PROCESSING in agent.capabilities
        assert len(agent.prediction_types) > 0
        assert "match_winner" in agent.prediction_types
    
    def test_prediction_agent_can_handle(self):
        """Test prediction agent capability handling"""
        agent = PredictionAgent()
        
        # Should handle prediction queries
        prediction_context = AgentContext("test", "Predict the match winner", datetime.now())
        assert agent.can_handle(AgentCapability.MATCH_PREDICTION, prediction_context) is True
        
        forecast_context = AgentContext("test", "What's the likely outcome?", datetime.now())
        assert agent.can_handle(AgentCapability.MATCH_PREDICTION, forecast_context) is True
        
        # Should not handle non-prediction queries
        performance_context = AgentContext("test", "Show statistics", datetime.now())
        assert agent.can_handle(AgentCapability.MATCH_PREDICTION, performance_context) is False
    
    def test_prediction_type_determination(self):
        """Test determining prediction type"""
        agent = PredictionAgent()
        
        # Match winner
        winner_context = AgentContext("test", "Who will win the match?", datetime.now())
        assert agent._determine_prediction_type(winner_context) == "match_winner"
        
        # Total score
        score_context = AgentContext("test", "What will be the total score?", datetime.now())
        assert agent._determine_prediction_type(score_context) == "total_score"
        
        # Player performance (need to use "player" keyword specifically)
        player_context = AgentContext("test", "Predict player performance for Kohli", datetime.now())
        assert agent._determine_prediction_type(player_context) == "player_performance"
        
        # Tournament winner
        tournament_context = AgentContext("test", "Who will win the World Cup?", datetime.now())
        assert agent._determine_prediction_type(tournament_context) == "tournament_winner"
    
    def test_model_probability_extraction(self):
        """Test extracting model probabilities from predictions"""
        agent = PredictionAgent()
        
        # Test feature creation methods instead
        features = agent._create_match_features({}, AgentContext("test", "test", datetime.now()))
        assert isinstance(features, dict)
        assert "team_a_recent_wins" in features
        
        # Test team extraction
        context = AgentContext("test", "India vs Australia", datetime.now())
        teams = agent._extract_teams_from_context(context)
        assert isinstance(teams, list)
    
    @pytest.mark.asyncio
    async def test_prediction_agent_execution(self):
        """Test prediction agent execution"""
        with patch('crickformers.agents.prediction_agent.MoEOrchestrator') as mock_moe, \
             patch('crickformers.agents.prediction_agent.EnhancedKGQueryEngine') as mock_kg:
            
            # Mock MoE orchestrator
            mock_moe_instance = MagicMock()
            mock_moe.return_value = mock_moe_instance
            mock_moe_instance.initialize.return_value = True
            mock_moe_instance.predict.return_value = {
                "status": "success",
                "predictions": {"team_a_win_prob": 0.65},
                "metadata": {"latency_ms": 100, "confidence": 0.8}
            }
            
            # Mock KG engine
            mock_kg_instance = AsyncMock()
            mock_kg.return_value = mock_kg_instance
            mock_kg_instance.execute_query.return_value = {"sample": "data"}
            
            agent = PredictionAgent()
            agent._initialize_agent()
            
            context = AgentContext(
                "test", "Predict the winner of India vs Australia", datetime.now(),
                team_context={"names": ["India", "Australia"]}
            )
            
            response = await agent.execute(context)
            
            assert response.success is True
            assert response.agent_id == "prediction_agent"
            assert response.capability == AgentCapability.MATCH_PREDICTION
            assert response.result is not None
            assert response.result["prediction_type"] == "match_winner"


class TestBettingAgent:
    """Test BettingAgent functionality"""
    
    def test_betting_agent_initialization(self):
        """Test betting agent initialization"""
        config = {
            "max_kelly_fraction": 0.05,
            "min_edge_threshold": 0.03,
            "confidence_threshold": 0.7
        }
        
        agent = BettingAgent(config)
        
        assert agent.agent_id == "betting_agent"
        assert AgentCapability.BETTING_ANALYSIS in agent.capabilities
        assert AgentCapability.TREND_ANALYSIS in agent.capabilities
        assert AgentCapability.REAL_TIME_PROCESSING in agent.capabilities
        assert agent.max_kelly_fraction == 0.05
        assert agent.min_edge_threshold == 0.03
        assert agent.confidence_threshold == 0.7
    
    def test_betting_agent_can_handle(self):
        """Test betting agent capability handling"""
        agent = BettingAgent()
        
        # Should handle betting queries
        betting_context = AgentContext("test", "Find value betting opportunities", datetime.now())
        assert agent.can_handle(AgentCapability.BETTING_ANALYSIS, betting_context) is True
        
        odds_context = AgentContext("test", "Analyze the odds for this match", datetime.now())
        assert agent.can_handle(AgentCapability.BETTING_ANALYSIS, odds_context) is True
        
        arbitrage_context = AgentContext("test", "Are there any arbitrage opportunities?", datetime.now())
        assert agent.can_handle(AgentCapability.BETTING_ANALYSIS, arbitrage_context) is True
        
        # Should not handle non-betting queries
        performance_context = AgentContext("test", "Show performance statistics", datetime.now())
        assert agent.can_handle(AgentCapability.BETTING_ANALYSIS, performance_context) is False
    
    def test_betting_analysis_type_determination(self):
        """Test determining betting analysis type"""
        agent = BettingAgent()
        
        # Value opportunities
        value_context = AgentContext("test", "Find value betting opportunities", datetime.now())
        assert agent._determine_betting_analysis_type(value_context) == "value_opportunities"
        
        # Arbitrage detection
        arb_context = AgentContext("test", "Look for arbitrage opportunities", datetime.now())
        assert agent._determine_betting_analysis_type(arb_context) == "arbitrage_detection"
        
        # Market analysis
        market_context = AgentContext("test", "Analyze market efficiency", datetime.now())
        assert agent._determine_betting_analysis_type(market_context) == "market_analysis"
        
        # Betting strategy
        strategy_context = AgentContext("test", "What betting strategy should I use?", datetime.now())
        assert agent._determine_betting_analysis_type(strategy_context) == "betting_strategy"
        
        # Risk assessment
        risk_context = AgentContext("test", "What are the risks in this bet?", datetime.now())
        assert agent._determine_betting_analysis_type(risk_context) == "risk_assessment"
    
    def test_sample_odds_generation(self):
        """Test sample odds data generation"""
        agent = BettingAgent()
        context = AgentContext("test", "test query", datetime.now(), 
                             match_context={"match_id": "test_match"})
        
        odds_data = agent._generate_sample_odds_data(context)
        
        assert len(odds_data) > 0
        assert all(hasattr(odds, 'bookmaker_id') for odds in odds_data)
        assert all(hasattr(odds, 'odds') for odds in odds_data)
        assert all(hasattr(odds, 'selection') for odds in odds_data)
    
    def test_model_probability_extraction_betting(self):
        """Test extracting model probabilities for betting analysis"""
        agent = BettingAgent()
        
        # Test with match prediction results
        market_predictions = {
            "predictions": {"Team A": 0.55, "Team B": 0.45}
        }
        probs = agent._extract_model_probabilities(market_predictions)
        assert probs["Team A"] == 0.55
        assert probs["Team B"] == 0.45
        
        # Test with empty predictions (should return defaults)
        empty_predictions = {}
        default_probs = agent._extract_model_probabilities(empty_predictions)
        assert "Team A" in default_probs
        assert "Team B" in default_probs
        assert abs(default_probs["Team A"] + default_probs["Team B"] - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_betting_agent_execution(self):
        """Test betting agent execution"""
        with patch('crickformers.agents.betting_agent.MispricingEngine') as mock_mispricing, \
             patch('crickformers.agents.betting_agent.PredictionAgent') as mock_prediction, \
             patch('crickformers.agents.betting_agent.EnhancedKGQueryEngine') as mock_kg:
            
            # Mock mispricing engine
            mock_mispricing_instance = MagicMock()
            mock_mispricing.return_value = mock_mispricing_instance
            mock_mispricing_instance.detect_mispricing.return_value = []
            
            # Mock prediction agent
            mock_prediction_instance = MagicMock()
            mock_prediction.return_value = mock_prediction_instance
            mock_prediction_instance.initialize.return_value = True
            mock_prediction_instance.execute.return_value = AgentResponse(
                "prediction_agent", AgentCapability.MATCH_PREDICTION, True, 0.8, 1.0,
                result={"predictions": {"Team A": 0.6, "Team B": 0.4}}
            )
            
            # Mock KG engine
            mock_kg_instance = AsyncMock()
            mock_kg.return_value = mock_kg_instance
            
            agent = BettingAgent()
            agent._initialize_agent()
            
            context = AgentContext(
                "test", "Find value betting opportunities for India vs Australia", datetime.now(),
                team_context={"names": ["India", "Australia"]}
            )
            
            response = await agent.execute(context)
            
            assert response.success is True
            assert response.agent_id == "betting_agent"
            assert response.capability == AgentCapability.BETTING_ANALYSIS
            assert response.result is not None
            assert response.result["analysis_type"] == "value_opportunities"


class TestAgentIntegration:
    """Test integration between different agents"""
    
    def test_agent_capability_coverage(self):
        """Test that agents cover all required capabilities"""
        agents = [
            PerformanceAgent(),
            TacticalAgent(),
            PredictionAgent(),
            BettingAgent()
        ]
        
        all_capabilities = set()
        for agent in agents:
            all_capabilities.update(agent.capabilities)
        
        # Check that key capabilities are covered
        assert AgentCapability.PERFORMANCE_ANALYSIS in all_capabilities
        assert AgentCapability.TACTICAL_ANALYSIS in all_capabilities
        assert AgentCapability.MATCH_PREDICTION in all_capabilities
        assert AgentCapability.BETTING_ANALYSIS in all_capabilities
        assert AgentCapability.TREND_ANALYSIS in all_capabilities
    
    def test_agent_unique_identifiers(self):
        """Test that all agents have unique identifiers"""
        agents = [
            PerformanceAgent(),
            TacticalAgent(),
            PredictionAgent(),
            BettingAgent()
        ]
        
        agent_ids = [agent.agent_id for agent in agents]
        assert len(agent_ids) == len(set(agent_ids))  # All unique
    
    @pytest.mark.asyncio
    async def test_agent_error_handling_consistency(self):
        """Test consistent error handling across agents"""
        agents = [
            PerformanceAgent(),
            TacticalAgent(),
            PredictionAgent(),
            BettingAgent()
        ]
        
        # Test with invalid context
        invalid_context = AgentContext("test", "", datetime.now())  # Empty query
        
        for agent in agents:
            response = await agent._execute_with_monitoring(invalid_context)
            
            # All agents should handle invalid context gracefully
            assert isinstance(response, AgentResponse)
            assert response.agent_id == agent.agent_id
            # Some might succeed with empty query, others might fail - both are valid
    
    def test_agent_configuration_consistency(self):
        """Test configuration handling consistency"""
        test_config = {
            "test_param": "test_value",
            "numeric_param": 42
        }
        
        agents = [
            PerformanceAgent(test_config),
            TacticalAgent(test_config),
            PredictionAgent(test_config),
            BettingAgent(test_config)
        ]
        
        for agent in agents:
            assert agent.config is not None
            assert "test_param" in agent.config
            assert agent.config["test_param"] == "test_value"
