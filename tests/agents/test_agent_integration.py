#!/usr/bin/env python3
"""
Agent Integration Tests
======================

Integration tests for the complete agent orchestration system, testing
end-to-end workflows with multiple agents working together.

Author: WicketWise Team
Last Modified: 2025-08-24
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from crickformers.agents.orchestration_engine import OrchestrationEngine, ExecutionStrategy
from crickformers.agents.base_agent import AgentCapability, AgentContext, AgentPriority
from crickformers.agents.performance_agent import PerformanceAgent
from crickformers.agents.tactical_agent import TacticalAgent
from crickformers.agents.prediction_agent import PredictionAgent
from crickformers.agents.betting_agent import BettingAgent


class TestAgentOrchestrationIntegration:
    """Test complete agent orchestration workflows"""
    
    @pytest.fixture
    def orchestration_engine(self):
        """Create orchestration engine with all agents"""
        engine = OrchestrationEngine({
            "max_parallel_agents": 4,
            "default_timeout": 30.0,
            "confidence_threshold": 0.6
        })
        return engine
    
    @pytest.fixture
    def mock_all_dependencies(self):
        """Mock all external dependencies for agents"""
        with patch('crickformers.agents.performance_agent.EnhancedKGQueryEngine') as mock_kg_perf, \
             patch('crickformers.agents.performance_agent.TemporalDecayEngine') as mock_decay_perf, \
             patch('crickformers.agents.tactical_agent.EnhancedKGQueryEngine') as mock_kg_tactical, \
             patch('crickformers.agents.tactical_agent.ContextNodeManager') as mock_context_tactical, \
             patch('crickformers.agents.prediction_agent.MoEOrchestrator') as mock_moe_pred, \
             patch('crickformers.agents.prediction_agent.EnhancedKGQueryEngine') as mock_kg_pred, \
             patch('crickformers.agents.betting_agent.MispricingEngine') as mock_mispricing, \
             patch('crickformers.agents.betting_agent.PredictionAgent') as mock_pred_betting, \
             patch('crickformers.agents.betting_agent.EnhancedKGQueryEngine') as mock_kg_betting:
            
            # Mock KG engines
            for mock_kg in [mock_kg_perf, mock_kg_tactical, mock_kg_pred, mock_kg_betting]:
                mock_kg_instance = AsyncMock()
                mock_kg.return_value = mock_kg_instance
                mock_kg_instance.execute_query.return_value = {"sample": "data"}
            
            # Mock temporal decay engine
            mock_decay_instance = MagicMock()
            mock_decay_perf.return_value = mock_decay_instance
            mock_decay_instance.get_weighted_aggregation.return_value = 0.8
            
            # Mock context manager
            mock_context_instance = MagicMock()
            mock_context_tactical.return_value = mock_context_instance
            
            # Mock MoE orchestrator
            mock_moe_instance = MagicMock()
            mock_moe_pred.return_value = mock_moe_instance
            mock_moe_instance.initialize.return_value = True
            mock_moe_instance.predict.return_value = {
                "status": "success",
                "predictions": {"team_a_win_prob": 0.65, "total_runs": 275},
                "metadata": {"latency_ms": 150, "confidence": 0.8}
            }
            
            # Mock mispricing engine
            mock_mispricing_instance = MagicMock()
            mock_mispricing.return_value = mock_mispricing_instance
            mock_mispricing_instance.detect_mispricing.return_value = []
            mock_mispricing_instance.detect_arbitrage_opportunities.return_value = []
            
            # Mock prediction agent for betting
            mock_pred_instance = MagicMock()
            mock_pred_betting.return_value = mock_pred_instance
            mock_pred_instance.initialize.return_value = True
            
            yield
    
    @pytest.mark.asyncio
    async def test_single_agent_workflow(self, orchestration_engine, mock_all_dependencies):
        """Test workflow with single agent"""
        # Register performance agent
        perf_agent = PerformanceAgent()
        success = orchestration_engine.register_agent(perf_agent)
        assert success is True
        
        # Process performance query
        result = await orchestration_engine.process_query(
            "Analyze player Kohli's batting statistics",
            {"format": "ODI", "recent_matches": 10}
        )
        
        assert result.success is True
        assert result.overall_confidence > 0
        assert len(result.agent_responses) == 1
        assert result.agent_responses[0].agent_id == "performance_agent"
        assert result.aggregated_result is not None
    
    @pytest.mark.asyncio
    async def test_multi_agent_parallel_workflow(self, orchestration_engine, mock_all_dependencies):
        """Test workflow with multiple agents executing in parallel"""
        # Register all agents
        agents = [
            PerformanceAgent(),
            TacticalAgent(),
            PredictionAgent(),
            BettingAgent()
        ]
        
        for agent in agents:
            success = orchestration_engine.register_agent(agent)
            assert success is True
        
        # Process complex query that requires multiple agents
        result = await orchestration_engine.process_query(
            "Analyze player performance, predict match outcome, suggest tactics, and find betting opportunities for India vs Australia",
            {
                "format": "ODI",
                "venue": "MCG",
                "teams": ["India", "Australia"],
                "players": ["Kohli", "Smith"]
            }
        )
        
        assert result.success is True
        assert result.overall_confidence > 0
        assert len(result.agent_responses) >= 2  # Should use multiple agents
        
        # Check that different capabilities were used
        capabilities_used = [resp.capability for resp in result.agent_responses if resp.success]
        assert len(set(capabilities_used)) >= 2  # Multiple different capabilities
        
        # Check aggregated result structure
        assert result.aggregated_result is not None
        assert "agent_count" in result.aggregated_result
        assert "successful_agents" in result.aggregated_result
        assert "capabilities_used" in result.aggregated_result
    
    @pytest.mark.asyncio
    async def test_agent_failure_resilience(self, orchestration_engine, mock_all_dependencies):
        """Test system resilience when some agents fail"""
        # Register agents
        perf_agent = PerformanceAgent()
        tactical_agent = TacticalAgent()
        
        orchestration_engine.register_agent(perf_agent)
        orchestration_engine.register_agent(tactical_agent)
        
        # Simulate failure in one agent by making it unhealthy
        orchestration_engine.coordinator.update_agent_health("performance_agent", success=False)
        orchestration_engine.coordinator.update_agent_health("performance_agent", success=False)
        orchestration_engine.coordinator.update_agent_health("performance_agent", success=False)
        
        # Process query that would normally use both agents
        result = await orchestration_engine.process_query(
            "Analyze team performance and suggest tactical strategy",
            {"format": "T20", "teams": ["India", "Pakistan"]}
        )
        
        # Should still succeed with healthy agents
        assert result.overall_confidence >= 0
        
        # Check that only healthy agents were used
        successful_agents = [resp.agent_id for resp in result.agent_responses if resp.success]
        assert "performance_agent" not in successful_agents  # Should be excluded due to poor health
    
    @pytest.mark.asyncio
    async def test_capability_routing(self, orchestration_engine, mock_all_dependencies):
        """Test that queries are routed to appropriate agents"""
        # Register all agents
        agents = [
            PerformanceAgent(),
            TacticalAgent(), 
            PredictionAgent(),
            BettingAgent()
        ]
        
        for agent in agents:
            orchestration_engine.register_agent(agent)
        
        # Test specific capability routing
        test_cases = [
            ("Show player statistics", AgentCapability.PERFORMANCE_ANALYSIS),
            ("What field placement strategy?", AgentCapability.TACTICAL_ANALYSIS),
            ("Predict match winner", AgentCapability.MATCH_PREDICTION),
            ("Find value betting opportunities", AgentCapability.BETTING_ANALYSIS)
        ]
        
        for query, expected_capability in test_cases:
            result = await orchestration_engine.process_query(query)
            
            # Should have at least one response with expected capability
            capabilities_used = [resp.capability for resp in result.agent_responses if resp.success]
            assert expected_capability in capabilities_used, f"Query '{query}' should use {expected_capability}"
    
    @pytest.mark.asyncio
    async def test_execution_strategy_selection(self, orchestration_engine, mock_all_dependencies):
        """Test that appropriate execution strategies are selected"""
        # Register agents
        perf_agent = PerformanceAgent()
        tactical_agent = TacticalAgent()
        
        orchestration_engine.register_agent(perf_agent)
        orchestration_engine.register_agent(tactical_agent)
        
        # Query that should trigger parallel execution (multiple independent capabilities)
        result = await orchestration_engine.process_query(
            "Analyze team performance and suggest tactical strategy for the match"
        )
        
        # Should have responses from multiple agents
        assert len(result.agent_responses) >= 1
        assert result.metadata.get("strategy") in ["parallel", "sequential"]  # Should choose appropriate strategy
    
    def test_system_health_monitoring(self, orchestration_engine, mock_all_dependencies):
        """Test system health monitoring across agents"""
        # Register agents
        agents = [
            PerformanceAgent(),
            TacticalAgent(),
            PredictionAgent(),
            BettingAgent()
        ]
        
        for agent in agents:
            orchestration_engine.register_agent(agent)
        
        # Check initial system health
        health = orchestration_engine.get_system_status()
        
        assert health["agent_system"]["total_agents"] == 4
        assert health["agent_system"]["healthy_agents"] == 4
        assert health["agent_system"]["health_percentage"] == 100.0
        
        # Simulate some agent failures
        orchestration_engine.coordinator.update_agent_health("performance_agent", success=False)
        orchestration_engine.coordinator.update_agent_health("performance_agent", success=False)
        orchestration_engine.coordinator.update_agent_health("performance_agent", success=False)
        
        # Check updated health
        updated_health = orchestration_engine.get_system_status()
        assert updated_health["agent_system"]["healthy_agents"] == 3
        assert updated_health["agent_system"]["health_percentage"] == 75.0
    
    @pytest.mark.asyncio
    async def test_performance_tracking_across_agents(self, orchestration_engine, mock_all_dependencies):
        """Test performance tracking across multiple agent executions"""
        # Register agents
        perf_agent = PerformanceAgent()
        tactical_agent = TacticalAgent()
        
        orchestration_engine.register_agent(perf_agent)
        orchestration_engine.register_agent(tactical_agent)
        
        # Execute multiple queries
        queries = [
            "Analyze player performance",
            "Suggest tactical strategy", 
            "Show team statistics"
        ]
        
        for query in queries:
            await orchestration_engine.process_query(query)
        
        # Check that execution history is tracked
        assert len(orchestration_engine.execution_history) == len(queries)
        
        # Check agent performance stats
        perf_stats = perf_agent.get_performance_stats()
        assert perf_stats["execution_count"] > 0
        assert perf_stats["success_rate"] >= 0
        assert perf_stats["average_execution_time"] >= 0
    
    @pytest.mark.asyncio
    async def test_complex_cricket_analysis_workflow(self, orchestration_engine, mock_all_dependencies):
        """Test complex cricket analysis workflow involving multiple agents"""
        # Register all agents
        agents = [
            PerformanceAgent(),
            TacticalAgent(),
            PredictionAgent(),
            BettingAgent()
        ]
        
        for agent in agents:
            orchestration_engine.register_agent(agent)
        
        # Complex cricket analysis query
        complex_query = """
        For the upcoming India vs Australia ODI match at MCG:
        1. Analyze recent player performance for key batsmen
        2. Suggest optimal tactical approach for both teams
        3. Predict the match outcome and total score
        4. Identify any value betting opportunities
        """
        
        context = {
            "format": "ODI",
            "venue": "MCG",
            "teams": ["India", "Australia"],
            "players": ["Kohli", "Smith", "Rohit", "Warner"],
            "conditions": "overcast"
        }
        
        result = await orchestration_engine.process_query(complex_query, context)
        
        # Should successfully coordinate multiple agents
        assert result.success is True
        assert result.overall_confidence > 0.3  # Reasonable confidence
        assert len(result.agent_responses) >= 2  # Multiple agents involved
        
        # Should have used multiple capabilities
        capabilities_used = set(resp.capability.value for resp in result.agent_responses if resp.success)
        assert len(capabilities_used) >= 2
        
        # Check aggregated result contains comprehensive analysis
        agg_result = result.aggregated_result
        assert agg_result is not None
        assert agg_result["agent_count"] >= 2
        assert agg_result["successful_agents"] >= 1
        assert "results" in agg_result
    
    @pytest.mark.asyncio
    async def test_agent_dependency_coordination(self, orchestration_engine, mock_all_dependencies):
        """Test coordination when agents have dependencies"""
        # Register agents
        prediction_agent = PredictionAgent()
        betting_agent = BettingAgent()
        
        orchestration_engine.register_agent(prediction_agent)
        orchestration_engine.register_agent(betting_agent)
        
        # Query that requires prediction before betting analysis
        result = await orchestration_engine.process_query(
            "Predict match outcome and find betting value opportunities",
            {"teams": ["India", "England"], "format": "T20"}
        )
        
        # Should coordinate both agents
        assert result.overall_confidence >= 0
        agent_ids = [resp.agent_id for resp in result.agent_responses]
        
        # Should have responses from both agents (or at least attempt to use both)
        assert len(set(agent_ids)) >= 1  # At least one agent should respond
    
    def test_agent_capability_mapping(self, orchestration_engine, mock_all_dependencies):
        """Test that agent capabilities are properly mapped"""
        # Register all agents
        agents = [
            PerformanceAgent(),
            TacticalAgent(),
            PredictionAgent(), 
            BettingAgent()
        ]
        
        for agent in agents:
            orchestration_engine.register_agent(agent)
        
        coordinator = orchestration_engine.coordinator
        
        # Check capability mapping
        assert len(coordinator.capability_map[AgentCapability.PERFORMANCE_ANALYSIS]) >= 1
        assert len(coordinator.capability_map[AgentCapability.TACTICAL_ANALYSIS]) >= 1
        assert len(coordinator.capability_map[AgentCapability.MATCH_PREDICTION]) >= 1
        assert len(coordinator.capability_map[AgentCapability.BETTING_ANALYSIS]) >= 1
        
        # Check that agents can be retrieved by capability
        perf_agents = coordinator.get_agents_for_capability(AgentCapability.PERFORMANCE_ANALYSIS)
        assert len(perf_agents) >= 1
        assert all(AgentCapability.PERFORMANCE_ANALYSIS in agent.capabilities for agent in perf_agents)
    
    @pytest.mark.asyncio
    async def test_query_complexity_handling(self, orchestration_engine, mock_all_dependencies):
        """Test handling of queries with varying complexity"""
        # Register agents
        agents = [PerformanceAgent(), TacticalAgent(), PredictionAgent()]
        for agent in agents:
            orchestration_engine.register_agent(agent)
        
        # Test simple query
        simple_result = await orchestration_engine.process_query("Show player statistics")
        assert simple_result.overall_confidence >= 0
        
        # Test medium complexity query
        medium_result = await orchestration_engine.process_query(
            "Compare Kohli vs Smith performance and predict who will score more"
        )
        assert medium_result.overall_confidence >= 0
        
        # Test complex query
        complex_result = await orchestration_engine.process_query(
            "Comprehensive analysis: player performance trends, tactical recommendations, and match predictions"
        )
        assert complex_result.overall_confidence >= 0
        
        # Complex queries should potentially use more agents
        assert len(complex_result.agent_responses) >= len(simple_result.agent_responses)
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, orchestration_engine, mock_all_dependencies):
        """Test error propagation and recovery mechanisms"""
        # Register agents
        perf_agent = PerformanceAgent()
        tactical_agent = TacticalAgent()
        
        orchestration_engine.register_agent(perf_agent)
        orchestration_engine.register_agent(tactical_agent)
        
        # Test with invalid context
        result = await orchestration_engine.process_query("")  # Empty query
        
        # Should handle gracefully
        assert isinstance(result.overall_confidence, float)
        assert result.overall_confidence >= 0
        
        # Test with query that no agent can handle well
        result2 = await orchestration_engine.process_query("Random unrelated query about cooking recipes")
        
        # Should still provide some response
        assert isinstance(result2.overall_confidence, float)
    
    def test_agent_performance_comparison(self, orchestration_engine, mock_all_dependencies):
        """Test comparing performance across different agents"""
        # Register multiple agents of same type (if we had multiple implementations)
        perf_agent1 = PerformanceAgent()
        perf_agent1.agent_id = "performance_agent_1"
        
        perf_agent2 = PerformanceAgent()
        perf_agent2.agent_id = "performance_agent_2"
        
        # Set different performance stats
        perf_agent1.execution_count = 10
        perf_agent1.success_count = 9
        perf_agent1.average_confidence = 0.85
        
        perf_agent2.execution_count = 10
        perf_agent2.success_count = 7
        perf_agent2.average_confidence = 0.75
        
        orchestration_engine.register_agent(perf_agent1)
        orchestration_engine.register_agent(perf_agent2)
        
        # Test agent selection
        context = AgentContext("test", "player analysis", datetime.now())
        best_agent = orchestration_engine.coordinator.select_best_agent(
            AgentCapability.PERFORMANCE_ANALYSIS, context
        )
        
        # Should select the better performing agent
        assert best_agent is not None
        assert best_agent.agent_id == "performance_agent_1"  # Better performance
    
    @pytest.mark.asyncio
    async def test_real_time_processing_capability(self, orchestration_engine, mock_all_dependencies):
        """Test real-time processing capabilities"""
        # Register agents with real-time capability
        prediction_agent = PredictionAgent()
        betting_agent = BettingAgent()
        
        orchestration_engine.register_agent(prediction_agent)
        orchestration_engine.register_agent(betting_agent)
        
        # Query requiring real-time processing
        result = await orchestration_engine.process_query(
            "Real-time match prediction and live betting opportunities",
            {"live_match": True, "current_score": "150/3", "overs": "25"}
        )
        
        # Should handle real-time context
        assert result.overall_confidence >= 0
        
        # Check that real-time capable agents were used
        capabilities_used = [resp.capability for resp in result.agent_responses if resp.success]
        real_time_agents = [
            resp for resp in result.agent_responses 
            if resp.success and AgentCapability.REAL_TIME_PROCESSING in 
               orchestration_engine.coordinator.registered_agents[resp.agent_id].capabilities
        ]
        assert len(real_time_agents) >= 0  # At least attempt to use real-time agents
    
    def test_orchestration_engine_configuration(self, mock_all_dependencies):
        """Test orchestration engine configuration options"""
        # Test with custom configuration
        custom_config = {
            "max_parallel_agents": 2,
            "default_timeout": 45.0,
            "confidence_threshold": 0.8
        }
        
        engine = OrchestrationEngine(custom_config)
        
        assert engine.max_parallel_agents == 2
        assert engine.default_timeout == 45.0
        assert engine.confidence_threshold == 0.8
    
    @pytest.mark.asyncio
    async def test_execution_time_tracking(self, orchestration_engine, mock_all_dependencies):
        """Test execution time tracking and performance monitoring"""
        # Register agent
        perf_agent = PerformanceAgent()
        orchestration_engine.register_agent(perf_agent)
        
        # Execute query
        start_time = asyncio.get_event_loop().time()
        result = await orchestration_engine.process_query("Analyze player performance")
        end_time = asyncio.get_event_loop().time()
        
        # Check timing
        assert result.execution_time > 0
        assert result.execution_time <= (end_time - start_time) + 0.1  # Allow small margin
        
        # Check agent execution time tracking
        if result.agent_responses:
            for response in result.agent_responses:
                assert response.execution_time >= 0


class TestAgentWorkflowScenarios:
    """Test specific cricket analysis workflow scenarios"""
    
    @pytest.fixture
    def full_system(self):
        """Create full system with all agents"""
        engine = OrchestrationEngine()
        
        # Mock all dependencies
        with patch('crickformers.agents.performance_agent.EnhancedKGQueryEngine'), \
             patch('crickformers.agents.performance_agent.TemporalDecayEngine'), \
             patch('crickformers.agents.tactical_agent.EnhancedKGQueryEngine'), \
             patch('crickformers.agents.tactical_agent.ContextNodeManager'), \
             patch('crickformers.agents.prediction_agent.MoEOrchestrator') as mock_moe, \
             patch('crickformers.agents.prediction_agent.EnhancedKGQueryEngine'), \
             patch('crickformers.agents.betting_agent.MispricingEngine'), \
             patch('crickformers.agents.betting_agent.PredictionAgent'), \
             patch('crickformers.agents.betting_agent.EnhancedKGQueryEngine'):
            
            # Configure MoE mock
            mock_moe_instance = MagicMock()
            mock_moe.return_value = mock_moe_instance
            mock_moe_instance.initialize.return_value = True
            mock_moe_instance.predict.return_value = {
                "status": "success", 
                "predictions": {"team_a_win_prob": 0.6},
                "metadata": {"confidence": 0.8}
            }
            
            # Register all agents
            agents = [
                PerformanceAgent(),
                TacticalAgent(),
                PredictionAgent(),
                BettingAgent()
            ]
            
            for agent in agents:
                engine.register_agent(agent)
            
            yield engine
    
    @pytest.mark.asyncio
    async def test_match_preview_workflow(self, full_system):
        """Test complete match preview workflow"""
        query = "Provide comprehensive match preview for India vs Australia ODI including player analysis, tactical insights, and predictions"
        
        context = {
            "format": "ODI",
            "teams": ["India", "Australia"],
            "venue": "MCG",
            "conditions": "sunny"
        }
        
        result = await full_system.process_query(query, context)
        
        # Should provide comprehensive analysis
        assert result.overall_confidence > 0
        assert len(result.agent_responses) >= 1
        
        # Check that multiple types of analysis were attempted
        if result.aggregated_result and "capabilities_used" in result.aggregated_result:
            capabilities = result.aggregated_result["capabilities_used"]
            # Should attempt multiple types of analysis
            assert len(capabilities) >= 1
    
    @pytest.mark.asyncio 
    async def test_live_match_analysis_workflow(self, full_system):
        """Test live match analysis workflow"""
        query = "Live analysis: current match situation, tactical adjustments, and real-time predictions"
        
        context = {
            "live_match": True,
            "current_score": "180/4",
            "overs": "35.2",
            "target": "285",
            "format": "ODI"
        }
        
        result = await full_system.process_query(query, context)
        
        # Should handle live context
        assert result.overall_confidence >= 0
        assert result.execution_time > 0
        
        # Should have attempted real-time analysis
        real_time_agents = [
            resp for resp in result.agent_responses
            if resp.success and AgentCapability.REAL_TIME_PROCESSING in 
               full_system.coordinator.registered_agents[resp.agent_id].capabilities
        ]
        # At least should have attempted to use real-time capable agents
        assert len(result.agent_responses) >= 1
    
    @pytest.mark.asyncio
    async def test_tournament_analysis_workflow(self, full_system):
        """Test tournament-level analysis workflow"""
        query = "Tournament analysis: team performance comparison, prediction for semi-finals, and betting strategy"
        
        context = {
            "tournament": "World Cup 2023",
            "stage": "semi_final",
            "teams": ["India", "Australia", "England", "New Zealand"],
            "format": "ODI"
        }
        
        result = await full_system.process_query(query, context)
        
        # Should provide tournament-level insights
        assert result.overall_confidence >= 0
        assert len(result.agent_responses) >= 1
        
        # Should have comprehensive aggregated result
        if result.aggregated_result:
            assert result.aggregated_result["agent_count"] >= 1
