#!/usr/bin/env python3
"""
Orchestration Engine Tests
==========================

Unit tests for the agent orchestration engine including coordination,
execution planning, and result aggregation.

Author: WicketWise Team
Last Modified: 2025-08-24
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from crickformers.agents.orchestration_engine import (
    OrchestrationEngine, AgentCoordinator, ExecutionPlan, ExecutionStrategy,
    AgentTask, OrchestrationResult
)
from crickformers.agents.base_agent import (
    BaseAgent, AgentCapability, AgentContext, AgentResponse, AgentPriority
)


class MockTestAgent(BaseAgent):
    """Mock agent for testing orchestration"""
    
    def __init__(self, agent_id: str, capabilities: list, should_fail: bool = False, delay: float = 0.0):
        super().__init__(agent_id, capabilities)
        self.should_fail = should_fail
        self.delay = delay
        self.execute_called = False
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """Mock execute with optional delay and failure"""
        self.execute_called = True
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_fail:
            raise Exception(f"Mock agent {self.agent_id} failed")
        
        return AgentResponse(
            agent_id=self.agent_id,
            capability=self.capabilities[0],
            success=True,
            confidence=0.8,
            execution_time=self.delay,
            result={"agent_result": f"Success from {self.agent_id}"}
        )
    
    def can_handle(self, capability: AgentCapability, context: AgentContext) -> bool:
        """Mock can_handle"""
        return capability in self.capabilities
    
    def _initialize_agent(self) -> bool:
        """Mock initialization"""
        return True


class TestAgentCoordinator:
    """Test AgentCoordinator functionality"""
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization"""
        coordinator = AgentCoordinator()
        
        assert len(coordinator.registered_agents) == 0
        assert len(coordinator.capability_map) == 0
        assert len(coordinator.agent_health) == 0
    
    def test_agent_registration(self):
        """Test agent registration"""
        coordinator = AgentCoordinator()
        agent = MockTestAgent("test_agent", [AgentCapability.PERFORMANCE_ANALYSIS])
        
        success = coordinator.register_agent(agent)
        
        assert success is True
        assert "test_agent" in coordinator.registered_agents
        assert AgentCapability.PERFORMANCE_ANALYSIS in coordinator.capability_map
        assert "test_agent" in coordinator.capability_map[AgentCapability.PERFORMANCE_ANALYSIS]
        assert "test_agent" in coordinator.agent_health
        assert coordinator.agent_health["test_agent"]["status"] == "healthy"
    
    def test_multiple_agent_registration(self):
        """Test registering multiple agents"""
        coordinator = AgentCoordinator()
        
        agent1 = MockTestAgent("agent1", [AgentCapability.PERFORMANCE_ANALYSIS])
        agent2 = MockTestAgent("agent2", [AgentCapability.TACTICAL_ANALYSIS])
        agent3 = MockTestAgent("agent3", [AgentCapability.PERFORMANCE_ANALYSIS, AgentCapability.TREND_ANALYSIS])
        
        assert coordinator.register_agent(agent1) is True
        assert coordinator.register_agent(agent2) is True
        assert coordinator.register_agent(agent3) is True
        
        # Check capability mapping
        perf_agents = coordinator.capability_map[AgentCapability.PERFORMANCE_ANALYSIS]
        assert len(perf_agents) == 2
        assert "agent1" in perf_agents
        assert "agent3" in perf_agents
        
        tactical_agents = coordinator.capability_map[AgentCapability.TACTICAL_ANALYSIS]
        assert len(tactical_agents) == 1
        assert "agent2" in tactical_agents
    
    def test_get_agents_for_capability(self):
        """Test getting agents by capability"""
        coordinator = AgentCoordinator()
        
        agent1 = MockTestAgent("agent1", [AgentCapability.PERFORMANCE_ANALYSIS])
        agent2 = MockTestAgent("agent2", [AgentCapability.PERFORMANCE_ANALYSIS])
        
        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)
        
        agents = coordinator.get_agents_for_capability(AgentCapability.PERFORMANCE_ANALYSIS)
        assert len(agents) == 2
        
        # Test non-existent capability
        agents = coordinator.get_agents_for_capability(AgentCapability.BETTING_ANALYSIS)
        assert len(agents) == 0
    
    def test_best_agent_selection(self):
        """Test selecting best agent for capability"""
        coordinator = AgentCoordinator()
        
        # Agent with better performance
        agent1 = MockTestAgent("agent1", [AgentCapability.PERFORMANCE_ANALYSIS])
        agent1.success_count = 8
        agent1.execution_count = 10
        agent1.average_confidence = 0.9
        
        # Agent with worse performance
        agent2 = MockTestAgent("agent2", [AgentCapability.PERFORMANCE_ANALYSIS])
        agent2.success_count = 5
        agent2.execution_count = 10
        agent2.average_confidence = 0.6
        
        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)
        
        context = AgentContext("test", "test query", datetime.now())
        best_agent = coordinator.select_best_agent(AgentCapability.PERFORMANCE_ANALYSIS, context)
        
        assert best_agent is not None
        assert best_agent.agent_id == "agent1"  # Should select better performing agent
    
    def test_agent_health_tracking(self):
        """Test agent health status tracking"""
        coordinator = AgentCoordinator()
        agent = MockTestAgent("test_agent", [AgentCapability.PERFORMANCE_ANALYSIS])
        coordinator.register_agent(agent)
        
        # Initial health should be healthy
        assert coordinator.agent_health["test_agent"]["status"] == "healthy"
        assert coordinator.agent_health["test_agent"]["consecutive_failures"] == 0
        
        # Record successful execution
        coordinator.update_agent_health("test_agent", success=True)
        assert coordinator.agent_health["test_agent"]["status"] == "healthy"
        assert coordinator.agent_health["test_agent"]["consecutive_failures"] == 0
        
        # Record failures
        coordinator.update_agent_health("test_agent", success=False)
        coordinator.update_agent_health("test_agent", success=False)
        coordinator.update_agent_health("test_agent", success=False)
        
        # Should be marked unhealthy after 3 failures
        assert coordinator.agent_health["test_agent"]["status"] == "unhealthy"
        assert coordinator.agent_health["test_agent"]["consecutive_failures"] == 3
    
    def test_system_health_status(self):
        """Test system health status reporting"""
        coordinator = AgentCoordinator()
        
        agent1 = MockTestAgent("agent1", [AgentCapability.PERFORMANCE_ANALYSIS])
        agent2 = MockTestAgent("agent2", [AgentCapability.TACTICAL_ANALYSIS])
        
        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)
        
        # Mark one agent as unhealthy
        coordinator.update_agent_health("agent1", success=False)
        coordinator.update_agent_health("agent1", success=False)
        coordinator.update_agent_health("agent1", success=False)
        
        health = coordinator.get_system_health()
        
        assert health["total_agents"] == 2
        assert health["healthy_agents"] == 1
        assert health["health_percentage"] == 50.0
        assert "agent_details" in health


class TestExecutionPlan:
    """Test ExecutionPlan functionality"""
    
    def test_execution_plan_creation(self):
        """Test execution plan creation"""
        context = AgentContext("test", "test query", datetime.now())
        task = AgentTask(
            agent_id="test_agent",
            capability=AgentCapability.PERFORMANCE_ANALYSIS,
            priority=AgentPriority.HIGH,
            context=context
        )
        
        plan = ExecutionPlan(
            plan_id="plan_123",
            original_query="test query",
            strategy=ExecutionStrategy.PARALLEL,
            tasks=[task],
            estimated_duration=10.0,
            confidence_threshold=0.7
        )
        
        assert plan.plan_id == "plan_123"
        assert plan.original_query == "test query"
        assert plan.strategy == ExecutionStrategy.PARALLEL
        assert len(plan.tasks) == 1
        assert plan.estimated_duration == 10.0
    
    def test_critical_tasks_identification(self):
        """Test identifying critical tasks"""
        context = AgentContext("test", "test query", datetime.now())
        
        critical_task = AgentTask("agent1", AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.CRITICAL, context)
        high_task = AgentTask("agent2", AgentCapability.TACTICAL_ANALYSIS, AgentPriority.HIGH, context)
        medium_task = AgentTask("agent3", AgentCapability.TREND_ANALYSIS, AgentPriority.MEDIUM, context)
        
        plan = ExecutionPlan(
            plan_id="plan_123",
            original_query="test",
            strategy=ExecutionStrategy.PARALLEL,
            tasks=[critical_task, high_task, medium_task],
            estimated_duration=10.0,
            confidence_threshold=0.7
        )
        
        critical_tasks = plan.get_critical_tasks()
        assert len(critical_tasks) == 1
        assert critical_tasks[0].agent_id == "agent1"
    
    def test_parallel_tasks_identification(self):
        """Test identifying parallel tasks"""
        context = AgentContext("test", "test query", datetime.now())
        
        parallel_task1 = AgentTask("agent1", AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.HIGH, context)
        parallel_task2 = AgentTask("agent2", AgentCapability.TACTICAL_ANALYSIS, AgentPriority.HIGH, context)
        dependent_task = AgentTask("agent3", AgentCapability.TREND_ANALYSIS, AgentPriority.HIGH, context, dependencies=["agent1"])
        
        plan = ExecutionPlan(
            plan_id="plan_123",
            original_query="test",
            strategy=ExecutionStrategy.PARALLEL,
            tasks=[parallel_task1, parallel_task2, dependent_task],
            estimated_duration=10.0,
            confidence_threshold=0.7
        )
        
        parallel_tasks = plan.get_parallel_tasks()
        assert len(parallel_tasks) == 2  # Two tasks without dependencies
        assert all(task.agent_id in ["agent1", "agent2"] for task in parallel_tasks)


class TestOrchestrationEngine:
    """Test OrchestrationEngine functionality"""
    
    def test_engine_initialization(self):
        """Test orchestration engine initialization"""
        config = {
            "max_parallel_agents": 3,
            "default_timeout": 45.0,
            "confidence_threshold": 0.8
        }
        
        engine = OrchestrationEngine(config)
        
        assert engine.max_parallel_agents == 3
        assert engine.default_timeout == 45.0
        assert engine.confidence_threshold == 0.8
        assert len(engine.execution_history) == 0
        assert len(engine.active_executions) == 0
    
    def test_agent_registration_with_engine(self):
        """Test agent registration with orchestration engine"""
        engine = OrchestrationEngine()
        agent = MockTestAgent("test_agent", [AgentCapability.PERFORMANCE_ANALYSIS])
        
        success = engine.register_agent(agent)
        
        assert success is True
        assert "test_agent" in engine.coordinator.registered_agents
    
    def test_query_capability_analysis(self):
        """Test query analysis for capability identification"""
        engine = OrchestrationEngine()
        
        # Performance analysis query
        perf_caps = engine._analyze_query_capabilities("Show me Kohli's batting performance statistics")
        assert any(cap[0] == AgentCapability.PERFORMANCE_ANALYSIS for cap in perf_caps)
        
        # Prediction query
        pred_caps = engine._analyze_query_capabilities("Predict the winner of India vs Australia match")
        assert any(cap[0] == AgentCapability.MATCH_PREDICTION for cap in pred_caps)
        
        # Betting query
        betting_caps = engine._analyze_query_capabilities("Find value betting opportunities in today's match")
        assert any(cap[0] == AgentCapability.BETTING_ANALYSIS for cap in betting_caps)
        
        # Tactical query
        tactical_caps = engine._analyze_query_capabilities("What field placement strategy should be used?")
        assert any(cap[0] == AgentCapability.TACTICAL_ANALYSIS for cap in tactical_caps)
    
    def test_execution_strategy_determination(self):
        """Test execution strategy determination"""
        engine = OrchestrationEngine()
        context = AgentContext("test", "test query", datetime.now())
        
        # Tasks without dependencies and with critical priority should use parallel strategy
        parallel_tasks = [
            AgentTask("agent1", AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.CRITICAL, context),
            AgentTask("agent2", AgentCapability.TACTICAL_ANALYSIS, AgentPriority.CRITICAL, context)
        ]
        strategy = engine._determine_execution_strategy(parallel_tasks)
        assert strategy == ExecutionStrategy.PARALLEL
        
        # Tasks with dependencies should use pipeline strategy
        pipeline_tasks = [
            AgentTask("agent1", AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.HIGH, context),
            AgentTask("agent2", AgentCapability.TACTICAL_ANALYSIS, AgentPriority.HIGH, context, dependencies=["agent1"])
        ]
        strategy = engine._determine_execution_strategy(pipeline_tasks)
        assert strategy == ExecutionStrategy.PIPELINE
    
    @pytest.mark.asyncio
    async def test_successful_query_processing(self):
        """Test successful query processing"""
        engine = OrchestrationEngine()
        
        # Register mock agents
        perf_agent = MockTestAgent("perf_agent", [AgentCapability.PERFORMANCE_ANALYSIS])
        tactical_agent = MockTestAgent("tactical_agent", [AgentCapability.TACTICAL_ANALYSIS])
        
        engine.register_agent(perf_agent)
        engine.register_agent(tactical_agent)
        
        # Process query
        result = await engine.process_query(
            "Analyze Kohli's performance and suggest tactical approaches",
            {"format": "ODI", "venue": "Wankhede"}
        )
        
        assert result.success is True
        assert result.overall_confidence > 0
        assert len(result.agent_responses) > 0
        assert result.aggregated_result is not None
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution of agents"""
        engine = OrchestrationEngine()
        context = AgentContext("test", "test query", datetime.now())
        
        # Create tasks with different delays
        tasks = [
            AgentTask("agent1", AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.HIGH, context),
            AgentTask("agent2", AgentCapability.TACTICAL_ANALYSIS, AgentPriority.HIGH, context),
            AgentTask("agent3", AgentCapability.TREND_ANALYSIS, AgentPriority.HIGH, context)
        ]
        
        # Register agents with delays
        for i, task in enumerate(tasks):
            agent = MockTestAgent(task.agent_id, [task.capability], delay=0.1 * (i + 1))
            engine.register_agent(agent)
        
        # Execute in parallel
        start_time = asyncio.get_event_loop().time()
        responses = await engine._execute_parallel(tasks)
        end_time = asyncio.get_event_loop().time()
        
        # Should complete in roughly the time of the slowest agent (0.3s)
        # rather than the sum of all agents (0.6s)
        execution_time = end_time - start_time
        assert execution_time < 0.5  # Should be much less than sequential time
        assert len(responses) == 3
        assert all(response.success for response in responses)
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test sequential execution of agents"""
        engine = OrchestrationEngine()
        context = AgentContext("test", "test query", datetime.now())
        
        tasks = [
            AgentTask("agent1", AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.HIGH, context),
            AgentTask("agent2", AgentCapability.TACTICAL_ANALYSIS, AgentPriority.HIGH, context)
        ]
        
        # Register agents
        for task in tasks:
            agent = MockTestAgent(task.agent_id, [task.capability])
            engine.register_agent(agent)
        
        responses = await engine._execute_sequential(tasks)
        
        assert len(responses) == 2
        assert all(response.success for response in responses)
    
    @pytest.mark.asyncio
    async def test_critical_task_failure_handling(self):
        """Test handling of critical task failures"""
        engine = OrchestrationEngine()
        context = AgentContext("test", "test query", datetime.now())
        
        # Critical task that will fail
        critical_task = AgentTask("failing_agent", AgentCapability.PERFORMANCE_ANALYSIS, AgentPriority.CRITICAL, context)
        normal_task = AgentTask("normal_agent", AgentCapability.TACTICAL_ANALYSIS, AgentPriority.HIGH, context)
        
        # Register agents
        failing_agent = MockTestAgent("failing_agent", [AgentCapability.PERFORMANCE_ANALYSIS], should_fail=True)
        normal_agent = MockTestAgent("normal_agent", [AgentCapability.TACTICAL_ANALYSIS])
        
        engine.register_agent(failing_agent)
        engine.register_agent(normal_agent)
        
        # Execute sequential (so critical failure stops execution)
        responses = await engine._execute_sequential([critical_task, normal_task])
        
        # Should have response from failing agent, but not from normal agent
        assert len(responses) == 1
        assert responses[0].success is False
        assert "failed" in responses[0].error_message
    
    def test_result_aggregation(self):
        """Test result aggregation from multiple agents"""
        engine = OrchestrationEngine()
        
        responses = [
            AgentResponse("agent1", AgentCapability.PERFORMANCE_ANALYSIS, True, 0.8, 1.0, {"result": "perf_data"}),
            AgentResponse("agent2", AgentCapability.TACTICAL_ANALYSIS, True, 0.7, 1.5, {"result": "tactical_data"}),
            AgentResponse("agent3", AgentCapability.TREND_ANALYSIS, False, 0.0, 0.5, error_message="Failed")
        ]
        
        aggregated = engine._aggregate_results(responses)
        
        assert aggregated["agent_count"] == 3
        assert aggregated["successful_agents"] == 2
        assert len(aggregated["capabilities_used"]) == 2
        assert "performance_analysis" in aggregated["capabilities_used"]
        assert "tactical_analysis" in aggregated["capabilities_used"]
        assert "results" in aggregated
    
    def test_confidence_calculation(self):
        """Test overall confidence calculation"""
        engine = OrchestrationEngine()
        
        responses = [
            AgentResponse("agent1", AgentCapability.PERFORMANCE_ANALYSIS, True, 0.8, 1.0),
            AgentResponse("agent2", AgentCapability.TACTICAL_ANALYSIS, True, 0.9, 1.5),
            AgentResponse("agent3", AgentCapability.TREND_ANALYSIS, False, 0.0, 0.5)
        ]
        
        confidence = engine._calculate_overall_confidence(responses)
        
        # Should be average of successful responses: (0.8 + 0.9) / 2 = 0.85
        assert abs(confidence - 0.85) < 1e-10
    
    def test_system_status_reporting(self):
        """Test system status reporting"""
        engine = OrchestrationEngine()
        
        # Register some agents
        agent1 = MockTestAgent("agent1", [AgentCapability.PERFORMANCE_ANALYSIS])
        agent2 = MockTestAgent("agent2", [AgentCapability.TACTICAL_ANALYSIS])
        
        engine.register_agent(agent1)
        engine.register_agent(agent2)
        
        status = engine.get_system_status()
        
        assert "orchestration_engine" in status
        assert "agent_system" in status
        assert "recent_performance" in status
        
        assert status["orchestration_engine"]["active_executions"] == 0
        assert status["agent_system"]["total_agents"] == 2
    
    @pytest.mark.asyncio
    async def test_execution_history_tracking(self):
        """Test execution history tracking"""
        engine = OrchestrationEngine()
        
        # Register agent
        agent = MockTestAgent("test_agent", [AgentCapability.PERFORMANCE_ANALYSIS])
        engine.register_agent(agent)
        
        # Process multiple queries
        for i in range(3):
            result = await engine.process_query(f"Query {i}")
            
        assert len(engine.execution_history) == 3
        assert all(isinstance(result, OrchestrationResult) for result in engine.execution_history)
    
    @pytest.mark.asyncio
    async def test_execution_history_limit(self):
        """Test execution history size limit"""
        engine = OrchestrationEngine()
        agent = MockTestAgent("test_agent", [AgentCapability.PERFORMANCE_ANALYSIS])
        engine.register_agent(agent)
        
        # Simulate many executions by directly adding to history
        for i in range(150):  # More than the 100 limit
            fake_result = OrchestrationResult(
                plan_id=f"plan_{i}",
                success=True,
                overall_confidence=0.8,
                execution_time=1.0,
                agent_responses=[]
            )
            engine.execution_history.append(fake_result)
        
        # Process one more query to trigger cleanup
        await engine.process_query("Test query")
        
        # History should be limited to 100
        assert len(engine.execution_history) <= 100
