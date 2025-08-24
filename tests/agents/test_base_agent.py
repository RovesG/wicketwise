#!/usr/bin/env python3
"""
Base Agent Tests
===============

Unit tests for the base agent architecture including abstract classes,
response formats, and common functionality.

Author: WicketWise Team
Last Modified: 2025-08-24
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from crickformers.agents.base_agent import (
    BaseAgent, AgentCapability, AgentContext, AgentResponse, AgentPriority
)


class TestAgentContext:
    """Test AgentContext data structure"""
    
    def test_basic_context_creation(self):
        """Test basic context creation"""
        context = AgentContext(
            request_id="test_123",
            user_query="Test query",
            timestamp=datetime.now()
        )
        
        assert context.request_id == "test_123"
        assert context.user_query == "Test query"
        assert isinstance(context.timestamp, datetime)
        assert context.confidence_threshold == 0.7
        assert context.max_execution_time == 30.0
    
    def test_context_with_cricket_context(self):
        """Test context with cricket-specific parameters"""
        match_context = {"team_a": "India", "team_b": "Australia", "venue": "MCG"}
        player_context = {"names": ["Kohli", "Smith"], "roles": ["batsman", "batsman"]}
        
        context = AgentContext(
            request_id="cricket_123",
            user_query="Compare Kohli vs Smith",
            timestamp=datetime.now(),
            match_context=match_context,
            player_context=player_context,
            format_context="Test"
        )
        
        assert context.match_context == match_context
        assert context.player_context == player_context
        assert context.format_context == "Test"


class TestAgentResponse:
    """Test AgentResponse data structure"""
    
    def test_successful_response_creation(self):
        """Test creating successful response"""
        response = AgentResponse(
            agent_id="test_agent",
            capability=AgentCapability.PERFORMANCE_ANALYSIS,
            success=True,
            confidence=0.8,
            execution_time=1.5,
            result={"analysis": "test result"}
        )
        
        assert response.agent_id == "test_agent"
        assert response.capability == AgentCapability.PERFORMANCE_ANALYSIS
        assert response.success is True
        assert response.confidence == 0.8
        assert response.execution_time == 1.5
        assert response.result == {"analysis": "test result"}
        assert response.error_message is None
    
    def test_error_response_creation(self):
        """Test creating error response"""
        response = AgentResponse(
            agent_id="test_agent",
            capability=AgentCapability.PERFORMANCE_ANALYSIS,
            success=False,
            confidence=0.0,
            execution_time=0.5,
            error_message="Test error occurred"
        )
        
        assert response.success is False
        assert response.confidence == 0.0
        assert response.error_message == "Test error occurred"
        assert response.result is None
    
    def test_response_to_dict(self):
        """Test converting response to dictionary"""
        response = AgentResponse(
            agent_id="test_agent",
            capability=AgentCapability.PERFORMANCE_ANALYSIS,
            success=True,
            confidence=0.8,
            execution_time=1.5,
            result={"test": "data"},
            dependencies_used=["kg", "moe"]
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["agent_id"] == "test_agent"
        assert response_dict["capability"] == "performance_analysis"
        assert response_dict["success"] is True
        assert response_dict["confidence"] == 0.8
        assert response_dict["dependencies_used"] == ["kg", "moe"]


class MockAgent(BaseAgent):
    """Mock agent implementation for testing"""
    
    def __init__(self, agent_id: str = "mock_agent", should_fail: bool = False):
        super().__init__(
            agent_id=agent_id,
            capabilities=[AgentCapability.PERFORMANCE_ANALYSIS, AgentCapability.TREND_ANALYSIS]
        )
        self.should_fail = should_fail
        self.execute_called = False
        self.can_handle_called = False
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """Mock execute implementation"""
        self.execute_called = True
        
        if self.should_fail:
            raise Exception("Mock agent execution failed")
        
        return AgentResponse(
            agent_id=self.agent_id,
            capability=AgentCapability.PERFORMANCE_ANALYSIS,
            success=True,
            confidence=0.8,
            execution_time=0.0,
            result={"mock_result": "success"}
        )
    
    def can_handle(self, capability: AgentCapability, context: AgentContext) -> bool:
        """Mock can_handle implementation"""
        self.can_handle_called = True
        return capability in self.capabilities


class TestBaseAgent:
    """Test BaseAgent abstract class functionality"""
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = MockAgent("test_agent")
        
        assert agent.agent_id == "test_agent"
        assert len(agent.capabilities) == 2
        assert AgentCapability.PERFORMANCE_ANALYSIS in agent.capabilities
        assert AgentCapability.TREND_ANALYSIS in agent.capabilities
        assert agent.execution_count == 0
        assert agent.success_count == 0
        assert agent.is_initialized is False
    
    def test_agent_initialization_with_config(self):
        """Test agent initialization with configuration"""
        config = {
            "max_execution_time": 60.0,
            "min_confidence_threshold": 0.6,
            "retry_attempts": 3
        }
        
        agent = MockAgent("configured_agent")
        agent.config = config
        agent.__init__("configured_agent")
        
        assert agent.max_execution_time == 30.0  # Default value
        assert agent.min_confidence_threshold == 0.5  # Default value
        assert agent.retry_attempts == 2  # Default value
    
    def test_agent_can_handle(self):
        """Test can_handle method"""
        agent = MockAgent()
        context = AgentContext("test", "test query", datetime.now())
        
        # Should handle capabilities it supports
        assert agent.can_handle(AgentCapability.PERFORMANCE_ANALYSIS, context) is True
        assert agent.can_handle(AgentCapability.TREND_ANALYSIS, context) is True
        assert agent.can_handle_called is True
        
        # Should not handle capabilities it doesn't support
        assert agent.can_handle(AgentCapability.BETTING_ANALYSIS, context) is False
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful agent execution"""
        agent = MockAgent()
        agent.initialize()
        
        context = AgentContext("test", "test query", datetime.now())
        response = await agent._execute_with_monitoring(context)
        
        assert response.success is True
        assert response.agent_id == "mock_agent"
        assert response.confidence == 0.8
        assert response.result == {"mock_result": "success"}
        assert agent.execute_called is True
        assert agent.execution_count == 1
        assert agent.success_count == 1
    
    @pytest.mark.asyncio
    async def test_failed_execution(self):
        """Test failed agent execution"""
        agent = MockAgent(should_fail=True)
        agent.initialize()
        
        context = AgentContext("test", "test query", datetime.now())
        response = await agent._execute_with_monitoring(context)
        
        assert response.success is False
        assert "Mock agent execution failed" in response.error_message
        assert agent.execution_count == 1
        assert agent.success_count == 0
    
    @pytest.mark.asyncio
    async def test_execution_without_initialization(self):
        """Test execution when agent not initialized"""
        agent = MockAgent()
        # Don't initialize
        
        context = AgentContext("test", "test query", datetime.now())
        response = await agent._execute_with_monitoring(context)
        
        # Should auto-initialize and succeed
        assert response.success is True
        assert agent.is_initialized is True
    
    def test_performance_stats_tracking(self):
        """Test performance statistics tracking"""
        agent = MockAgent()
        
        # Initial stats
        stats = agent.get_performance_stats()
        assert stats["execution_count"] == 0
        assert stats["success_count"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["average_confidence"] == 0.0
        
        # Update confidence tracking
        agent.success_count = 1
        agent._update_confidence_tracking(0.8)
        agent.success_count = 2
        agent._update_confidence_tracking(0.9)
        
        # Check updated stats
        stats = agent.get_performance_stats()
        assert stats["success_count"] == 2
        assert 0.8 < stats["average_confidence"] < 0.9  # Should be between the two values
    
    def test_performance_stats_reset(self):
        """Test resetting performance statistics"""
        agent = MockAgent()
        
        # Set some stats
        agent.execution_count = 5
        agent.success_count = 3
        agent.total_execution_time = 10.0
        agent.average_confidence = 0.7
        
        # Reset stats
        agent.reset_performance_stats()
        
        # Check reset
        assert agent.execution_count == 0
        assert agent.success_count == 0
        assert agent.total_execution_time == 0.0
        assert agent.average_confidence == 0.0
    
    def test_context_validation(self):
        """Test context validation"""
        agent = MockAgent()
        
        # Valid context
        valid_context = AgentContext("test", "test query", datetime.now())
        assert agent._validate_context(valid_context) is True
        
        # Invalid context - empty query
        invalid_context = AgentContext("test", "", datetime.now())
        assert agent._validate_context(invalid_context) is False
        
        # Invalid context - negative execution time
        invalid_context2 = AgentContext("test", "query", datetime.now(), max_execution_time=-1)
        assert agent._validate_context(invalid_context2) is False
    
    def test_error_response_creation(self):
        """Test error response creation"""
        agent = MockAgent()
        context = AgentContext("test", "test query", datetime.now())
        
        error_response = agent._create_error_response(context, "Test error", 0.0)
        
        assert error_response.success is False
        assert error_response.error_message == "Test error"
        assert error_response.agent_id == "mock_agent"
        assert error_response.confidence == 0.0
    
    def test_agent_string_representations(self):
        """Test string representations of agent"""
        agent = MockAgent("test_agent")
        
        str_repr = str(agent)
        assert "test_agent" in str_repr
        assert "capabilities=2" in str_repr
        assert "initialized=False" in str_repr
        
        repr_str = repr(agent)
        assert "BaseAgent" in repr_str
        assert "test_agent" in repr_str


class TestAgentCapabilityEnum:
    """Test AgentCapability enumeration"""
    
    def test_capability_values(self):
        """Test capability enum values"""
        assert AgentCapability.PERFORMANCE_ANALYSIS.value == "performance_analysis"
        assert AgentCapability.TACTICAL_ANALYSIS.value == "tactical_analysis"
        assert AgentCapability.MATCH_PREDICTION.value == "match_prediction"
        assert AgentCapability.BETTING_ANALYSIS.value == "betting_analysis"
        assert AgentCapability.DATA_RETRIEVAL.value == "data_retrieval"
        assert AgentCapability.TREND_ANALYSIS.value == "trend_analysis"
        assert AgentCapability.COMPARISON_ANALYSIS.value == "comparison_analysis"
        assert AgentCapability.CONTEXTUAL_REASONING.value == "contextual_reasoning"
        assert AgentCapability.MULTI_FORMAT_ANALYSIS.value == "multi_format_analysis"
        assert AgentCapability.REAL_TIME_PROCESSING.value == "real_time_processing"
    
    def test_capability_uniqueness(self):
        """Test that all capability values are unique"""
        values = [capability.value for capability in AgentCapability]
        assert len(values) == len(set(values))


class TestAgentPriorityEnum:
    """Test AgentPriority enumeration"""
    
    def test_priority_values(self):
        """Test priority enum values"""
        assert AgentPriority.CRITICAL.value == 1
        assert AgentPriority.HIGH.value == 2
        assert AgentPriority.MEDIUM.value == 3
        assert AgentPriority.LOW.value == 4
    
    def test_priority_ordering(self):
        """Test priority ordering (lower number = higher priority)"""
        priorities = [p.value for p in AgentPriority]
        assert priorities == sorted(priorities)  # Should be in ascending order


@pytest.mark.asyncio
class TestAsyncAgentBehavior:
    """Test asynchronous agent behavior"""
    
    async def test_concurrent_execution(self):
        """Test concurrent agent execution"""
        agents = [MockAgent(f"agent_{i}") for i in range(3)]
        
        for agent in agents:
            agent.initialize()
        
        contexts = [
            AgentContext(f"req_{i}", f"query_{i}", datetime.now())
            for i in range(3)
        ]
        
        # Execute all agents concurrently
        tasks = [
            agent._execute_with_monitoring(context)
            for agent, context in zip(agents, contexts)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(response.success for response in responses)
        assert len(responses) == 3
        
        # Each agent should have been called
        assert all(agent.execute_called for agent in agents)
    
    async def test_timeout_behavior(self):
        """Test agent timeout behavior"""
        # This would require more complex mocking to test actual timeouts
        # For now, just test that execution completes normally
        agent = MockAgent()
        agent.initialize()
        
        context = AgentContext("test", "test query", datetime.now(), max_execution_time=0.1)
        response = await agent._execute_with_monitoring(context)
        
        assert response.success is True
        assert response.execution_time >= 0
